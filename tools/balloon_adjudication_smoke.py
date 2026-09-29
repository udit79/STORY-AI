"""One real local grouped-balloon -> Candidate Bank -> Qwen adjudication smoke."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.balloon_adjudication import Qwen3VLBalloonAdjudicator, adjudicate_balloon
from app.candidate_bank import produce_candidate_bank
from app.layout_grouping import PageLayoutGrouper
from app.models.balloon_crops import BalloonCropStore
from app.models.candidate_adapters import (
    PaddleOCRCandidateAdapter,
    Qwen3VLCandidateAdapter,
    candidate_producers,
)
from app.models.crops import CTDCropStore
from app.models.ctd import CTDPageLocalizer, load_ctd_detector
from app.models.local_runners import load_paddle_recognizer, load_qwen3vl_runner
from app.models.manga_layout import load_manga109_balloon_provider
from app.schemas.page import PageRepresentation

DEFAULT_SEQUENCE = "seq_952f154fb1505883"
DEFAULT_IMAGE = (
    ROOT / "dataset" / "development" / "images" / DEFAULT_SEQUENCE / "01.png"
)
DEFAULT_CTD_MODEL = (
    ROOT / "vendor" / "comic-text-detector" / "data" / "comictextdetector.pt.onnx"
)
DEFAULT_LAYOUT_WEIGHTS = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--huyvux3005--manga109-segmentation-bubble"
    / "snapshots"
    / "f9a4108c4955136a810e5e92207972f3fb3a65fd"
    / "best.pt"
)
DEFAULT_PADDLE_MODEL_DIR = (
    Path.home() / ".paddlex" / "official_models" / "en_PP-OCRv5_mobile_rec_onnx"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one real local balloon-image adjudication smoke; no metrics are computed."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--sequence-id", default=DEFAULT_SEQUENCE)
    parser.add_argument("--page-index", type=int, default=0)
    parser.add_argument("--ctd-model", type=Path, default=DEFAULT_CTD_MODEL)
    parser.add_argument("--layout-weights", type=Path, default=DEFAULT_LAYOUT_WEIGHTS)
    parser.add_argument("--paddle-model-dir", type=Path, default=DEFAULT_PADDLE_MODEL_DIR)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "story-ai-balloon-adjudication-smoke",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Development page not found: {args.image}")
    page_image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if page_image is None:
        raise OSError(f"Could not read page image: {args.image}")
    height, width = page_image.shape[:2]

    localizer = CTDPageLocalizer(load_ctd_detector(args.ctd_model, device="cpu"))
    layout_provider = load_manga109_balloon_provider(args.layout_weights, image_size=1600)
    regions = localizer.localize(args.image)
    page = PageRepresentation(
        page_index=args.page_index,
        image_path=str(args.image),
        text_regions=regions,
    )
    layout = PageLayoutGrouper(
        panel_provider=layout_provider,
        balloon_provider=layout_provider,
    ).group_page(
        page,
        args.sequence_id,
        args.image,
        page_size=(width, height),
    )
    multi_fragment_balloons = [
        balloon for balloon in layout.page.balloons if len(balloon.text_region_ids) > 1
    ]
    if not multi_fragment_balloons:
        raise RuntimeError(
            "The selected real page has no grouped balloon with multiple CTD fragments; "
            "choose a known multi-fragment development page."
        )
    balloon = max(multi_fragment_balloons, key=lambda item: len(item.text_region_ids))
    regions_by_id = {region.id: region for region in layout.page.text_regions}
    balloon_regions = [regions_by_id[region_id] for region_id in balloon.text_region_ids]

    balloon_crop = BalloonCropStore(args.artifact_root / "balloons").create(
        args.image,
        args.sequence_id,
        args.page_index,
        balloon,
    )
    region_crops = CTDCropStore(args.artifact_root / "regions").generate(
        args.image,
        args.sequence_id,
        args.page_index,
        balloon_regions,
    )
    failed_crops = set(region_crops.failures)
    if failed_crops:
        raise RuntimeError(f"Could not crop balloon CTD fragments: {sorted(failed_crops)}")
    crops_for_bank = {
        region_id: region_crops.crop_paths[region_id]
        for region_id in balloon.text_region_ids
    }

    paddle = load_paddle_recognizer(
        args.paddle_model_dir,
        model_name="en_PP-OCRv5_mobile_rec",
        engine="onnxruntime",
        device="cpu",
    )
    qwen_runner = load_qwen3vl_runner(local_files_only=True, quantize_4bit=True)
    producers = candidate_producers(
        paddle=PaddleOCRCandidateAdapter(paddle),
        qwen3_vl=Qwen3VLCandidateAdapter(qwen_runner),
    )
    candidate_bank = produce_candidate_bank(balloon_regions, crops_for_bank, producers)

    adjudicator = Qwen3VLBalloonAdjudicator(qwen_runner)
    result = adjudicate_balloon(
        adjudicator,
        balloon,
        candidate_bank,
        layout.page.text_regions,
        balloon_crop,
    )
    candidate_texts = [
        {
            "candidate_id": candidate.candidate_id,
            "source": candidate.source,
            "region_id": candidate.region_id,
            "text": candidate.text,
        }
        for group in candidate_bank.groups
        for candidate in group.candidates
    ]
    print(json.dumps({
        "balloon_id": balloon.id,
        "ctd_fragment_count": len(balloon.text_region_ids),
        "candidate_count": len(candidate_texts),
        "candidates": candidate_texts,
        "adjudicator": {
            "final_text": result.final_text,
            "text_type": result.text_type,
            "include_in_story": result.include_in_story,
            "confidence": result.confidence,
            "corrections": [item.model_dump(mode="json") for item in result.corrections],
            "notes": result.notes,
            "diagnostics": [item.model_dump(mode="json") for item in result.diagnostics],
        },
        "artifact_root": str(args.artifact_root),
    }, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
