"""Real-page structural smoke for panel, character, balloon, and speaker layers."""

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

from app.character_perception import CharacterPerceptionService
from app.layout_grouping import PageLayoutGrouper
from app.models.balloon_crops import BalloonCropStore
from app.models.character_crops import CharacterCropStore
from app.models.ctd import CTDPageLocalizer, load_ctd_detector
from app.models.manga_layout import load_manga109_balloon_provider
from app.schemas.page import PageRepresentation
from app.speaker_grounding import SpeakerResolver

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one development page through character perception and speaker grounding."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--sequence-id", default=DEFAULT_SEQUENCE)
    parser.add_argument("--page-index", type=int, default=0)
    parser.add_argument("--ctd-model", type=Path, default=DEFAULT_CTD_MODEL)
    parser.add_argument("--layout-weights", type=Path, default=DEFAULT_LAYOUT_WEIGHTS)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "story-ai-character-speaker-smoke",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read development page: {args.image}")
    height, width = image.shape[:2]

    regions = CTDPageLocalizer(
        load_ctd_detector(args.ctd_model, device="cpu")
    ).localize(args.image)
    layout_provider = load_manga109_balloon_provider(args.layout_weights)
    page = PageRepresentation(
        page_index=args.page_index,
        image_path=str(args.image),
        text_regions=regions,
    )
    layout_result = PageLayoutGrouper(
        panel_provider=layout_provider,
        balloon_provider=layout_provider,
    ).group_page(
        page,
        args.sequence_id,
        args.image,
        page_size=(width, height),
    )

    # No suitable licensed manga character detector is installed in this workspace.
    character_result = CharacterPerceptionService(
        provider=None,
        crop_store=CharacterCropStore(args.artifact_root / "characters"),
    ).process_page(layout_result.page, args.sequence_id, args.image)
    for balloon in character_result.page.balloons:
        BalloonCropStore(args.artifact_root / "balloons").create(
            args.image,
            args.sequence_id,
            args.page_index,
            balloon,
        )
    speaker_result = SpeakerResolver().resolve_page(
        character_result.page,
        page_size=(width, height),
    )
    decisions = {decision.balloon_id: decision for decision in speaker_result.decisions}
    report = {
        "page": f"{args.sequence_id} / page_{args.page_index + 1:02d}",
        "panels": [panel.model_dump(mode="json") for panel in speaker_result.page.panels],
        "character_instances": [
            character.model_dump(mode="json") for character in speaker_result.page.characters
        ],
        "balloons": [
            {
                "balloon_id": balloon.id,
                "ctd_region_ids": balloon.text_region_ids,
                "candidate_speakers": balloon.candidate_character_ids,
                "selected_speaker": decisions[balloon.id].selected_character_instance_id,
                "confidence": decisions[balloon.id].confidence,
                "method": decisions[balloon.id].method,
            }
            for balloon in speaker_result.page.balloons
        ],
        "character_diagnostics": [
            item.model_dump(mode="json") for item in character_result.diagnostics
        ],
        "speaker_diagnostics": [
            item.model_dump(mode="json") for item in speaker_result.diagnostics
        ],
        "artifact_root": str(args.artifact_root),
    }
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
