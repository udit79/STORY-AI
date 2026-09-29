"""One structural, local-only CTD-to-Paddle page-perception smoke path."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models.candidate_adapters import PaddleOCRCandidateAdapter
from app.models.crops import CTDCropStore
from app.models.ctd import CTDPageLocalizer, load_ctd_detector
from app.models.local_runners import load_paddle_recognizer
from app.page_perception import PagePerceptionService

DEFAULT_SEQUENCE = "seq_2032620aa4e4ac7f"
DEFAULT_IMAGE = (
    ROOT / "dataset" / "development" / "images" / DEFAULT_SEQUENCE / "01.png"
)
DEFAULT_CTD_MODEL = (
    ROOT / "vendor" / "comic-text-detector" / "data" / "comictextdetector.pt.onnx"
)
DEFAULT_PADDLE_MODEL_DIR = (
    Path.home() / ".paddlex" / "official_models" / "en_PP-OCRv5_mobile_rec_onnx"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one local CTD -> crop -> PaddleOCR -> CandidateBank smoke path."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--sequence-id", default=DEFAULT_SEQUENCE)
    parser.add_argument("--page-index", type=int, default=0)
    parser.add_argument("--ctd-model", type=Path, default=DEFAULT_CTD_MODEL)
    parser.add_argument("--paddle-model-dir", type=Path, default=DEFAULT_PADDLE_MODEL_DIR)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "story-ai-page-perception-smoke",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Development page not found: {args.image}")
    detector = load_ctd_detector(args.ctd_model, device="cpu")
    recognizer = load_paddle_recognizer(
        args.paddle_model_dir,
        model_name="en_PP-OCRv5_mobile_rec",
        engine="onnxruntime",
        device="cpu",
    )
    service = PagePerceptionService(
        CTDPageLocalizer(detector),
        CTDCropStore(args.artifact_root),
        {"paddle": PaddleOCRCandidateAdapter(recognizer)},
    )
    result = service.process_page(args.image, args.sequence_id, args.page_index)
    summary = {
        "image": str(args.image),
        "page_index": result.page.page_index,
        "ctd_regions": len(result.page.text_regions),
        "candidate_groups": len(result.candidate_bank.groups),
        "candidate_count": sum(
            len(group.candidates) for group in result.candidate_bank.groups
        ),
        "diagnostics": [diagnostic.model_dump(mode="json") for diagnostic in result.diagnostics],
        "artifact_root": str(args.artifact_root),
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["candidate_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
