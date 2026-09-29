"""One real-page CTD to panel/balloon grouping structural smoke path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.layout_grouping import PageLayoutGrouper
from app.models.ctd import CTDPageLocalizer, load_ctd_detector
from app.models.manga_layout import load_manga109_balloon_provider
from app.schemas.page import PageRepresentation

DEFAULT_SEQUENCE = "seq_2032620aa4e4ac7f"
DEFAULT_IMAGE = (
    ROOT / "dataset" / "development" / "images" / DEFAULT_SEQUENCE / "01.png"
)
DEFAULT_CTD_MODEL = (
    ROOT / "vendor" / "comic-text-detector" / "data" / "comictextdetector.pt.onnx"
)
DEFAULT_LAYOUT_WEIGHTS = (
    Path.home()
    / ".cache"
    / "story-ai"
    / "manga-layout"
    / "huyvux3005-manga109-segmentation-bubble-best.pt"
)


class RecordingLayoutProvider:
    def __init__(self, provider):
        self.provider = provider
        self.balloon_proposals = []

    def detect_panels(self, image_path, page_width, page_height):
        return self.provider.detect_panels(image_path, page_width, page_height)

    def propose_balloons(self, image_path, page_width, page_height):
        self.balloon_proposals = list(
            self.provider.propose_balloons(image_path, page_width, page_height)
        )
        return self.balloon_proposals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one local CTD -> panel fallback -> balloon grouping smoke path."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--sequence-id", default=DEFAULT_SEQUENCE)
    parser.add_argument("--page-index", type=int, default=0)
    parser.add_argument("--ctd-model", type=Path, default=DEFAULT_CTD_MODEL)
    parser.add_argument("--layout-weights", type=Path, default=DEFAULT_LAYOUT_WEIGHTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Development page not found: {args.image}")
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not read page image: {args.image}")
    height, width = image.shape[:2]

    localizer = CTDPageLocalizer(load_ctd_detector(args.ctd_model, device="cpu"))
    layout_provider = RecordingLayoutProvider(
        load_manga109_balloon_provider(args.layout_weights, image_size=1600)
    )
    regions = localizer.localize(args.image)
    page = PageRepresentation(
        page_index=args.page_index,
        image_path=str(args.image),
        text_regions=regions,
    )
    result = PageLayoutGrouper(
        panel_provider=layout_provider,
        balloon_provider=layout_provider,
    ).group_page(
        page,
        args.sequence_id,
        args.image,
        page_size=(width, height),
    )

    print(f"page: {args.sequence_id} / page_{args.page_index + 1:02d}")
    print(f"panels: {len(result.page.panels)}")
    print(f"balloon proposals: {result.balloon_proposal_count}")
    print(f"CTD regions: {len(result.page.text_regions)}")
    for region in result.page.text_regions:
        print(f"CTD bbox: {region.id} -> {region.bbox.model_dump()}")
    for proposal_index, proposal in enumerate(layout_provider.balloon_proposals, start=1):
        print(
            f"proposal {proposal_index}: bbox={proposal.bbox.model_dump()} "
            f"confidence={proposal.confidence}"
        )
    print(f"balloons: {len(result.page.balloons)}")
    for balloon in result.page.balloons:
        print(f"{balloon.id}: {balloon.text_region_ids}")
    print(f"unassigned CTD regions: {result.unassigned_text_region_ids}")
    if result.diagnostics:
        print(f"diagnostics: {[item.model_dump(mode='json') for item in result.diagnostics]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
