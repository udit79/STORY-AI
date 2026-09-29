"""Real 3-page reading-order structural smoke for seq_952f154fb1505883.

Runs:
  page_01.png → page_02.png → page_03.png

Prints:
  PAGE 0
      panel order
      balloon order
  PAGE 1
      panel order
      balloon order
  PAGE 2
      panel order
      balloon order

  FINAL:
      ordered_balloon_ids = [...]
      diagnostics (if any)

No model downloads.  Uses local CTD + local MangaLens YOLO layout weights.
CharacterProvider = None (provider_unavailable diagnostic expected).
No adjudication required: all balloons included in story.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.character_perception import CharacterPerceptionService
from app.layout_grouping import PageLayoutGrouper
from app.models.character_crops import CharacterCropStore
from app.models.ctd import CTDPageLocalizer, load_ctd_detector
from app.models.manga_layout import load_manga109_balloon_provider
from app.reading_order import DeterministicGeometryRanker, order_sequence
from app.schemas.page import PageRepresentation
from app.speaker_grounding import SpeakerResolver

DEFAULT_SEQUENCE = "seq_952f154fb1505883"
DEFAULT_IMAGE_DIR = ROOT / "dataset" / "development" / "images" / DEFAULT_SEQUENCE
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
        description="3-page reading-order smoke for seq_952f154fb1505883."
    )
    parser.add_argument("--sequence-id", default=DEFAULT_SEQUENCE)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--ctd-model", type=Path, default=DEFAULT_CTD_MODEL)
    parser.add_argument("--layout-weights", type=Path, default=DEFAULT_LAYOUT_WEIGHTS)
    return parser.parse_args()


def process_page(
    image_path: Path,
    page_index: int,
    sequence_id: str,
    localizer: CTDPageLocalizer,
    layout_provider: object,
    crop_store: CharacterCropStore,
) -> tuple[PageRepresentation, tuple[int, int]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read page image: {image_path}")
    height, width = image.shape[:2]

    regions = localizer.localize(image_path)
    page = PageRepresentation(
        page_index=page_index,
        image_path=str(image_path),
        text_regions=regions,
    )
    layout_result = PageLayoutGrouper(
        panel_provider=layout_provider,
        balloon_provider=layout_provider,
    ).group_page(page, sequence_id, image_path, page_size=(width, height))

    char_result = CharacterPerceptionService(
        provider=None,
        crop_store=crop_store,
    ).process_page(layout_result.page, sequence_id, image_path)

    speaker_result = SpeakerResolver().resolve_page(
        char_result.page, page_size=(width, height)
    )
    return speaker_result.page, (width, height)


def print_page_order(page_order, page_index: int, panel_map: dict) -> None:
    print(f"\n{'═' * 60}")
    print(f"  PAGE {page_index}")
    print(f"{'═' * 60}")

    panel_ids = page_order.panel_ids_in_order
    if panel_ids:
        print(f"  Panel order ({len(panel_ids)} panel(s)):")
        for rank, pid in enumerate(panel_ids, start=1):
            if pid is None:
                print(f"    [{rank}] <page-level fallback>")
            elif pid in panel_map:
                p = panel_map[pid]
                cx = (p.bbox.x1 + p.bbox.x2) / 2
                cy = (p.bbox.y1 + p.bbox.y2) / 2
                print(f"    [{rank}] {pid}  center=({cx:.0f}, {cy:.0f})")
            else:
                print(f"    [{rank}] {pid}")
    else:
        print("  Panel order: (none)")

    print(f"  Balloon order ({len(page_order.balloon_ids_in_order)} story balloon(s)):")
    if page_order.balloon_ids_in_order:
        for rank, bid in enumerate(page_order.balloon_ids_in_order, start=1):
            print(f"    [{rank}] {bid}")
    else:
        print("    (no story balloons)")

    if page_order.diagnostics:
        print("  Diagnostics:")
        for d in page_order.diagnostics:
            print(f"    [{d.code}] {textwrap.shorten(d.message, width=80)}")


def main() -> int:
    args = parse_args()
    image_files = sorted(args.image_dir.glob("*.png"))[:3]
    if not image_files:
        raise FileNotFoundError(f"No PNG pages found in {args.image_dir}")

    print(f"Sequence:  {args.sequence_id}")
    print(f"Pages:     {[f.name for f in image_files]}")
    print(f"CTD model: {args.ctd_model}")

    localizer = CTDPageLocalizer(load_ctd_detector(str(args.ctd_model), device="cpu"))
    layout_provider = load_manga109_balloon_provider(args.layout_weights)
    crop_store = CharacterCropStore(ROOT / ".smoke" / "reading-order" / args.sequence_id / "crops")

    pages: list[PageRepresentation] = []
    page_sizes: list[tuple[int, int]] = []

    for page_index, image_path in enumerate(image_files):
        print(f"\nProcessing page {page_index}: {image_path.name} …", end="", flush=True)
        page, size = process_page(
            image_path, page_index, args.sequence_id,
            localizer, layout_provider, crop_store,
        )
        pages.append(page)
        page_sizes.append(size)
        print(
            f" {len(page.panels)} panel(s), "
            f"{len(page.balloons)} balloon(s), "
            f"{len(page.text_regions)} CTD region(s)"
        )

    ranker = DeterministicGeometryRanker()
    sequence_order = order_sequence(pages, ranker=ranker, page_sizes=page_sizes)

    # Print per-page detail
    for page_idx, page_order in enumerate(sequence_order.page_orders):
        panel_map = {p.id: p for p in pages[page_idx].panels}
        print_page_order(page_order, page_idx, panel_map)

    # Final summary
    print(f"\n{'═' * 60}")
    print("  FINAL READING ORDER")
    print(f"{'═' * 60}")
    print(f"  ordered_balloon_ids ({len(sequence_order.ordered_balloon_ids)} total) = [")
    for position, bid in enumerate(sequence_order.ordered_balloon_ids):
        # Find which page this balloon belongs to
        page_label = None
        for page_idx, page in enumerate(pages):
            if any(b.id == bid for b in page.balloons):
                page_label = f"p{page_idx}"
                break
        print(f"    {position:3d}: [{page_label}] {bid}")
    print("  ]")

    if sequence_order.diagnostics:
        print("\n  Sequence diagnostics:")
        for d in sequence_order.diagnostics:
            print(f"    [{d.code}] {d.message}")

    total_decisions = sum(
        len(po.decisions) for po in sequence_order.page_orders
    )
    total_diag = sum(
        len(po.diagnostics) for po in sequence_order.page_orders
    )
    print(f"\n  Summary: {len(sequence_order.ordered_balloon_ids)} story balloons ordered")
    print(f"           {total_decisions} pairwise decisions made")
    print(f"           {total_diag} ordering diagnostics")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
