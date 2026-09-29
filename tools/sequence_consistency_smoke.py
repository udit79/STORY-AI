"""Real 3-page structural smoke for SequenceConsistencyResolver.

Runs a full development sequence through:
    CTD > layout/grouping > adjudication (PaddleOCR-based) > speaker grounding
    > reading order > character identity > sequence consistency resolver

Prints a structured report:
    SEQUENCE:          sequence_id
    BALLOONS:          ordered positions
    Per balloon:       page / panel / balloon_id / text / include_in_story /
                       speaker_char_instance / identity_id / anonymous_label
    IDENTITIES:        A = [...], B = [...], ...
    DIAGNOSTICS:       resolver + validation errors

No GPU required.
No model downloads.
No benchmark accuracy claims.
Character identity/speaker may be unresolved if detector evidence is insufficient.
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

from app.character_identity import CharacterIdentityResolver
from app.character_perception import CharacterPerceptionService
from app.layout_grouping import PageLayoutGrouper
from app.models.character_crops import CharacterCropStore
from app.models.ctd import CTDPageLocalizer, load_ctd_detector
from app.models.manga_layout import load_manga109_balloon_provider
from app.reading_order import DeterministicGeometryRanker, order_sequence
from app.schemas.adjudication import BalloonAdjudicationResult
from app.schemas.page import PageRepresentation
from app.sequence_resolver import SequenceConsistencyResolver
from app.sequence_validator import validate_sequence_resolution
from app.speaker_grounding import PageSpeakerGroundingResult, SpeakerResolver
from app.submission_serializer import serialize_resolution_to_jsonl

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
DEFAULT_SMOKE_DIR = ROOT / ".smoke" / "sequence-consistency"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-id", default=DEFAULT_SEQUENCE)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--ctd-model", type=Path, default=DEFAULT_CTD_MODEL)
    parser.add_argument("--layout-weights", type=Path, default=DEFAULT_LAYOUT_WEIGHTS)
    parser.add_argument("--smoke-dir", type=Path, default=DEFAULT_SMOKE_DIR)
    return parser.parse_args()


def _hr(char: str = "=", width: int = 68) -> str:
    return char * width


def process_page(
    image_path: Path,
    page_index: int,
    sequence_id: str,
    localizer: CTDPageLocalizer,
    layout_provider: object,
    crop_store: CharacterCropStore,
) -> tuple[PageRepresentation, PageSpeakerGroundingResult, tuple[int, int]]:
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
    return speaker_result.page, speaker_result, (width, height)


def adjudicate_page_balloons(
    page: PageRepresentation,
    sequence_id: str,
) -> list[BalloonAdjudicationResult]:
    """Minimal structural adjudication for the smoke tool.

    Reads raw_text from CTD regions (populated by PaddleOCR when available).
    When raw_text is empty (CTD-only run without OCR), assigns a structural
    placeholder text so the resolver and serializer can exercise all code paths.
    In a production run, BalloonAdjudicationResult objects come from the real
    multimodal adjudicator (Qwen3-VL) — this function is smoke-only.
    """
    results: list[BalloonAdjudicationResult] = []

    # Build a text-region index for this page
    region_index = {r.id: r for r in page.text_regions}

    kind_map = {
        "speech": "dialogue",
        "thought": "thought",
        "narration": "narration",
        "unknown": "dialogue",
    }

    for balloon in page.balloons:
        # Gather any region text (present when PaddleOCR has run)
        texts = [
            region_index[rid].raw_text.strip()
            for rid in balloon.text_region_ids
            if rid in region_index and region_index[rid].raw_text.strip()
        ]
        has_regions = bool(balloon.text_region_ids)
        text_type = kind_map.get(balloon.kind, "dialogue")

        if texts:
            final_text = " ".join(texts)
            results.append(BalloonAdjudicationResult(
                balloon_id=balloon.id,
                final_text=final_text,
                text_type=text_type,
                include_in_story=True,
            ))
        elif has_regions:
            # Regions exist but OCR has not been run (CTD-only mode).
            # Use a structural placeholder so the resolver exercises all paths.
            final_text = f"[balloon {balloon.id} — OCR pending]"
            results.append(BalloonAdjudicationResult(
                balloon_id=balloon.id,
                final_text=final_text,
                text_type=text_type,
                include_in_story=True,
            ))
        else:
            # No regions at all → exclude
            results.append(BalloonAdjudicationResult(
                balloon_id=balloon.id,
                final_text="",
                text_type="unknown",
                include_in_story=False,
            ))

    return results


def main() -> int:
    args = parse_args()

    image_files = sorted(args.image_dir.glob("*.png"))[:3]
    if not image_files:
        raise FileNotFoundError(f"No PNG pages found in {args.image_dir}")

    print(_hr())
    print(f"  SEQUENCE CONSISTENCY SMOKE")
    print(_hr())
    print(f"  sequence_id  : {args.sequence_id}")
    print(f"  images       : {[f.name for f in image_files]}")
    print(f"  ctd_model    : {args.ctd_model}")
    print()

    # ── Per-page processing ──────────────────────────────────────────────────
    localizer = CTDPageLocalizer(load_ctd_detector(str(args.ctd_model), device="cpu"))
    layout_provider = load_manga109_balloon_provider(args.layout_weights)
    crop_store = CharacterCropStore(
        args.smoke_dir / args.sequence_id / "crops"
    )

    pages: list[PageRepresentation] = []
    speaker_results: list[PageSpeakerGroundingResult] = []
    all_adjudication: list[BalloonAdjudicationResult] = []
    page_sizes: list[tuple[int, int]] = []

    for page_index, image_path in enumerate(image_files):
        print(f"  Processing page {page_index}: {image_path.name} …", end="", flush=True)
        page, speaker_result, size = process_page(
            image_path, page_index, args.sequence_id,
            localizer, layout_provider, crop_store,
        )
        pages.append(page)
        speaker_results.append(speaker_result)
        page_sizes.append(size)

        adj_results = adjudicate_page_balloons(page, args.sequence_id)
        all_adjudication.extend(adj_results)

        n_story = sum(1 for r in adj_results if r.include_in_story)
        print(
            f" {len(page.panels)} panel(s), "
            f"{len(page.balloons)} balloon(s), "
            f"{n_story} story balloon(s), "
            f"{len(page.text_regions)} CTD region(s)"
        )

    # ── Reading order ────────────────────────────────────────────────────────
    adj_index = {r.balloon_id: r for r in all_adjudication}
    reading_order = order_sequence(
        pages,
        ranker=DeterministicGeometryRanker(),
        adjudication_results=adj_index,
        page_sizes=page_sizes,
    )
    print(f"\n  Reading order: {len(reading_order.ordered_balloon_ids)} story balloon(s)")

    # ── Character identity ───────────────────────────────────────────────────
    identity = CharacterIdentityResolver().resolve(args.sequence_id, pages)
    print(f"  Identity: {len(identity.clusters)} cluster(s), "
          f"{len(identity.diagnostics)} identity diagnostic(s)")

    # ── Sequence consistency resolver ────────────────────────────────────────
    resolver = SequenceConsistencyResolver()
    resolution = resolver.resolve(
        sequence_id=args.sequence_id,
        pages=pages,
        adjudication_results=all_adjudication,
        reading_order=reading_order,
        speaker_results=speaker_results,
        identity=identity,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print()
    print(_hr())
    print(f"  SEQUENCE:  {resolution.sequence_id}")
    print(_hr())

    print(f"\n  BALLOONS  ({len(resolution.ordered_balloons)} story)")
    print(f"  {'pos':>3}  {'pg':>3}  {'panel':>24}  {'text_type':>14}  {'speaker_label':>14}")
    print(f"  {'-'*3}  {'-'*3}  {'-'*24}  {'-'*14}  {'-'*14}")

    for b in resolution.ordered_balloons:
        panel_str = (b.panel_id or "(none)")[-24:]
        label_str = b.speaker_label or "(none)"
        print(
            f"  {b.sequence_position:>3}  "
            f"{b.page_index:>3}  "
            f"{panel_str:>24}  "
            f"{b.text_type:>14}  "
            f"{label_str:>14}"
        )
        print(f"       balloon: {b.balloon_id}")
        print(f"       text:    {textwrap.shorten(b.text, width=60)!r}")
        char_str = b.speaker_character_instance_id or "(none)"
        id_str   = b.speaker_identity_id or "(none)"
        print(f"       speaker: char={char_str}  identity={id_str}")
        if b.diagnostics:
            for d in b.diagnostics:
                print(f"       [{d.severity}] {d.code}: {textwrap.shorten(d.message, 60)}")

    print(f"\n  EXCLUDED BALLOONS  ({len(resolution.excluded_balloons)})")
    for b in resolution.excluded_balloons:
        print(f"    [{b.page_index}] {b.balloon_id}  type={b.text_type}")

    print(f"\n  IDENTITIES  ({len(resolution.identities)})")
    for ident in resolution.identities:
        members_str = ", ".join(ident.member_character_ids)
        print(f"    {ident.label} = [{members_str}]  state={ident.state}  pages={ident.page_indices}")

    print(f"\n  RESOLVER DIAGNOSTICS  ({len(resolution.diagnostics)})")
    for d in resolution.diagnostics:
        print(f"    [{d.severity}] {d.code}: {textwrap.shorten(d.message, 70)}")

    # ── Validation ────────────────────────────────────────────────────────────
    known_chars = {char.id for page in pages for char in page.characters}
    validation_errors = validate_sequence_resolution(
        resolution,
        known_character_ids=known_chars,
        expected_sequence_id=args.sequence_id,
    )
    print(f"\n  VALIDATION  ({len(validation_errors)} error(s))")
    for err in validation_errors:
        print(f"    [FAIL] {err.rule}: {textwrap.shorten(err.message, 70)}")
    if not validation_errors:
        print("    All checks passed.")

    # ── Sample JSONL line ─────────────────────────────────────────────────────
    print(f"\n  SAMPLE JSONL LINE:")
    jsonl_line = serialize_resolution_to_jsonl(resolution)
    # Truncate for readability
    display = jsonl_line[:200] + (" …" if len(jsonl_line) > 200 else "")
    print(f"    {display}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(_hr())
    print("  SUMMARY")
    print(_hr())
    print(f"  story balloons:          {len(resolution.ordered_balloons)}")
    print(f"  excluded balloons:       {len(resolution.excluded_balloons)}")
    print(f"  unresolved items:        {len(resolution.unresolved_items)}")
    print(f"  identity clusters:       {len(resolution.identities)}")
    print(f"  resolver diagnostics:    {len(resolution.diagnostics)}")
    print(f"  validation errors:       {len(validation_errors)}")
    n_with_speaker = sum(1 for b in resolution.ordered_balloons if b.speaker_label is not None)
    n_unknown = sum(
        1 for b in resolution.ordered_balloons
        if b.speaker_label is None and b.text_type not in ("narration", "thought")
    )
    print(f"  balloons with label:     {n_with_speaker}")
    print(f"  balloons unresolved spk: {n_unknown}")
    print(_hr())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
