"""Real 3-page cross-page character identity smoke for seq_952f154fb1505883.

Runs the full pipeline:
    CTD → layout grouping → character perception → character crops
    → MobileNetV3 embeddings → cross-page identity graph → clusters

NOTE on character detection:
    The RT-DETRv4 ONNX detector is the production CharacterProvider but its
    weights file is not bundled in this workspace.  When --rtdetr-model is not
    supplied, the smoke injects synthetic character crops sampled from page-image
    regions to demonstrate the full identity pipeline on real image data.
    Structural output is valid; identity labels are not semantically meaningful
    for synthetic crops.

Prints:
    PAGE 0:  character instances
    PAGE 1:  character instances
    PAGE 2:  character instances
    IDENTITY GRAPH:  char-X ↔ char-Y = score
    CLUSTERS:  identity-01 = [...]
    ANONYMOUS LABELS:  A = [...]
    UNRESOLVED / AMBIGUOUS instances

Does NOT compare against GT.  Does NOT claim identity accuracy.
This is a structural smoke only.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import math

import numpy as np

from app.character_identity import CharacterIdentityResolver
from app.character_perception import CharacterPerceptionService
from app.layout_grouping import PageLayoutGrouper
from app.models.character_crops import CharacterCropPaths, CharacterCropStore
from app.models.character_embeddings import MobileNetV3EmbeddingProvider, NullEmbeddingProvider
from app.models.ctd import CTDPageLocalizer, load_ctd_detector
from app.models.manga_layout import load_manga109_balloon_provider
from app.schemas.page import BoundingBox, CharacterInstance, PageRepresentation

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
SEP = "=" * 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3-page character identity smoke for seq_952f154fb1505883."
    )
    parser.add_argument("--sequence-id", default=DEFAULT_SEQUENCE)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--ctd-model", type=Path, default=DEFAULT_CTD_MODEL)
    parser.add_argument("--layout-weights", type=Path, default=DEFAULT_LAYOUT_WEIGHTS)
    parser.add_argument(
        "--rtdetr-model", type=Path, default=None,
        help="Path to RT-DETRv4 ONNX weights (optional). "
             "If omitted, synthetic character crops are injected.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "story-ai-identity-smoke",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip MobileNetV3; use NullEmbeddingProvider (shows unmatched singletons).",
    )
    parser.add_argument(
        "--match-threshold", type=float, default=0.65,
        help="Cosine similarity threshold for cross-page identity match (default: 0.65).",
    )
    return parser.parse_args()


def _inject_synthetic_characters(
    image_path: Path,
    page_index: int,
    sequence_id: str,
    artifact_root: Path,
    n_chars: int = 3,
) -> tuple[list[CharacterInstance], dict[str, CharacterCropPaths]]:
    """Sample n_chars non-overlapping regions from the page image as synthetic
    character instances.  This is a structural placeholder only — the crops
    contain real image data but the instances are NOT real characters."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return [], {}
    h, w = image.shape[:2]
    strip_h = max(64, h // (n_chars + 1))
    strip_w = max(64, min(w // 3, 200))

    chars: list[CharacterInstance] = []
    crops: dict[str, CharacterCropPaths] = {}
    for i in range(n_chars):
        y1 = max(0, (h // (n_chars + 1)) * i)
        y2 = min(h, y1 + strip_h)
        x1 = max(0, w // 4)
        x2 = min(w, x1 + strip_w)
        if y2 <= y1 or x2 <= x1:
            continue
        char_id = f"synth-{sequence_id}-p{page_index:02d}-{i + 1:03d}"
        chars.append(CharacterInstance(
            id=char_id,
            bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
        ))
        # Write real crops from the image
        char_dir = artifact_root / sequence_id / "crops" / char_id
        char_dir.mkdir(parents=True, exist_ok=True)
        crop = image[y1:y2, x1:x2]
        char_path = char_dir / "character.png"
        cv2.imwrite(str(char_path), crop)
        # face = top third of crop
        face_crop = crop[:max(1, (y2 - y1) // 3), :]
        face_path = char_dir / "face.png"
        cv2.imwrite(str(face_path), face_crop)
        crops[char_id] = CharacterCropPaths(character=char_path, face=face_path)
    return chars, crops


def process_page(
    image_path: Path,
    page_index: int,
    sequence_id: str,
    localizer: CTDPageLocalizer,
    layout_provider: object,
    crop_store: CharacterCropStore,
    char_provider: object | None = None,
    artifact_root: Path | None = None,
) -> tuple[PageRepresentation, dict, tuple[int, int]]:
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
        provider=char_provider,
        crop_store=crop_store,
    ).process_page(layout_result.page, sequence_id, image_path)

    crop_paths = dict(char_result.crop_paths)

    # If no provider and no real detections: inject synthetic characters for
    # the structural smoke.
    if char_provider is None and not char_result.page.characters and artifact_root is not None:
        synth_chars, synth_crops = _inject_synthetic_characters(
            image_path, page_index, sequence_id, artifact_root
        )
        crop_paths.update(synth_crops)
        final_page = char_result.page.model_copy(update={"characters": synth_chars})
        return final_page, crop_paths, (width, height)

    return char_result.page, crop_paths, (width, height)


def main() -> int:
    args = parse_args()
    image_files = sorted(args.image_dir.glob("*.png"))[:3]
    if not image_files:
        raise FileNotFoundError(f"No PNG pages found in {args.image_dir}")

    print(SEP)
    print(f"  CHARACTER IDENTITY SMOKE: {args.sequence_id}")
    print(SEP)
    print(f"  Pages:     {[f.name for f in image_files]}")
    print(f"  Threshold: {args.match_threshold}")
    print(f"  Embedding: {'NullProvider (--no-embeddings)' if args.no_embeddings else 'MobileNetV3-Small'}")

    localizer = CTDPageLocalizer(load_ctd_detector(str(args.ctd_model), device="cpu"))
    layout_provider = load_manga109_balloon_provider(args.layout_weights)
    crop_store = CharacterCropStore(args.artifact_root / args.sequence_id / "crops")

    # Optional: real RT-DETRv4 character detector
    char_provider = None
    if args.rtdetr_model is not None:
        from app.models.rtdetrv4_character_provider import load_rtdetrv4_character_provider
        char_provider = load_rtdetrv4_character_provider(args.rtdetr_model)
        print(f"  CharProvider: RT-DETRv4 ({args.rtdetr_model.name})")
    else:
        print("  CharProvider: None (synthetic crops will be injected)")

    pages: list[PageRepresentation] = []
    all_crops: dict = {}

    for page_index, image_path in enumerate(image_files):
        print(f"\nProcessing page {page_index}: {image_path.name} …", end="", flush=True)
        page, crop_paths, _ = process_page(
            image_path, page_index, args.sequence_id,
            localizer, layout_provider, crop_store,
            char_provider=char_provider,
            artifact_root=args.artifact_root,
        )
        pages.append(page)
        all_crops.update(crop_paths)
        synthetic = "(synthetic)" if char_provider is None else ""
        print(f" {len(page.characters)} character(s) {synthetic}, {len(page.balloons)} balloon(s)")

    # Print per-page character instances
    for page_idx, page in enumerate(pages):
        print(f"\n{SEP}")
        print(f"  PAGE {page_idx}   ({len(page.characters)} instance(s))")
        print(SEP)
        if page.characters:
            for char in page.characters:
                cx = (char.bbox.x1 + char.bbox.x2) / 2
                cy = (char.bbox.y1 + char.bbox.y2) / 2
                face = "face✓" if char.face_bbox else "face✗"
                body = "body✓" if char.body_bbox else "body✗"
                crops_info = "crops✓" if char.id in all_crops else "crops✗"
                print(
                    f"    {char.id}  center=({cx:.0f},{cy:.0f})  "
                    f"conf={char.confidence or '?'}  {face}  {body}  {crops_info}"
                )
        else:
            print("    (no character instances detected)")

    # Run identity resolution
    print(f"\n{SEP}")
    print("  RUNNING IDENTITY RESOLUTION …")
    print(SEP)

    if args.no_embeddings:
        provider = NullEmbeddingProvider()
    else:
        provider = MobileNetV3EmbeddingProvider(device="cpu")

    resolver = CharacterIdentityResolver(
        provider,
        match_threshold=args.match_threshold,
        ambiguity_low=0.45,
        ambiguity_high=args.match_threshold,
    )
    result = resolver.resolve(args.sequence_id, pages, all_crops)

    # Print identity graph (cross-page edges)
    print(f"\n{SEP}")
    cross_page_ev = [ev for ev in result.pair_evidence if not ev.same_page]
    print(f"  IDENTITY GRAPH  ({len(cross_page_ev)} cross-page edge(s))")
    print(SEP)
    if cross_page_ev:
        sorted_ev = sorted(cross_page_ev, key=lambda e: -e.combined_score)
        for ev in sorted_ev:
            face_str = f"face={ev.face_similarity:.3f}" if ev.face_similarity is not None else "face=n/a"
            body_str = f"body={ev.body_similarity:.3f}" if ev.body_similarity is not None else "body=n/a"
            full_str = f"full={ev.full_similarity:.3f}" if ev.full_similarity is not None else "full=n/a"
            print(
                f"    {ev.first_character_id}  <->  {ev.second_character_id}"
                f"  score={ev.combined_score:.3f}  conf={ev.confidence:.2f}"
                f"  [{full_str}  {face_str}  {body_str}]"
            )
    else:
        print("    (no cross-page edges — embeddings may all be None)")

    # Print clusters
    print(f"\n{SEP}")
    print(f"  CLUSTERS  ({len(result.clusters)} total)")
    print(SEP)
    matched = [cl for cl in result.clusters if cl.state == "matched"]
    unmatched = [cl for cl in result.clusters if cl.state == "unmatched"]
    ambiguous = [cl for cl in result.clusters if cl.state == "ambiguous"]
    print(f"    matched={len(matched)}  unmatched={len(unmatched)}  ambiguous={len(ambiguous)}")
    for cluster in result.clusters:
        member_str = ", ".join(
            f"p{m.page_index}:{m.character_id}" for m in cluster.members
        )
        print(
            f"    {cluster.identity_id}  [{cluster.state}]  "
            f"conf={cluster.confidence:.2f}  members=[{member_str}]"
        )

    # Print anonymous labels
    print(f"\n{SEP}")
    print("  ANONYMOUS LABELS")
    print(SEP)
    label_to_ids: dict[str, list[str]] = {}
    for cid, label in result.character_to_label.items():
        label_to_ids.setdefault(label or "?", []).append(cid)
    for label in sorted(label_to_ids):
        print(f"    {label:>3} = {label_to_ids[label]}")

    # Print unresolved / ambiguous
    unresolved_ids = [
        cid for cid, iid in result.character_to_identity.items() if iid is None
    ]
    if unresolved_ids:
        print(f"\n  UNRESOLVED:  {unresolved_ids}")
    if result.diagnostics:
        print(f"\n  DIAGNOSTICS  ({len(result.diagnostics)}):")
        for d in result.diagnostics[:10]:
            print(f"    [{d.code}] {d.message[:80]}")
        if len(result.diagnostics) > 10:
            print(f"    … and {len(result.diagnostics) - 10} more")

    print(f"\n  Summary: {len(pages)} pages, "
          f"{sum(len(p.characters) for p in pages)} character instances, "
          f"{len(result.clusters)} identity clusters "
          f"({len(matched)} matched, {len(unmatched)} unmatched, {len(ambiguous)} ambiguous)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
