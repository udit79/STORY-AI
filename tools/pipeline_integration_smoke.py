"""Full real 3-page development pipeline integration smoke.

Architecture exercised (LOCKED):
    3 consecutive pages
        ↓ CTD localization (real local model)
        ↓ Manga balloon layout/grouping (real local YOLO weights)
        ↓ CTD → balloon grouping
        ↓ Balloon crops
        ↓ PaddleOCR proposals  (real local cached model)
        ↓ Qwen3-VL proposals   (real local cached model, CUDA)
        ↓ Nemotron             (WSL bridge -- reported unavailable if absent)
        ↓ Candidate Bank
        ↓ Qwen multimodal adjudication (real local)
        ↓ Character perception (real RT-DETRv4 ONNX)
        ↓ Character crops
        ↓ Speaker grounding (geometry)
        ↓ Reading order (deterministic geometry)
        ↓ Cross-page character identity (MobileNetV3 embeddings)
        ↓ Sequence consistency resolver
        ↓ Validator
        ↓ Competition JSONL

No ground-truth labels are used.
No synthetic data is injected.
One component failing does NOT stop the rest.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Pipeline imports --------------------------------------------------------
from app.balloon_adjudication import Qwen3VLBalloonAdjudicator, adjudicate_balloon
from app.candidate_bank import build_candidate_bank
from app.character_identity import CharacterIdentityResolver
from app.character_perception import CharacterPerceptionService
from app.layout_grouping import PageLayoutGrouper
from app.models.balloon_crops import BalloonCropStore
from app.models.candidate_adapters import (
    NemotronCandidateAdapter,
    PaddleOCRCandidateAdapter,
    Qwen3VLCandidateAdapter,
)
from app.models.character_crops import CharacterCropPaths, CharacterCropStore
from app.models.character_embeddings import (
    MobileNetV3EmbeddingProvider,
    NullEmbeddingProvider,
)
from app.models.crops import CTDCropStore
from app.models.ctd import CTDPageLocalizer, load_ctd_detector
from app.models.manga_layout import load_manga109_balloon_provider
from app.models.rtdetrv4_character_provider import load_rtdetrv4_character_provider
from app.reading_order import DeterministicGeometryRanker, order_sequence
from app.schemas.adjudication import BalloonAdjudicationResult
from app.schemas.candidates import CandidateBank
from app.schemas.page import PageRepresentation
from app.sequence_resolver import SequenceConsistencyResolver
from app.sequence_validator import validate_sequence_resolution
from app.speaker_grounding import (
    PageSpeakerGroundingResult,
    QwenVisualSpeakerGrounder,
    SpeakerContextImageStore,
    SpeakerResolver,
    VisualSpeakerContext,
)
from app.submission_serializer import serialize_resolution_to_jsonl

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SEQUENCE    = "seq_952f154fb1505883"
DEFAULT_IMAGE_DIR   = ROOT / "dataset" / "development" / "images" / DEFAULT_SEQUENCE
DEFAULT_CTD_MODEL   = ROOT / "vendor" / "comic-text-detector" / "data" / "comictextdetector.pt.onnx"
DEFAULT_LAYOUT_WEIGHTS = (
    Path.home() / ".cache" / "huggingface" / "hub"
    / "models--huyvux3005--manga109-segmentation-bubble"
    / "snapshots" / "f9a4108c4955136a810e5e92207972f3fb3a65fd" / "best.pt"
)
DEFAULT_PADDLE_MODEL_DIR = (
    Path.home() / ".paddlex" / "official_models" / "en_PP-OCRv5_mobile_rec_onnx"
)
DEFAULT_RTDETR_MODEL = (
    Path.home() / ".cache" / "huggingface" / "hub"
    / "models--tori29umai--rtdetrv4-x-manga109s_v2"
    / "snapshots" / "864c3bfb837a03ecc62557d5152a5ade5566489b" / "model.onnx"
)
DEFAULT_SMOKE_DIR = ROOT / ".smoke" / "pipeline-integration"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence-id",       default=DEFAULT_SEQUENCE)
    p.add_argument("--image-dir",         type=Path, default=DEFAULT_IMAGE_DIR)
    p.add_argument("--ctd-model",         type=Path, default=DEFAULT_CTD_MODEL)
    p.add_argument("--layout-weights",    type=Path, default=DEFAULT_LAYOUT_WEIGHTS)
    p.add_argument("--paddle-model-dir",  type=Path, default=DEFAULT_PADDLE_MODEL_DIR)
    p.add_argument("--rtdetr-model",      type=Path, default=DEFAULT_RTDETR_MODEL)
    p.add_argument("--smoke-dir",         type=Path, default=DEFAULT_SMOKE_DIR)
    p.add_argument("--skip-qwen",         action="store_true",
                   help="Skip Qwen proposals and adjudication (use structural placeholder)")
    p.add_argument("--skip-paddle",       action="store_true",
                   help="Skip PaddleOCR proposals")
    p.add_argument("--skip-nemotron",     action="store_true",
                   help="Skip Nemotron WSL bridge (report unavailable)")
    p.add_argument("--skip-visual-speaker", action="store_true",
                   help="Skip Qwen visual speaker grounding (geometry only)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hr(char: str = "=", width: int = 72) -> str:
    return char * width


def _t() -> float:
    return time.perf_counter()


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


class _ComponentStatus:
    """Track which real backends loaded successfully vs. reported unavailable."""

    def __init__(self) -> None:
        self.available:   list[str] = []
        self.unavailable: dict[str, str] = {}

    def ok(self, name: str) -> None:
        self.available.append(name)

    def fail(self, name: str, reason: str) -> None:
        self.unavailable[name] = reason


# ---------------------------------------------------------------------------
# Backend initialization
# ---------------------------------------------------------------------------

def _init_backends(
    args: argparse.Namespace,
    status: _ComponentStatus,
) -> dict:
    """Load all real local backends; record availability without crashing."""
    backends: dict = {}

    # CTD
    t = _t()
    try:
        detector = load_ctd_detector(str(args.ctd_model), device="cpu")
        backends["ctd_localizer"] = CTDPageLocalizer(detector)
        status.ok(f"CTD ({_elapsed(t)})")
    except Exception as exc:  # noqa: BLE001
        status.fail("CTD", f"{type(exc).__name__}: {exc}")
        backends["ctd_localizer"] = None

    # Manga balloon layout provider
    t = _t()
    try:
        backends["layout_provider"] = load_manga109_balloon_provider(args.layout_weights)
        status.ok(f"MangaLayout YOLO ({_elapsed(t)})")
    except Exception as exc:  # noqa: BLE001
        status.fail("MangaLayout", f"{type(exc).__name__}: {exc}")
        backends["layout_provider"] = None

    # PaddleOCR
    paddle_adapter = None
    if not args.skip_paddle:
        t = _t()
        try:
            from app.models.local_runners import load_paddle_recognizer
            paddle = load_paddle_recognizer(
                args.paddle_model_dir,
                model_name="en_PP-OCRv5_mobile_rec",
                engine="onnxruntime",
                device="cpu",
            )
            paddle_adapter = PaddleOCRCandidateAdapter(paddle, preprocessing="base")
            status.ok(f"PaddleOCR ({_elapsed(t)})")
        except Exception as exc:  # noqa: BLE001
            status.fail("PaddleOCR", f"{type(exc).__name__}: {exc}")
    else:
        status.fail("PaddleOCR", "skipped via --skip-paddle")
    backends["paddle_adapter"] = paddle_adapter

    # Qwen3-VL runner (shared for proposals AND adjudication)
    qwen_runner = None
    if not args.skip_qwen:
        t = _t()
        try:
            from app.models.local_runners import load_qwen3vl_runner
            qwen_runner = load_qwen3vl_runner(
                local_files_only=True,
                quantize_4bit=True,
                max_new_tokens=256,
            )
            status.ok(f"Qwen3-VL 4B-Instruct ({_elapsed(t)})")
        except Exception as exc:  # noqa: BLE001
            status.fail("Qwen3-VL", f"{type(exc).__name__}: {exc}")
    else:
        status.fail("Qwen3-VL", "skipped via --skip-qwen")
    backends["qwen_runner"] = qwen_runner

    qwen_candidate_adapter = Qwen3VLCandidateAdapter(qwen_runner) if qwen_runner is not None else None
    backends["qwen_candidate_adapter"] = qwen_candidate_adapter

    qwen_adjudicator = Qwen3VLBalloonAdjudicator(qwen_runner) if qwen_runner is not None else None
    backends["qwen_adjudicator"] = qwen_adjudicator

    # Nemotron -- WSL persistent bridge
    nemotron_adapter = None
    if not getattr(args, "skip_nemotron", False):
        t = _t()
        try:
            from app.models.nemotron_wsl_bridge import NemotronWSLBridge
            bridge = NemotronWSLBridge.start(startup_timeout=180.0)
            # Wrap in NemotronCandidateAdapter (bridge is callable like NemotronOCRV2)
            nemotron_adapter = NemotronCandidateAdapter(bridge, merge_level="paragraph")
            # Keep reference so caller can shut down after pipeline
            backends["_nemotron_bridge"] = bridge
            status.ok(f"Nemotron WSL bridge ({_elapsed(t)})")
        except Exception as exc:  # noqa: BLE001
            status.fail("Nemotron", f"{type(exc).__name__}: {exc}")
    else:
        status.fail("Nemotron", "skipped via --skip-nemotron")
    backends["nemotron_adapter"] = nemotron_adapter

    # RT-DETRv4 character provider
    t = _t()
    try:
        backends["character_provider"] = load_rtdetrv4_character_provider(args.rtdetr_model)
        status.ok(f"RT-DETRv4 character provider ({_elapsed(t)})")
    except Exception as exc:  # noqa: BLE001
        status.fail("RTDETRv4", f"{type(exc).__name__}: {exc}")
        backends["character_provider"] = None

    # MobileNetV3 character embedding provider
    t = _t()
    try:
        embedding_provider = MobileNetV3EmbeddingProvider(device="cpu")
        backends["embedding_provider"] = embedding_provider
        status.ok(f"MobileNetV3 embeddings ({_elapsed(t)})")
    except Exception as exc:  # noqa: BLE001
        status.fail("MobileNetV3", f"{type(exc).__name__}: {exc}")
        backends["embedding_provider"] = NullEmbeddingProvider()

    return backends


# ---------------------------------------------------------------------------
# Per-page processing
# ---------------------------------------------------------------------------

def _process_page(
    image_path: Path,
    page_index: int,
    sequence_id: str,
    backends: dict,
    ctd_crop_store: CTDCropStore,
    balloon_crop_store: BalloonCropStore,
    char_crop_store: CharacterCropStore,
    speaker_context_store: SpeakerContextImageStore | None,
    timings: dict,
) -> tuple[
    PageRepresentation,             # page with characters + balloons
    CandidateBank,                  # per-region candidate bank
    list[BalloonAdjudicationResult], # adjudicated balloons
    PageSpeakerGroundingResult,     # speaker decisions
    tuple[int, int],                # (width, height)
    list[str],                      # per-page warnings
]:
    warnings: list[str] = []
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read page image: {image_path}")
    page_h, page_w = image.shape[:2]

    # ── CTD localization ──────────────────────────────────────────────────────
    t = _t()
    if backends["ctd_localizer"] is not None:
        try:
            regions = backends["ctd_localizer"].localize(image_path)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"CTD failed: {exc}")
            regions = []
    else:
        warnings.append("CTD localizer unavailable; no regions")
        regions = []
    timings.setdefault("ctd", []).append(time.perf_counter() - t)

    page = PageRepresentation(
        page_index=page_index,
        image_path=str(image_path),
        text_regions=regions,
    )

    # ── Layout/balloon grouping ───────────────────────────────────────────────
    t = _t()
    layout_provider = backends.get("layout_provider")
    try:
        layout_grouper = PageLayoutGrouper(
            panel_provider=layout_provider,
            balloon_provider=layout_provider,
        )
        layout_result = layout_grouper.group_page(
            page, sequence_id, image_path, page_size=(page_w, page_h)
        )
        grouped_page = layout_result.page
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Layout grouping failed: {exc}")
        grouped_page = page
    timings.setdefault("layout", []).append(time.perf_counter() - t)

    # ── CTD crops ─────────────────────────────────────────────────────────────
    t = _t()
    try:
        ctd_crop_result = ctd_crop_store.generate(
            image_path, sequence_id, page_index, grouped_page.text_regions
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"CTD crop store failed: {exc}")
        ctd_crop_result = None
    timings.setdefault("ctd_crops", []).append(time.perf_counter() - t)

    # ── OCR/VLM candidate production ─────────────────────────────────────────
    t = _t()
    all_candidates = []
    producers_to_run = []
    if backends["paddle_adapter"] is not None:
        producers_to_run.append(("paddle", backends["paddle_adapter"]))
    if backends.get("nemotron_adapter") is not None:
        producers_to_run.append(("nemotron", backends["nemotron_adapter"]))
    if backends["qwen_candidate_adapter"] is not None:
        producers_to_run.append(("qwen_proposal", backends["qwen_candidate_adapter"]))

    if ctd_crop_result is not None:
        for producer_name, producer in producers_to_run:
            for region in grouped_page.text_regions:
                region_crops = ctd_crop_result.crop_paths.get(region.id)
                if region_crops is None:
                    continue
                variant = getattr(producer, "preprocessing", "base")
                if variant in {"none", "identity"}:
                    variant = "base"
                crop_path = region_crops.get(variant)
                if crop_path is None:
                    continue
                try:
                    produced = producer.candidates(region, crop_path)
                    all_candidates.extend(produced)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"Producer {producer_name} failed on {region.id}: {exc}")
    timings.setdefault("ocr", []).append(time.perf_counter() - t)

    candidate_bank = build_candidate_bank(grouped_page.text_regions, all_candidates)

    # ── Balloon crops + Qwen adjudication ────────────────────────────────────
    t = _t()
    adjudication_results: list[BalloonAdjudicationResult] = []

    for balloon in grouped_page.balloons:
        if not balloon.text_region_ids:
            adjudication_results.append(BalloonAdjudicationResult(
                balloon_id=balloon.id,
                final_text="",
                text_type="unknown",
                include_in_story=False,
            ))
            continue

        # Create balloon crop
        balloon_crop_path = None
        try:
            balloon_crop_path = balloon_crop_store.create(
                image_path, sequence_id, page_index, balloon
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Balloon crop failed for {balloon.id}: {exc}")

        if balloon_crop_path is None or backends["qwen_adjudicator"] is None:
            # Fallback: use candidate text if available, else placeholder
            texts = []
            for group in candidate_bank.groups:
                if group.region_id in balloon.text_region_ids and group.candidates:
                    best = max(group.candidates, key=lambda c: c.evidence.ocr_confidence or 0.0)
                    if best.text.strip():
                        texts.append(best.text.strip())
            kind_map = {"speech": "dialogue", "thought": "thought", "narration": "narration", "unknown": "dialogue"}
            text_type = kind_map.get(balloon.kind, "dialogue")
            if texts:
                final_text = " ".join(texts)
                include_in_story = True
            elif balloon.text_region_ids:
                final_text = f"[balloon {balloon.id[:24]} -- adjudication unavailable]"
                include_in_story = True
            else:
                final_text = ""
                include_in_story = False
            adjudication_results.append(BalloonAdjudicationResult(
                balloon_id=balloon.id,
                final_text=final_text,
                text_type=text_type,
                include_in_story=include_in_story,
            ))
            continue

        # Real Qwen adjudication
        try:
            result = adjudicate_balloon(
                backends["qwen_adjudicator"],
                balloon,
                candidate_bank,
                grouped_page.text_regions,
                balloon_crop_path,
            )
            adjudication_results.append(result)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Qwen adjudication failed for {balloon.id}: {exc}")
            adjudication_results.append(BalloonAdjudicationResult(
                balloon_id=balloon.id,
                final_text="",
                text_type="unknown",
                include_in_story=False,
            ))

    timings.setdefault("adjudication", []).append(time.perf_counter() - t)

    # ── Character perception ──────────────────────────────────────────────────
    t = _t()
    char_perception = CharacterPerceptionService(
        provider=backends["character_provider"],
        crop_store=char_crop_store,
    )
    try:
        char_result = char_perception.process_page(grouped_page, sequence_id, image_path)
        page_with_chars = char_result.page
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Character perception failed: {exc}")
        page_with_chars = grouped_page
    timings.setdefault("char_detect", []).append(time.perf_counter() - t)

    # ── Speaker grounding ─────────────────────────────────────────────────────
    visual_grounder = None
    if backends.get("qwen_runner") is not None and speaker_context_store is not None:
        try:
            visual_grounder = QwenVisualSpeakerGrounder(backends["qwen_runner"])
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Could not init Qwen visual speaker grounder: {exc}")

    speaker_resolver = SpeakerResolver(visual_grounder=visual_grounder)

    # Build visual contexts: for each story balloon, create a labeled panel image.
    visual_contexts: dict[str, VisualSpeakerContext] = {}
    if speaker_context_store is not None and page_with_chars.characters:
        panel_index = {panel.id: panel for panel in page_with_chars.panels}
        for balloon in page_with_chars.balloons:
            try:
                panel = panel_index.get(balloon.panel_id) if balloon.panel_id else None
                context = speaker_context_store.create(
                    image_path,
                    sequence_id,
                    page_index,
                    balloon,
                    page_with_chars.characters,
                    panel=panel,
                )
                visual_contexts[balloon.id] = context
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Speaker context image failed for {balloon.id}: {exc}")

    try:
        speaker_result = speaker_resolver.resolve_page(
            page_with_chars,
            page_size=(page_w, page_h),
            visual_contexts=visual_contexts or None,
        )
        final_page = speaker_result.page
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Speaker grounding failed: {exc}")
        speaker_result = PageSpeakerGroundingResult(
            page=page_with_chars,
            decisions=[],
            diagnostics=[],
        )
        final_page = page_with_chars

    return final_page, candidate_bank, adjudication_results, speaker_result, (page_w, page_h), warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    # Keep help and progress output usable on the default Windows console.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()

    image_files = sorted(args.image_dir.glob("*.png"))[:3]
    if not image_files:
        raise FileNotFoundError(f"No PNG pages found in {args.image_dir}")

    print(_hr())
    print("  STORY-AI  FULL REAL 3-PAGE PIPELINE INTEGRATION SMOKE")
    print(_hr())
    print(f"  sequence_id   : {args.sequence_id}")
    print(f"  images        : {[f.name for f in image_files]}")
    print()

    status = _ComponentStatus()
    timings: dict[str, list[float]] = {}

    # ── Backend initialization ────────────────────────────────────────────────
    print("  Initializing real local backends …")
    t_init = _t()
    backends = _init_backends(args, status)
    print(f"  Backend initialization: {_elapsed(t_init)}")
    print()

    print("  BACKEND STATUS")
    for name in status.available:
        print(f"    [OK] {name}")
    for name, reason in status.unavailable.items():
        print(f"    [--] {name} -- {reason}")
    print()

    # ── Storage roots ─────────────────────────────────────────────────────────
    smoke_root = args.smoke_dir / args.sequence_id
    ctd_crop_store      = CTDCropStore(smoke_root / "ctd_crops")
    balloon_crop_store  = BalloonCropStore(smoke_root / "balloon_crops")
    char_crop_store     = CharacterCropStore(smoke_root / "char_crops")

    # Speaker context image store (labeled panel crops for Qwen visual grounding)
    use_visual_speaker = not getattr(args, "skip_visual_speaker", False)
    speaker_context_store: SpeakerContextImageStore | None = None
    if use_visual_speaker:
        try:
            speaker_context_store = SpeakerContextImageStore(smoke_root / "speaker_contexts")
            print(f"  Speaker context store: {smoke_root / 'speaker_contexts'}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠  Speaker context store failed: {exc}")

    # ── Per-page processing ───────────────────────────────────────────────────
    pages:           list[PageRepresentation]         = []
    candidate_banks: list[CandidateBank]              = []
    all_adj:         list[BalloonAdjudicationResult]  = []
    speaker_results: list[PageSpeakerGroundingResult] = []
    page_sizes:      list[tuple[int, int]]            = []
    page_stats = []

    try:
        for page_index, image_path in enumerate(image_files):
            print(f"  Processing page {page_index} ({image_path.name}) …", end="", flush=True)
            t_page = _t()
            try:
                (
                    final_page,
                    candidate_bank,
                    adj_results,
                    speaker_result,
                    size,
                    page_warnings,
                ) = _process_page(
                    image_path, page_index, args.sequence_id,
                    backends, ctd_crop_store, balloon_crop_store, char_crop_store,
                    speaker_context_store,
                    timings,
                )
            except Exception as exc:
                print(f" FAILED: {exc}")
                raise

            pages.append(final_page)
            candidate_banks.append(candidate_bank)
            all_adj.extend(adj_results)
            speaker_results.append(speaker_result)
            page_sizes.append(size)

            n_ctd     = len(final_page.text_regions)
            n_balloon = len(final_page.balloons)
            n_story   = sum(1 for r in adj_results if r.include_in_story)
            n_cands   = sum(len(g.candidates) for g in candidate_bank.groups)
            n_chars   = len(final_page.characters)
            page_stats.append({
                "ctd": n_ctd, "balloons": n_balloon, "story": n_story,
                "candidates": n_cands, "characters": n_chars,
            })

            elapsed_page = _elapsed(t_page)
            print(
                f" {n_ctd} CTD / {n_balloon} balloons / {n_story} story / "
                f"{n_cands} candidates / {n_chars} characters  [{elapsed_page}]"
            )
            for w in page_warnings:
                print(f"    [!]  {w}")

    finally:
        # Shut down Nemotron WSL worker (if it was started) regardless of outcome.
        bridge = backends.get("_nemotron_bridge")
        if bridge is not None:
            try:
                bridge.shutdown()
            except Exception:  # noqa: BLE001, S110
                pass

    # ── Reading order ─────────────────────────────────────────────────────────
    t = _t()
    adj_index = {r.balloon_id: r for r in all_adj}
    reading_order = order_sequence(
        pages,
        ranker=DeterministicGeometryRanker(rtl=True),
        adjudication_results=adj_index,
        page_sizes=page_sizes,
    )
    timings.setdefault("reading_order", []).append(time.perf_counter() - t)
    print(f"\n  Reading order: {len(reading_order.ordered_balloon_ids)} story balloon(s)")

    # ── Cross-page character identity ─────────────────────────────────────────
    t = _t()
    crops_by_character: dict = {}
    try:
        # Gather all crop paths from char_crop_store
        for char_id_dir in (smoke_root / "char_crops" / args.sequence_id).rglob("character.png"):
            char_id = char_id_dir.parent.name
            face_path = char_id_dir.parent / "face.png"
            body_path = char_id_dir.parent / "body.png"
            crops_by_character[char_id] = CharacterCropPaths(
                character=char_id_dir,
                face=face_path if face_path.exists() else None,
                body=body_path if body_path.exists() else None,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠  Could not collect character crops for identity: {exc}")

    identity_resolver = CharacterIdentityResolver(
        provider=backends["embedding_provider"],
    )
    identity = identity_resolver.resolve(
        args.sequence_id, pages, crops_by_character
    )
    timings.setdefault("identity", []).append(time.perf_counter() - t)
    print(
        f"  Identity: {len(identity.clusters)} cluster(s), "
        f"{len(identity.pair_evidence)} pair(s), "
        f"{len(identity.diagnostics)} diagnostic(s)"
    )

    # ── Sequence consistency resolver ─────────────────────────────────────────
    t = _t()
    resolver = SequenceConsistencyResolver()
    resolution = resolver.resolve(
        sequence_id=args.sequence_id,
        pages=pages,
        adjudication_results=all_adj,
        reading_order=reading_order,
        speaker_results=speaker_results,
        identity=identity,
    )
    timings.setdefault("resolver", []).append(time.perf_counter() - t)

    # ── Validation ────────────────────────────────────────────────────────────
    known_chars = {char.id for page in pages for char in page.characters}
    validation_errors = validate_sequence_resolution(
        resolution,
        known_character_ids=known_chars,
        expected_sequence_id=args.sequence_id,
    )

    # ── JSONL serialization ───────────────────────────────────────────────────
    jsonl_line = serialize_resolution_to_jsonl(resolution)
    parsed_submission = json.loads(jsonl_line)

    # Speaker contract validation
    all_speaker_valid = all(
        isinstance(item.get("speaker"), str) and item["speaker"].strip()
        for page_items in parsed_submission["pages"]
        for item in page_items
    )
    all_text_valid = all(
        isinstance(item.get("text"), str) and item["text"].strip()
        for page_items in parsed_submission["pages"]
        for item in page_items
    )

    # ── Full report ───────────────────────────────────────────────────────────
    print()
    print(_hr())
    print("  END-TO-END SMOKE REPORT")
    print(_hr())
    print(f"\n  SEQUENCE:  {args.sequence_id}")

    for i, stats in enumerate(page_stats):
        print(f"\n  PAGE {i}:")
        print(f"    CTD regions:    {stats['ctd']}")
        print(f"    balloons:       {stats['balloons']}")
        print(f"    story balloons: {stats['story']}")
        print(f"    candidates:     {stats['candidates']}")

    total_chars = sum(s["characters"] for s in page_stats)
    print("\n  CHARACTERS:")
    print(f"    total: {total_chars}")
    for i, stats in enumerate(page_stats):
        print(f"    page {i}: {stats['characters']}")

    n_resolved   = sum(1 for d in resolution.ordered_balloons if d.speaker_label not in (None, "NARRATION") and d.identity_state != "null")
    n_narration  = sum(1 for d in resolution.ordered_balloons if d.speaker_label == "NARRATION")
    n_unresolved = sum(1 for d in resolution.ordered_balloons if d.speaker_label is None)
    n_ambiguous  = sum(1 for d in resolution.ordered_balloons if d.identity_state == "ambiguous")
    print("\n  SPEAKERS:")
    print(f"    resolved:   {n_resolved}")
    print(f"    narration:  {n_narration}")
    print(f"    unresolved: {n_unresolved}")
    print(f"    ambiguous:  {n_ambiguous}")

    n_matched   = sum(1 for c in identity.clusters if c.state == "matched")
    n_amb_id    = sum(1 for c in identity.clusters if c.state == "ambiguous")
    n_unmatched = sum(1 for c in identity.clusters if c.state == "unmatched")
    print("\n  IDENTITIES:")
    print(f"    clusters:   {len(identity.clusters)}")
    print(f"    matched:    {n_matched}")
    print(f"    ambiguous:  {n_amb_id}")
    print(f"    unmatched:  {n_unmatched}")
    print(f"    pairs:      {len(identity.pair_evidence)}")

    page_boundaries = {item.page_index for item in reading_order.ordered_items}
    print("\n  READING ORDER:")
    print(f"    ordered balloon count: {len(reading_order.ordered_balloon_ids)}")
    print(f"    page boundaries:       {sorted(page_boundaries)}")

    print("\n  SEQUENCE RESOLUTION:")
    print(f"    ordered story balloons: {len(resolution.ordered_balloons)}")
    print(f"    excluded balloons:      {len(resolution.excluded_balloons)}")
    print(f"    unresolved items:       {len(resolution.unresolved_items)}")
    print(f"    resolver diagnostics:   {len(resolution.diagnostics)}")
    if resolution.diagnostics:
        for d in resolution.diagnostics[:5]:
            print(f"      [{d.severity}] {d.code}: {textwrap.shorten(d.message, 60)}")
        if len(resolution.diagnostics) > 5:
            print(f"      … ({len(resolution.diagnostics) - 5} more)")

    print("\n  SUBMISSION:")
    print(f"    serialized successfully: {'yes' if jsonl_line else 'no'}")
    print(f"    exact top-level keys:    {sorted(parsed_submission.keys())}")
    print(f"    exact page count:        {len(parsed_submission['pages'])}")
    print(f"    all speaker strings valid: {all_speaker_valid}")
    print(f"    all text strings valid:    {all_text_valid}")
    total_items = sum(len(p) for p in parsed_submission["pages"])
    print(f"    total serialized items:  {total_items}")

    print(f"\n  VALIDATION:  {len(validation_errors)} error(s)")
    for err in validation_errors:
        print(f"    [FAIL] {err.rule}: {textwrap.shorten(err.message, 70)}")
    if not validation_errors:
        print("    All checks passed.")

    print("\n  APPROXIMATE STAGE RUNTIMES (sum over all pages):")
    stage_names = {
        "ctd":          "CTD localization",
        "layout":       "Layout/balloon grouping",
        "ctd_crops":    "CTD crops",
        "ocr":          "OCR/Qwen proposals",
        "adjudication": "Qwen adjudication",
        "char_detect":  "Character detection",
        "reading_order":"Reading order",
        "identity":     "Character identity",
        "resolver":     "Sequence resolver",
    }
    for key, label in stage_names.items():
        vals = timings.get(key, [])
        if vals:
            total = sum(vals)
            print(f"    {label:<28}: {total:.2f}s  (n={len(vals)})")

    print()
    print(_hr())
    print("  COMPONENT SUMMARY")
    print(_hr())
    print(f"  Real components used ({len(status.available)}):")
    for name in status.available:
        print(f"    [OK] {name}")
    print(f"\n  Unavailable components ({len(status.unavailable)}):")
    for name, reason in status.unavailable.items():
        print(f"    [--] {name}: {reason}")

    print()
    print(_hr())
    # Sample from JSONL output (first 3 items)
    print("  SAMPLE JSONL OUTPUT (first 3 story items):")
    count = 0
    for pi, page_items in enumerate(parsed_submission["pages"]):
        for item in page_items:
            if count >= 3:
                break
            print(f"    page {pi}: speaker={item['speaker']!r}  text={textwrap.shorten(item['text'], 50)!r}")
            count += 1
        if count >= 3:
            break

    print()
    print(_hr())
    verdict = "PASS" if not validation_errors else f"FAIL ({len(validation_errors)} validation error(s))"
    print(f"  RESULT: {verdict}")
    print(_hr())
    return 0 if not validation_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
