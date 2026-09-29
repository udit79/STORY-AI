from pathlib import Path

import pytest
from pydantic import ValidationError

from app.candidate_bank import (
    LayaInput,
    LayaPolicyAdapter,
    build_candidate_bank,
    normalize_candidate,
    produce_candidate_bank,
    to_laya_input,
)
from app.models.candidate_adapters import (
    NemotronCandidateAdapter,
    PaddleOCRCandidateAdapter,
    Qwen3VLCandidateAdapter,
    candidate_producers,
)
from app.schemas.candidates import CandidateEvidence, TranscriptionCandidate
from app.schemas.page import BoundingBox, TextRegion


def make_region(region_id: str = "ctd-17") -> TextRegion:
    return TextRegion(
        id=region_id,
        bbox=BoundingBox(x1=10, y1=20, x2=110, y2=70),
        raw_text="",
        confidence=0.9,
    )


def make_candidate(
    candidate_id: str,
    source: str = "ocr_base",
    text: str = "Hello there!",
    region_id: str = "ctd-17",
) -> TranscriptionCandidate:
    return TranscriptionCandidate(
        candidate_id=candidate_id,
        region_id=region_id,
        source=source,
        text=text,
    )


def test_paddle_candidate_adapter_normalizes_paddle_result(tmp_path: Path):
    class Recognizer:
        def predict(self, *, input: str, batch_size: int):
            assert input == str(tmp_path / "region.png")
            assert batch_size == 1
            return [{"res": {"rec_text": "Hello!", "rec_score": 0.83}}]

    adapter = PaddleOCRCandidateAdapter(Recognizer())
    candidate = adapter.candidates(make_region(), tmp_path / "region.png")[0]

    assert candidate.source == "ocr_base"
    assert candidate.evidence.preprocessing == "base"
    assert candidate.evidence.ocr_confidence == 0.83
    assert candidate.text == "Hello!"


def test_nemotron_and_qwen_adapters_preserve_independent_evidence(tmp_path: Path):
    region = make_region()
    crop = tmp_path / "region.png"
    calls = []

    def nemotron(path: str, *, merge_level: str):
        calls.append((path, merge_level))
        return [{"text": "Hello!", "confidence": 0.72, "left": 0.1,
                 "upper": 0.2, "right": 0.9, "lower": 0.8}]

    paddle_candidate = PaddleOCRCandidateAdapter(
        type("Recognizer", (), {"predict": lambda self, **_: [
            {"rec_text": "Hello!", "rec_score": 0.8}
        ]})()
    ).candidates(region, crop)[0]
    nemotron_candidate = NemotronCandidateAdapter(nemotron).candidates(region, crop)[0]
    qwen_candidate = Qwen3VLCandidateAdapter(
        lambda path: {
            "transcription": "Hello?",
            "candidate_supported": True,
            "visual_confidence": 0.61,
            "text_type": "dialogue",
        }
    ).candidates(region, crop)[0]

    assert calls == [(str(crop), "paragraph")]
    assert {paddle_candidate.source, nemotron_candidate.source, qwen_candidate.source} == {
        "ocr_base", "nemotron", "qwen3_vl"
    }
    assert nemotron_candidate.region_id == qwen_candidate.region_id == region.id
    assert nemotron_candidate.bbox == BoundingBox(x1=20, y1=30, x2=100, y2=60)
    assert nemotron_candidate.evidence.ocr_confidence == 0.72
    assert qwen_candidate.evidence.visual_confidence == 0.61
    assert qwen_candidate.evidence.candidate_supported is True
    assert qwen_candidate.evidence.semantic_type == "dialogue"


def test_evidence_confidence_validation_rejects_invalid_direct_values():
    with pytest.raises(ValidationError):
        CandidateEvidence(ocr_confidence=1.2)
    with pytest.raises(ValidationError):
        CandidateEvidence(visual_confidence=-0.1)


@pytest.mark.parametrize("confidence", [None, "not-a-number", float("nan"), 2.0])
def test_normalization_treats_malformed_confidence_as_missing(confidence):
    candidate = normalize_candidate(
        make_region(),
        "nemotron",
        {"text": "Visible text", "confidence": confidence},
    )

    assert candidate is not None
    assert candidate.evidence.ocr_confidence is None


def test_normalization_preserves_source_and_raw_supporting_metadata():
    candidate = normalize_candidate(
        make_region(),
        "qwen3_vl",
        {
            "transcription": "Wait...",
            "visual_confidence": 0.55,
            "candidate_supported": False,
            "text_type": "thought",
            "raw_output": "model response",
        },
    )

    assert candidate is not None
    assert candidate.source == "qwen3_vl"
    assert candidate.evidence.source_metadata["raw_output"] == "model response"
    assert candidate.evidence.candidate_supported is False
    assert candidate.evidence.semantic_type == "thought"


def test_empty_candidate_text_is_skipped_and_schema_rejects_blank_text():
    assert normalize_candidate(make_region(), "nemotron", {"text": "  "}) is None
    with pytest.raises(ValidationError):
        make_candidate("empty", text="  ")


def test_grouping_uses_region_id_and_keeps_conflicting_sources():
    region = make_region()
    bank = build_candidate_bank(
        [region],
        [
            make_candidate("paddle", "ocr_base", "Hello there!"),
            make_candidate("nemotron", "nemotron", "Hello, there!"),
        ],
    )

    assert len(bank.groups) == 1
    assert bank.groups[0].region_id == region.id
    assert bank.groups[0].bbox == region.bbox
    assert [candidate.source for candidate in bank.groups[0].candidates] == [
        "ocr_base", "nemotron"
    ]


def test_multiple_candidates_from_same_region_are_retained():
    bank = build_candidate_bank(
        [make_region()],
        [
            make_candidate("base", "ocr_base", "Hello there!"),
            make_candidate("upscale", "ocr_upscale", "Hello there."),
            make_candidate("qwen", "qwen3_vl", "Hello there?"),
        ],
    )

    assert len(bank.groups[0].candidates) == 3


def test_deduplication_only_removes_exact_duplicate_evidence_records():
    original = make_candidate("first", "ocr_base", "Same text")
    other_source = make_candidate("nemotron", "nemotron", "Same text")
    bank = build_candidate_bank(
        [make_region()],
        [original, original.model_copy(deep=True), other_source],
    )

    assert bank.groups[0].candidates == [original, other_source]


def test_bank_rejects_candidates_for_unknown_regions():
    with pytest.raises(ValueError, match="unknown CTD region"):
        build_candidate_bank([make_region()], [make_candidate("wrong", region_id="other")])


def test_laya_input_has_comparison_geometry_and_source_features():
    group = build_candidate_bank(
        [make_region()],
        [
            make_candidate("paddle", "ocr_base", "Hello there!"),
            make_candidate("qwen", "qwen3_vl", "Hello there?"),
        ],
    ).groups[0]
    laya_input = to_laya_input(group, page_size=(200, 100))

    assert laya_input.candidate_count == 2
    assert laya_input.source_presence["ocr_base"] is True
    assert laya_input.source_presence["nemotron"] is False
    assert laya_input.region_geometry.width == 100
    assert laya_input.region_geometry.normalized_width == 0.5
    assert laya_input.candidates[0].word_count == 2
    assert laya_input.candidates[0].normalized_text_similarity["qwen"] > 0.8
    assert laya_input.candidates[0].spatial_consistency.region_iou is None


def test_laya_adapter_passes_structured_input_and_requires_explicit_selection():
    class Policy:
        def predict(self, state, questions):
            assert state["region_id"] == "ctd-17"
            assert "raw_image" not in state
            assert set(questions["select_transcription"]["criteria"]) == {
                "paddle", "qwen"
            }
            return {
                "selected_candidate_id": "qwen",
                "decision_confidence": 0.7,
                "reason": "visual support",
                "features": {"agreement": 0.8},
            }

    group = build_candidate_bank(
        [make_region()],
        [make_candidate("paddle"), make_candidate("qwen", "qwen3_vl")],
    ).groups[0]
    decision = LayaPolicyAdapter(Policy()).decide(to_laya_input(group))

    assert decision.region_id == group.region_id
    assert decision.selected_candidate_id == "qwen"
    assert decision.decision_confidence == 0.7
    assert decision.features_used == {"agreement": 0.8}


def test_laya_rejects_unrecognized_or_non_candidate_selection():
    class Policy:
        def predict(self, state, questions):
            return {"selected_candidate_id": "invented"}

    group = build_candidate_bank([make_region()], [make_candidate("paddle")]).groups[0]
    with pytest.raises(ValueError, match="explicitly select"):
        LayaPolicyAdapter(Policy()).decide(to_laya_input(group))


def test_end_to_end_candidate_path_uses_same_ctd_region_for_each_producer(tmp_path):
    region = make_region()
    crop_path = tmp_path / "ctd-17.png"
    producer_list = candidate_producers(
        paddle=PaddleOCRCandidateAdapter(
            type("Recognizer", (), {"predict": lambda self, **_: [
                {"rec_text": "Paddle", "rec_score": 0.8}
            ]})()
        ),
        nemotron=NemotronCandidateAdapter(
            lambda image, **_: [{"text": "Nemotron", "confidence": 0.7}]
        ),
        qwen3_vl=Qwen3VLCandidateAdapter(
            lambda image: {"transcription": "Qwen", "visual_confidence": 0.6}
        ),
    )
    bank = produce_candidate_bank([region], {region.id: crop_path}, producer_list)
    laya_input: LayaInput = to_laya_input(bank.groups[0])

    assert bank.groups[0].region_id == region.id
    assert {candidate.source for candidate in bank.groups[0].candidates} == {
        "ocr_base", "nemotron", "qwen3_vl"
    }
    assert laya_input.candidate_count == 3
    assert laya_input.source_presence["qwen3_vl"] is True


def test_preprocessing_variants_use_their_own_localized_crop(tmp_path):
    region = make_region()
    seen_paths = []

    class Recognizer:
        def predict(self, *, input: str, batch_size: int):
            seen_paths.append(input)
            return [{"rec_text": "Variant", "rec_score": 0.8}]

    producer = PaddleOCRCandidateAdapter(Recognizer(), preprocessing="upscale")
    bank = produce_candidate_bank(
        [region],
        {region.id: {"base": tmp_path / "base.png", "upscale": tmp_path / "upscale.png"}},
        [producer],
    )

    assert seen_paths == [str(tmp_path / "upscale.png")]
    assert bank.groups[0].candidates[0].source == "ocr_upscale"


def test_laya_input_preserves_source_specific_evidence():
    candidate = normalize_candidate(
        make_region(),
        "qwen3_vl",
        {"transcription": "Hello!", "extra_evidence": {"marker": "kept"}},
    )
    group = build_candidate_bank([make_region()], [candidate]).groups[0]

    assert to_laya_input(group).candidates[0].source_metadata["extra_evidence"] == {
        "marker": "kept"
    }