import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from app.balloon_adjudication import (
    Qwen3VLBalloonAdjudicator,
    adjudicate_balloon,
    build_adjudication_prompt,
    build_balloon_adjudication_input,
)
from app.candidate_bank import build_candidate_bank
from app.models.balloon_crops import BalloonCropStore
from app.schemas.adjudication import (
    BalloonAdjudicationInput,
    BalloonAdjudicationResult,
    BalloonCandidateEvidence,
    BalloonCorrection,
    BalloonRegionReference,
)
from app.schemas.candidates import (
    CandidateBank,
    CandidateEvidence,
    TranscriptionCandidate,
)
from app.schemas.page import Balloon, BoundingBox, TextRegion


def make_region(region_id: str = "ctd-01", bbox: BoundingBox | None = None) -> TextRegion:
    return TextRegion(
        id=region_id,
        bbox=bbox or BoundingBox(x1=10, y1=10, x2=50, y2=30),
        raw_text="",
    )


def make_candidate(
    candidate_id: str,
    region_id: str,
    source: str,
    text: str,
    *,
    ocr_confidence: float | None = None,
    visual_confidence: float | None = None,
) -> TranscriptionCandidate:
    return TranscriptionCandidate(
        candidate_id=candidate_id,
        region_id=region_id,
        source=source,
        text=text,
        evidence=CandidateEvidence(
            ocr_confidence=ocr_confidence,
            visual_confidence=visual_confidence,
            candidate_supported=True if visual_confidence is not None else None,
            semantic_type="dialogue",
            preprocessing="base" if source == "ocr_base" else None,
            source_metadata={"model_detail": f"from-{source}"},
        ),
    )


def make_balloon(*region_ids: str) -> Balloon:
    return Balloon(
        id="balloon-01",
        bbox=BoundingBox(x1=5, y1=5, x2=60, y2=40),
        text_region_ids=list(region_ids or ("ctd-01",)),
    )


def make_input(*, candidates=None, panel_image_path=None) -> BalloonAdjudicationInput:
    region = make_region()
    return BalloonAdjudicationInput(
        balloon_id="balloon-01",
        balloon_bbox=BoundingBox(x1=5, y1=5, x2=60, y2=40),
        balloon_image_path="C:/tmp/balloon.png",
        panel_id="panel-01" if panel_image_path else None,
        panel_bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100) if panel_image_path else None,
        panel_image_path=panel_image_path,
        region_references=[
            BalloonRegionReference(region_id=region.id, bbox=region.bbox)
        ],
        candidates=list(candidates or []),
    )


class FakeRunner:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_json(self, image_paths, prompt):
        self.calls.append((image_paths, prompt))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_input_preserves_disagreeing_candidates_as_separate_evidence():
    region = make_region()
    candidates = [
        make_candidate("paddle-01", region.id, "ocr_base", "UISUL", ocr_confidence=0.9),
        make_candidate("nemotron-01", region.id, "nemotron", "UISUL", ocr_confidence=0.8),
        make_candidate("qwen-01", region.id, "qwen3_vl", "USUAI", visual_confidence=0.6),
    ]
    bank = build_candidate_bank([region], candidates)
    adjudication_input = build_balloon_adjudication_input(
        make_balloon(region.id), bank, [region], "C:/tmp/balloon.png"
    )
    prompt = build_adjudication_prompt(adjudication_input)
    serialized_candidates = json.loads(
        prompt.split("do not vote or collapse them:\n", 1)[1]
        .split("\n\nReturn", 1)[0]
    )

    assert [candidate.source for candidate in adjudication_input.candidates] == [
        "ocr_base", "nemotron", "qwen3_vl"
    ]
    assert [candidate["text"] for candidate in serialized_candidates] == [
        "UISUL", "UISUL", "USUAI"
    ]
    assert "do not vote" in prompt.lower()
    assert adjudication_input.candidates[0].source_metadata["model_detail"] == "from-ocr_base"


def test_adjudicator_passes_only_balloon_and_optional_panel_images():
    runner = FakeRunner({"final_text": "USUAL", "include_in_story": True})
    adjudicator = Qwen3VLBalloonAdjudicator(runner)
    adjudication_input = make_input(panel_image_path="C:/tmp/panel.png")

    result = adjudicator.adjudicate(adjudication_input)

    image_paths, prompt = runner.calls[0]
    assert image_paths == [Path("C:/tmp/balloon.png"), Path("C:/tmp/panel.png")]
    assert "A second image" in prompt
    assert "C:/tmp/page.png" not in prompt
    assert result.final_text == "USUAL"


def test_adjudicator_can_return_new_text_and_trace_corrections():
    candidate = BalloonCandidateEvidence(
        candidate_id="paddle-01",
        region_id="ctd-01",
        source="ocr_base",
        text="UISUL",
    )
    runner = FakeRunner({
        "final_text": "USUAL",
        "text_type": "dialogue",
        "include_in_story": True,
        "confidence": 0.72,
        "corrections": [
            {"candidate_id": "paddle-01", "from_text": "UISUL", "to_text": "USUAL"}
        ],
    })

    result = Qwen3VLBalloonAdjudicator(runner).adjudicate(make_input(candidates=[candidate]))

    assert result.final_text == "USUAL"
    assert result.final_text not in {candidate.text}
    assert result.corrections == [
        BalloonCorrection(candidate_id="paddle-01", from_text="UISUL", to_text="USUAL")
    ]


def test_empty_candidate_bank_still_allows_image_only_adjudication():
    runner = FakeRunner({
        "final_text": "?!",
        "text_type": "vocalisation",
        "include_in_story": True,
    })

    result = Qwen3VLBalloonAdjudicator(runner).adjudicate(make_input())

    assert result.final_text == "?!"
    assert result.include_in_story is True
    assert runner.calls[0][1].endswith("Only cite candidate IDs that were supplied.")


def test_punctuation_only_result_is_valid_story_text():
    result = BalloonAdjudicationResult(
        balloon_id="balloon-01",
        final_text="...?!",
        text_type="thought",
        include_in_story=True,
    )

    assert result.final_text == "...?!"
    assert result.include_in_story is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"balloon_id": "", "balloon_bbox": BoundingBox(x1=0, y1=0, x2=1, y2=1),
         "balloon_image_path": "balloon.png", "region_references": [
             BalloonRegionReference(region_id="r", bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1))
         ]},
        {"balloon_id": "balloon-01", "balloon_bbox": BoundingBox(x1=0, y1=0, x2=1, y2=1),
         "balloon_image_path": "balloon.png", "region_references": [
             BalloonRegionReference(region_id="r", bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1))
         ], "panel_image_path": "  "},
    ],
)
def test_adjudication_input_rejects_empty_ids_or_image_paths(kwargs):
    with pytest.raises(ValidationError):
        BalloonAdjudicationInput(**kwargs)


def test_result_requires_nonempty_text_when_story_included():
    with pytest.raises(ValidationError):
        BalloonAdjudicationResult(
            balloon_id="balloon-01",
            final_text="  ",
            include_in_story=True,
        )


def test_result_rejects_semantic_types_outside_shared_taxonomy():
    with pytest.raises(ValidationError):
        BalloonAdjudicationResult(
            balloon_id="balloon-01",
            final_text="text",
            text_type="exposition",
            include_in_story=True,
        )


def test_malformed_qwen_json_returns_structured_uncertainty():
    result = Qwen3VLBalloonAdjudicator(FakeRunner("not-json")).adjudicate(make_input())

    assert result.final_text == ""
    assert result.include_in_story is False
    assert result.text_type == "unknown"
    assert result.diagnostics[0].code == "malformed_json"


def test_fenced_json_is_parsed():
    result = Qwen3VLBalloonAdjudicator(
        FakeRunner('```json\n{"final_text":"Hello!","include_in_story":true}\n```')
    ).adjudicate(make_input())

    assert result.final_text == "Hello!"
    assert result.include_in_story is True


def test_missing_optional_fields_use_safe_defaults():
    result = Qwen3VLBalloonAdjudicator(
        FakeRunner({"final_text": "Hello!", "include_in_story": True})
    ).adjudicate(make_input())

    assert result.final_text == "Hello!"
    assert result.text_type == "unknown"
    assert result.confidence is None
    assert result.corrections == []
    assert result.notes is None


def test_invalid_confidence_is_omitted_with_diagnostic():
    result = Qwen3VLBalloonAdjudicator(
        FakeRunner({"final_text": "Hello!", "include_in_story": True, "confidence": 2.0})
    ).adjudicate(make_input())

    assert result.final_text == "Hello!"
    assert result.confidence is None
    assert result.diagnostics[0].code == "invalid_confidence"


def test_unknown_correction_ids_are_removed_and_reported():
    result = Qwen3VLBalloonAdjudicator(
        FakeRunner({
            "final_text": "Hello!",
            "include_in_story": True,
            "corrections": [{
                "candidate_id": "not-in-bank",
                "from_text": "helo",
                "to_text": "Hello",
            }],
        })
    ).adjudicate(make_input())

    assert result.corrections == []
    assert result.diagnostics[0].code == "unknown_correction_candidate"


def test_backend_failure_returns_structured_uncertainty():
    result = Qwen3VLBalloonAdjudicator(FakeRunner(RuntimeError("offline"))).adjudicate(
        make_input()
    )

    assert result.include_in_story is False
    assert result.diagnostics[0].code == "backend_failure"


def test_adjudicator_result_is_deterministic_for_same_backend_response():
    response = {
        "final_text": "WHAT?!",
        "text_type": "dialogue",
        "include_in_story": True,
        "confidence": 0.5,
    }
    first = Qwen3VLBalloonAdjudicator(FakeRunner(response)).adjudicate(make_input())
    second = Qwen3VLBalloonAdjudicator(FakeRunner(response)).adjudicate(make_input())

    assert first == second


def test_balloon_input_aggregates_all_fragment_candidates_without_fusion():
    first_region = make_region("ctd-a", BoundingBox(x1=10, y1=10, x2=40, y2=25))
    second_region = make_region("ctd-b", BoundingBox(x1=10, y1=27, x2=40, y2=42))
    candidates = [
        make_candidate("a-paddle", "ctd-a", "ocr_base", "GOOD"),
        make_candidate("a-nemotron", "ctd-a", "nemotron", "G00D"),
        make_candidate("b-paddle", "ctd-b", "ocr_base", "LUCK!"),
    ]
    bank = build_candidate_bank([first_region, second_region], candidates)
    balloon = make_balloon(first_region.id, second_region.id)
    adjudication_input = build_balloon_adjudication_input(
        balloon, bank, [first_region, second_region], "balloon.png"
    )

    assert [reference.region_id for reference in adjudication_input.region_references] == [
        "ctd-a", "ctd-b"
    ]
    assert [candidate.candidate_id for candidate in adjudication_input.candidates] == [
        "a-paddle", "a-nemotron", "b-paddle"
    ]
    assert adjudication_input.candidates[0].similarity_to_candidates["a-nemotron"] >= 0.5


def test_missing_region_reference_rejects_input_instead_of_losing_provenance():
    with pytest.raises(ValueError, match="missing CTD regions"):
        build_balloon_adjudication_input(
            make_balloon("ctd-missing"), CandidateBank(), [], "balloon.png"
        )


def test_balloon_crop_store_is_deterministic_and_clamps_to_page(tmp_path):
    page_path = tmp_path / "page.png"
    image = np.full((10, 12, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(page_path), image)
    balloon = Balloon(
        id="balloon:one",
        bbox=BoundingBox(x1=-2.1, y1=2, x2=8.2, y2=20),
        text_region_ids=["ctd-1"],
    )
    store = BalloonCropStore(tmp_path / "crops")

    first_path = store.create(page_path, "seq-one", 0, balloon)
    second_path = store.create(page_path, "seq-one", 0, balloon)
    crop = cv2.imread(str(first_path))

    assert first_path == second_path
    assert first_path.name == "balloon.png"
    assert crop.shape[:2] == (8, 9)
    assert balloon.bbox == BoundingBox(x1=-2.1, y1=2, x2=8.2, y2=20)


def test_end_to_end_service_uses_balloon_crop_and_candidate_bank_only():
    region = make_region()
    candidate = make_candidate("paddle-01", region.id, "ocr_base", "UISUL")
    bank = build_candidate_bank([region], [candidate])
    balloon = make_balloon(region.id)
    runner = FakeRunner({"final_text": "USUAL", "include_in_story": True})
    adjudicator = Qwen3VLBalloonAdjudicator(runner)
    result = adjudicate_balloon(
        adjudicator,
        balloon,
        bank,
        [region],
        "C:/tmp/balloon.png",
    )

    image_paths, prompt = runner.calls[0]
    assert image_paths == [Path("C:/tmp/balloon.png")]
    assert "UISUL" in prompt
    assert "page.png" not in prompt
    assert result.balloon_id == balloon.id
    assert result.final_text == "USUAL"