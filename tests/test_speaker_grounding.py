from pathlib import Path

import cv2
import numpy as np

from app.schemas.page import (
    Balloon,
    BoundingBox,
    CharacterInstance,
    PageRepresentation,
    Panel,
    Point2D,
)
from app.speaker_grounding import (
    BalloonTailGeometry,
    MaskPolygonTailGeometryProvider,
    QwenVisualSpeakerGrounder,
    SpeakerCandidate,
    SpeakerCandidateEvidence,
    SpeakerContextImageStore,
    SpeakerResolver,
    VisualSpeakerContext,
    VisualSpeakerGrounding,
    generate_speaker_candidates,
)


def make_balloon(balloon_id="balloon-01", panel_id=None):
    return Balloon(
        id=balloon_id,
        bbox=BoundingBox(x1=40, y1=10, x2=65, y2=30),
        panel_id=panel_id,
        text_region_ids=["ctd-01"],
    )


def make_character(character_id, bbox, panel_id=None):
    return CharacterInstance(id=character_id, bbox=bbox, panel_id=panel_id)


def make_candidate(balloon_id, character_id, **evidence):
    return SpeakerCandidate(
        balloon_id=balloon_id,
        character_instance_id=character_id,
        evidence=SpeakerCandidateEvidence(**evidence),
    )


class FakeRunner:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_json(self, image_paths, prompt):
        self.calls.append((image_paths, prompt))
        return self.response


def test_same_panel_candidates_keep_instance_ids_and_panel_evidence():
    balloon = make_balloon(panel_id="panel-1")
    panels = [Panel(
        id="panel-1",
        bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
        source="layout_model",
    )]
    characters = [
        make_character("char-page1-001", BoundingBox(x1=5, y1=30, x2=20, y2=80), "panel-1"),
        make_character("char-page1-002", BoundingBox(x1=70, y1=30, x2=90, y2=80), "panel-1"),
        make_character("char-page1-003", BoundingBox(x1=101, y1=30, x2=115, y2=80), "panel-2"),
    ]

    candidates = generate_speaker_candidates(
        balloon, characters, panels, page_size=(120, 100)
    )

    assert {candidate.character_instance_id for candidate in candidates} == {
        "char-page1-001", "char-page1-002"
    }
    assert all(candidate.evidence.same_panel is True for candidate in candidates)


def test_page_fallback_uses_page_level_candidates_without_panel_claims():
    balloon = make_balloon(panel_id="panel-fallback")
    fallback_panel = Panel(
        id="panel-fallback",
        bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
        source="page_fallback",
    )
    characters = [
        make_character("char-1", BoundingBox(x1=5, y1=35, x2=20, y2=80)),
        make_character("char-2", BoundingBox(x1=75, y1=35, x2=90, y2=80)),
    ]

    candidates = generate_speaker_candidates(balloon, characters, [fallback_panel])

    assert len(candidates) == 2
    assert all(candidate.evidence.same_panel is None for candidate in candidates)


def test_candidate_geometry_includes_distance_overlap_and_face_body_features():
    balloon = make_balloon()
    character = CharacterInstance(
        id="char-1",
        bbox=BoundingBox(x1=70, y1=35, x2=90, y2=85),
        face_bbox=BoundingBox(x1=73, y1=37, x2=87, y2=54),
        body_bbox=BoundingBox(x1=70, y1=54, x2=90, y2=85),
    )

    candidate = generate_speaker_candidates(
        balloon, [character], [], page_size=(120, 100)
    )[0]

    assert candidate.evidence.center_distance_px is not None
    assert candidate.evidence.normalized_center_distance is not None
    assert candidate.evidence.balloon_character_overlap == 0.0
    assert candidate.evidence.face_available is True
    assert candidate.evidence.body_available is True


def test_tail_endpoint_evidence_can_resolve_a_speaker_without_closest_only_rule():
    balloon = make_balloon()
    first = make_character("char-tail", BoundingBox(x1=65, y1=20, x2=90, y2=70))
    second = make_character("char-near", BoundingBox(x1=100, y1=20, x2=120, y2=70))
    tail = BalloonTailGeometry(
        endpoint=Point2D(x=70, y=40),
        method="injected",
        evidence={"tip_supported": True},
    )
    candidates = generate_speaker_candidates(balloon, [first, second], [], tail_geometry=tail)

    assert candidates[0].evidence.tail_endpoint_inside_character is True
    decision = SpeakerResolver().resolve(balloon, candidates, tail_geometry=tail)

    assert decision.selected_character_instance_id == "char-tail"
    assert decision.method == "geometry"
    assert decision.evidence["rule"] == "unique_tail_endpoint_inside_character"


def test_missing_tail_and_multiple_characters_remains_unresolved_not_nearest_wins():
    balloon = make_balloon()
    characters = [
        make_character("char-near", BoundingBox(x1=65, y1=20, x2=90, y2=70)),
        make_character("char-far", BoundingBox(x1=100, y1=20, x2=115, y2=70)),
    ]
    candidates = generate_speaker_candidates(balloon, characters, [], page_size=(120, 100))

    decision = SpeakerResolver().resolve(balloon, candidates, tail_geometry=None)

    assert decision.selected_character_instance_id is None
    assert decision.method == "unknown"
    assert len(decision.candidates) == 2
    assert decision.evidence["selection_rule"] == "distance alone never selects a speaker"


def test_only_character_in_detected_panel_can_be_geometry_candidate():
    balloon = make_balloon(panel_id="panel-1")
    panel = Panel(
        id="panel-1", bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100), source="layout_model"
    )
    character = make_character(
        "char-only", BoundingBox(x1=70, y1=30, x2=90, y2=80), panel_id="panel-1"
    )
    candidate = generate_speaker_candidates(balloon, [character], [panel])[0]

    decision = SpeakerResolver().resolve(balloon, [candidate])

    assert decision.selected_character_instance_id == "char-only"
    assert decision.method == "geometry"
    assert decision.evidence["rule"] == "only_visible_character_instance_in_panel"


def test_visual_grounding_is_only_called_when_geometry_is_ambiguous():
    balloon = make_balloon(panel_id="panel-1")
    characters = [
        make_character("char-1", BoundingBox(x1=10, y1=20, x2=30, y2=70), "panel-1"),
        make_character("char-2", BoundingBox(x1=75, y1=20, x2=95, y2=70), "panel-1"),
    ]
    panel = Panel(
        id="panel-1", bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100), source="layout_model"
    )
    candidates = generate_speaker_candidates(balloon, characters, [panel])

    class Grounder:
        def __init__(self):
            self.calls = 0

        def resolve(self, balloon_id, candidates, context):
            self.calls += 1
            return VisualSpeakerGrounding(
                character_instance_id="char-2",
                confidence=0.84,
                reason="Labeled tail points to C2.",
            )

    grounder = Grounder()
    resolver = SpeakerResolver(grounder)
    context = VisualSpeakerContext(
        image_paths=["annotated.png"],
        character_labels={"C1": "char-1", "C2": "char-2"},
    )

    decision = resolver.resolve(balloon, candidates, visual_context=context)

    assert grounder.calls == 1
    assert decision.selected_character_instance_id == "char-2"
    assert decision.method == "hybrid"


def test_invalid_qwen_character_id_is_rejected():
    balloon = make_balloon()
    candidates = [make_candidate(balloon.id, "char-1")]
    grounder = QwenVisualSpeakerGrounder(
        FakeRunner({"character_instance_id": "char-invented", "confidence": 0.99})
    )

    result = grounder.resolve(
        balloon.id,
        candidates,
        VisualSpeakerContext(image_paths=["marked.png"], character_labels={"C1": "char-1"}),
    )

    assert result.character_instance_id is None
    assert result.diagnostics[0].code == "invalid_visual_character_id"


def test_qwen_grounder_prompt_and_images_include_only_supplied_candidates():
    balloon = make_balloon()
    candidates = [make_candidate(balloon.id, "char-1"), make_candidate(balloon.id, "char-2")]
    runner = FakeRunner({"character_instance_id": "char-2", "confidence": 0.83})
    grounder = QwenVisualSpeakerGrounder(runner)
    context = VisualSpeakerContext(
        image_paths=["balloon-crop.png", "labeled-panel.png"],
        character_labels={"C1": "char-1", "C2": "char-2"},
    )

    result = grounder.resolve(balloon.id, candidates, context)

    image_paths, prompt = runner.calls[0]
    assert image_paths == [Path("balloon-crop.png"), Path("labeled-panel.png")]
    assert "char-1" in prompt and "char-2" in prompt
    assert "page.png" not in prompt
    assert result.character_instance_id == "char-2"


def test_mask_tail_provider_returns_only_a_supported_sharp_polygon_tip():
    balloon = Balloon(
        id="balloon-tail",
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=20),
        mask_polygon=[
            Point2D(x=0, y=0),
            Point2D(x=10, y=0),
            Point2D(x=10, y=10),
            Point2D(x=6, y=10),
            Point2D(x=5, y=20),
            Point2D(x=4, y=10),
            Point2D(x=0, y=10),
        ],
    )
    rectangle = Balloon(
        id="balloon-no-tail",
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=20),
        mask_polygon=[
            Point2D(x=0, y=0),
            Point2D(x=10, y=0),
            Point2D(x=10, y=20),
            Point2D(x=0, y=20),
        ],
    )
    provider = MaskPolygonTailGeometryProvider()

    tail = provider.detect_tail(balloon)

    assert tail is not None
    assert tail.endpoint == Point2D(x=5, y=20)
    assert tail.evidence["tip_supported"] is True
    assert provider.detect_tail(rectangle) is None


def test_resolve_page_populates_candidates_without_mutating_reading_order():
    balloon = make_balloon(panel_id=None)
    character = make_character("char-1", BoundingBox(x1=70, y1=30, x2=90, y2=80))
    page = PageRepresentation(
        page_index=0,
        image_path="unused.png",
        balloons=[balloon],
        characters=[character],
        reading_order=["keep-later-layer"],
    )

    result = SpeakerResolver().resolve_page(page, page_size=(120, 100))

    assert result.page.balloons[0].candidate_character_ids == [character.id]
    assert result.page.reading_order == ["keep-later-layer"]
    assert result.decisions[0].selected_character_instance_id is None
    assert result.decisions[0].method == "unknown"


def test_context_image_marks_balloon_and_candidate_character_ids(tmp_path):
    page_path = tmp_path / "page.png"
    image = np.full((80, 100, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(page_path), image)
    balloon = Balloon(
        id="balloon-context",
        bbox=BoundingBox(x1=40, y1=5, x2=75, y2=30),
        text_region_ids=["ctd-01"],
    )
    characters = [
        make_character("char-instance-1", BoundingBox(x1=5, y1=30, x2=30, y2=75)),
        make_character("char-instance-2", BoundingBox(x1=70, y1=30, x2=95, y2=75)),
    ]

    context = SpeakerContextImageStore(tmp_path / "context").create(
        page_path,
        "seq",
        0,
        balloon,
        characters,
    )

    assert context.character_labels == {
        "C1": "char-instance-1",
        "C2": "char-instance-2",
    }
    assert Path(context.image_paths[0]).is_file()