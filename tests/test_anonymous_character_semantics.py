"""Regression coverage for anonymous speaker labels and the full label path."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from app.character_identity import CharacterIdentityResolver, NullEmbeddingProvider
from app.character_perception import CharacterPerceptionService, CharacterProposal
from app.models.character_crops import CharacterCropStore
from app.schemas.adjudication import BalloonAdjudicationResult
from app.schemas.character_identity import (
    CharacterIdentityCluster,
    IdentityMember,
    SequenceCharacterIdentity,
)
from app.schemas.page import (
    Balloon,
    BoundingBox,
    CharacterInstance,
    PageRepresentation,
    Panel,
    Point2D,
)
from app.schemas.reading_order import ReadingOrderItem, SequenceReadingOrder
from app.schemas.resolution import SequenceResolution
from app.sequence_resolver import SequenceConsistencyResolver
from app.speaker_grounding import (
    BalloonTailGeometry,
    PageSpeakerGroundingResult,
    SpeakerDecision,
    SpeakerResolver,
)
from app.submission_serializer import serialize_to_submission


def box(x1=0, y1=0, x2=20, y2=40):
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def page(index, balloons, characters):
    return PageRepresentation(
        page_index=index,
        image_path=f"page_{index}.png",
        panels=[Panel(id=f"panel-{index}", bbox=box(0, 0, 120, 100))],
        balloons=balloons,
        characters=characters,
    )


def balloon(balloon_id, *, text_type="dialogue"):
    return Balloon(id=balloon_id, bbox=box(40, 0, 60, 15), panel_id=None), text_type


def identity_for(clusters):
    cluster_models = [
        CharacterIdentityCluster(
            identity_id=identity_id,
            label=label,
            members=[IdentityMember(character_id=character_id, page_index=page_index)
                     for character_id, page_index in members],
            state=state,
        )
        for identity_id, label, members, state in clusters
    ]
    character_to_identity = {}
    character_to_label = {}
    for cluster in cluster_models:
        for member in cluster.members:
            character_to_identity[member.character_id] = cluster.identity_id
            character_to_label[member.character_id] = cluster.label
    return SequenceCharacterIdentity(
        sequence_id="seq-test",
        clusters=cluster_models,
        character_to_identity=character_to_identity,
        character_to_label=character_to_label,
    )


def resolve(pages, decisions, identity, adjudications):
    speaker_results = []
    for page_index, page_value in enumerate(pages):
        speaker_results.append(PageSpeakerGroundingResult(
            page=page_value,
            decisions=[SpeakerDecision(
                balloon_id=balloon_id,
                selected_character_instance_id=character_id,
                method="geometry",
            ) for balloon_id, character_id in decisions.get(page_index, [])],
        ))
    order = SequenceReadingOrder(
        ordered_items=[
            ReadingOrderItem(
                balloon_id=result.id,
                page_index=page_index,
                panel_id=None,
                sequence_position=position,
            )
            for position, (page_index, result) in enumerate(
                (page_index, item) for page_index, page_value in enumerate(pages)
                for item in page_value.balloons
            )
        ],
        ordered_balloon_ids=[item.id for page_value in pages for item in page_value.balloons],
        page_orders=[],
    )
    return SequenceConsistencyResolver().resolve(
        "seq-test", pages, adjudications, order, speaker_results, identity
    )


def serialized_speakers(resolution: SequenceResolution):
    record = serialize_to_submission(resolution)
    return [item.speaker for page_items in record.pages for item in page_items], record


def test_anonymous_grounded_character_serializes_as_character_a():
    char = CharacterInstance(id="x", bbox=box())
    b, _ = balloon("b0")
    identity = CharacterIdentityResolver(NullEmbeddingProvider()).resolve(
        "seq-test", [page(0, [b], [char]), page(1, [], []), page(2, [], [])]
    )
    resolution = resolve(
        [page(0, [b], [char]), page(1, [], []), page(2, [], [])],
        {0: [("b0", "x")]}, identity,
        [BalloonAdjudicationResult(balloon_id="b0", final_text="Hello", text_type="dialogue", include_in_story=True)],
    )
    speakers, _ = serialized_speakers(resolution)
    assert speakers == ["Character A"]


def test_two_anonymous_characters_get_distinct_labels():
    chars = [CharacterInstance(id="x", bbox=box(10, 10, 30, 50)), CharacterInstance(id="y", bbox=box(70, 10, 90, 50))]
    balloons = [balloon("bx")[0], balloon("by")[0]]
    resolver = CharacterIdentityResolver(NullEmbeddingProvider())
    identity = resolver.resolve(
        "seq-test", [page(0, balloons, chars), page(1, [], []), page(2, [], [])]
    )
    identity_again = resolver.resolve(
        "seq-test", [page(0, balloons, chars), page(1, [], []), page(2, [], [])]
    )
    assert identity.character_to_label == identity_again.character_to_label
    resolution = resolve(
        [page(0, balloons, chars), page(1, [], []), page(2, [], [])],
        {0: [("bx", "x"), ("by", "y")]}, identity,
        [BalloonAdjudicationResult(balloon_id=bid, final_text=bid, text_type="dialogue", include_in_story=True) for bid in ("bx", "by")],
    )
    speakers, _ = serialized_speakers(resolution)
    assert speakers == ["Character A", "Character B"]
    assert "UNKNOWN" not in speakers


def test_same_anonymous_identity_keeps_character_a_across_three_pages():
    chars = [CharacterInstance(id=f"x{index}", bbox=box()) for index in range(3)]
    pages = [page(index, [balloon(f"b{index}")[0]], [chars[index]]) for index in range(3)]
    identity = identity_for([("identity-001", "A", [(f"x{index}", index) for index in range(3)], "matched")])
    resolution = resolve(
        pages,
        {0: [("b0", "x0")], 1: [("b1", "x1")], 2: [("b2", "x2")]},
        identity,
        [BalloonAdjudicationResult(balloon_id=f"b{index}", final_text="line", text_type="dialogue", include_in_story=True) for index in range(3)],
    )
    speakers, _ = serialized_speakers(resolution)
    assert speakers == ["Character A", "Character A", "Character A"]


def test_two_identities_remain_consistent_when_page_detection_order_changes():
    x0, y0 = CharacterInstance(id="x0", bbox=box()), CharacterInstance(id="y0", bbox=box())
    y1, x1 = CharacterInstance(id="y1", bbox=box()), CharacterInstance(id="x1", bbox=box())
    x2, y2 = CharacterInstance(id="x2", bbox=box()), CharacterInstance(id="y2", bbox=box())
    pages = [
        page(0, [balloon("b0x")[0], balloon("b0y")[0]], [x0, y0]),
        page(1, [balloon("b1y")[0], balloon("b1x")[0]], [y1, x1]),
        page(2, [balloon("b2x")[0], balloon("b2y")[0]], [x2, y2]),
    ]
    identity = identity_for([
        ("identity-001", "A", [("x0", 0), ("x1", 1), ("x2", 2)], "matched"),
        ("identity-002", "B", [("y0", 0), ("y1", 1), ("y2", 2)], "matched"),
    ])
    adjudications = [
        BalloonAdjudicationResult(balloon_id=balloon_id, final_text="line", text_type="dialogue", include_in_story=True)
        for balloon_id in ("b0x", "b0y", "b1y", "b1x", "b2x", "b2y")
    ]
    decisions = {
        0: [("b0x", "x0"), ("b0y", "y0")],
        1: [("b1y", "y1"), ("b1x", "x1")],
        2: [("b2x", "x2"), ("b2y", "y2")],
    }
    resolution = resolve(pages, decisions, identity, adjudications)
    speakers, _ = serialized_speakers(resolution)
    assert speakers == ["Character A", "Character B", "Character B", "Character A", "Character A", "Character B"]


def test_genuinely_unresolved_speaker_stays_unknown():
    char = CharacterInstance(id="x", bbox=box())
    b, _ = balloon("b0")
    identity = identity_for([("identity-001", "A", [("x", 0)], "unmatched")])
    resolution = resolve(
        [page(0, [b], [char]), page(1, [], []), page(2, [], [])], {}, identity,
        [BalloonAdjudicationResult(balloon_id="b0", final_text="Hello", text_type="dialogue", include_in_story=True)],
    )
    speakers, _ = serialized_speakers(resolution)
    assert speakers == ["UNKNOWN"]


def test_narration_does_not_route_through_character_identity():
    b, _ = balloon("b0", text_type="narration")
    resolution = resolve(
        [page(0, [b], []), page(1, [], []), page(2, [], [])], {}, None,
        [BalloonAdjudicationResult(balloon_id="b0", final_text="Caption", text_type="narration", include_in_story=True)],
    )
    speakers, _ = serialized_speakers(resolution)
    assert speakers == ["NARRATION"]


class _FixtureCharacterProvider:
    positions = {
        0: [(10, 20, 35, 90), (75, 20, 100, 90)],
        1: [(75, 20, 100, 90), (10, 20, 35, 90)],
        2: [(10, 20, 35, 90), (75, 20, 100, 90)],
    }

    def detect(self, page_image_path, panel_bbox=None):
        page_index = int(Path(page_image_path).stem[-1])
        return [CharacterProposal(bbox=box(*coords), confidence=0.99) for coords in self.positions[page_index]]


class _FixtureEmbeddingProvider:
    def __init__(self, vectors):
        self.vectors = vectors

    def embed(self, image_path):
        return self.vectors[Path(image_path).parent.name]


class _FixtureTailProvider:
    def __init__(self, points):
        self.points = points

    def detect_tail(self, balloon):
        point = self.points.get(balloon.id)
        if point is None:
            return None
        return BalloonTailGeometry(endpoint=Point2D(x=point[0], y=point[1]), method="injected")


def test_end_to_end_anonymous_identity_grounding_and_submission(tmp_path):
    pages = []
    for page_index in range(3):
        image_path = tmp_path / f"page{page_index}.png"
        cv2.imwrite(str(image_path), np.full((100, 120, 3), 255, dtype=np.uint8))
        balloons = [
            Balloon(id=f"p{page_index}-x", bbox=box(5, 0, 20, 15), panel_id=f"panel-{page_index}"),
            Balloon(id=f"p{page_index}-y", bbox=box(90, 0, 110, 15), panel_id=f"panel-{page_index}"),
        ]
        if page_index == 2:
            balloons.extend([
                Balloon(id="narration", bbox=box(40, 0, 60, 15), panel_id=f"panel-{page_index}"),
                Balloon(id="unresolved", bbox=box(45, 0, 65, 15), panel_id=f"panel-{page_index}"),
            ])
        pages.append(PageRepresentation(
            page_index=page_index,
            image_path=str(image_path),
            panels=[Panel(id=f"panel-{page_index}", bbox=box(0, 0, 120, 100))],
            balloons=balloons,
        ))

    perception = CharacterPerceptionService(_FixtureCharacterProvider(), CharacterCropStore(tmp_path / "crops"))
    perceived = [perception.process_page(value, "seq-test", value.image_path) for value in pages]
    pages_with_characters = [value.page for value in perceived]
    assert sum(len(value.characters) for value in pages_with_characters) == 6

    vectors = {}
    for page_value in pages_with_characters:
        for character in page_value.characters:
            vectors[character.id] = [1.0, 0.0] if character.bbox.x1 < 50 else [0.0, 1.0]
    identity = CharacterIdentityResolver(_FixtureEmbeddingProvider(vectors)).resolve("seq-test", pages_with_characters, {
        character.id: perceived[page_value.page_index].crop_paths[character.id]
        for page_value in pages_with_characters for character in page_value.characters
    })
    assert len(identity.clusters) == 2

    tail_points = {
        f"p{page_index}-x": (20 if page_index != 1 else 85, 50) for page_index in range(3)
    } | {
        f"p{page_index}-y": (85 if page_index != 1 else 20, 50) for page_index in range(3)
    }
    grounding = SpeakerResolver(tail_geometry_provider=_FixtureTailProvider(tail_points))
    speaker_results = [grounding.resolve_page(value, page_size=(120, 100)) for value in pages_with_characters]
    decisions = {
        result.page.page_index: result.decisions for result in speaker_results
    }
    selected = {
        decision.balloon_id: decision.selected_character_instance_id
        for result in speaker_results for decision in result.decisions
    }
    assert all(selected[f"p{page_index}-x"] for page_index in range(3))
    assert all(selected[f"p{page_index}-y"] for page_index in range(3))
    assert selected["unresolved"] is None

    adjudications = []
    for page_value in pages_with_characters:
        for item in page_value.balloons:
            text_type = "narration" if item.id == "narration" else "dialogue"
            adjudications.append(BalloonAdjudicationResult(
                balloon_id=item.id,
                final_text=item.id,
                text_type=text_type,
                include_in_story=True,
            ))
    order_items = [
        ReadingOrderItem(balloon_id=item.id, page_index=page_value.page_index, panel_id=item.panel_id, sequence_position=position)
        for position, (page_value, item) in enumerate(
            (page_value, item) for page_value in pages_with_characters for item in page_value.balloons
        )
    ]
    resolution = SequenceConsistencyResolver().resolve(
        "seq-test", pages_with_characters, adjudications,
        SequenceReadingOrder(
            ordered_items=order_items,
            ordered_balloon_ids=[item.balloon_id for item in order_items],
            page_orders=[],
        ),
        speaker_results,
        identity,
    )
    record = serialize_to_submission(resolution)
    payload = record.to_dict()
    assert set(payload) == {"sequence_id", "pages"}
    assert len(payload["pages"]) == 3
    items = [item for page_items in payload["pages"] for item in page_items]
    assert all(item["speaker"] and item["text"] for item in items)
    assert [item["speaker"] for item in payload["pages"][0]] == ["Character A", "Character B"]
    assert [item["speaker"] for item in payload["pages"][1]] == ["Character B", "Character A"]
    assert [item["speaker"] for item in payload["pages"][2]] == ["Character A", "Character B", "NARRATION", "UNKNOWN"]

    json_line = json.dumps(payload, ensure_ascii=False)
    assert json.loads(json_line) == payload
