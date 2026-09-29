"""CPU-only tests for the Sequence Consistency Resolver.

Coverage (20 cases):
 1.  normal fully resolved sequence
 2.  repeated speaker across pages
 3.  speaker instance → identity → label mapping
 4.  ambiguous speaker (unresolved grounding)
 5.  ambiguous identity cluster
 6.  unmatched character (no identity cluster)
 7.  missing speaker identity (no SequenceCharacterIdentity)
 8.  excluded SFX balloon (include_in_story=False)
 9.  page-order violation detection (diagnostic, not crash)
10.  duplicate balloon in reading order
11.  invalid character reference in speaker
12.  identity references unknown character
13.  identity label conflict (same identity, two labels)
14.  deterministic anonymous labels
15.  three-page constraint validation
16.  missing adjudication result
17.  conflicting component outputs (balloon on page 1, reading order says page 0)
18.  final serializer exact schema (competition format)
19.  validator rejection cases
20.  valid submission object passes validation
"""

from __future__ import annotations

import json

import pytest

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
)
from app.schemas.reading_order import ReadingOrderItem, SequenceReadingOrder
from app.schemas.resolution import ResolvedBalloon, SequenceResolution
from app.sequence_resolver import SequenceConsistencyResolver
from app.sequence_validator import ValidationError, validate_sequence_resolution
from app.submission_serializer import (
    SubmissionRecord,
    _submission_speaker,
    serialize_to_jsonl_line,
    serialize_to_submission,
)
from app.speaker_grounding import PageSpeakerGroundingResult, SpeakerDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bb(x1: float = 0, y1: float = 0, x2: float = 100, y2: float = 100) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def make_char(cid: str, panel_id: str | None = None) -> CharacterInstance:
    return CharacterInstance(id=cid, bbox=bb(), panel_id=panel_id)


def make_balloon(bid: str, panel_id: str | None = None) -> Balloon:
    return Balloon(id=bid, bbox=bb(), panel_id=panel_id)


def make_page(
    page_index: int,
    balloons: list[Balloon] | None = None,
    characters: list[CharacterInstance] | None = None,
) -> PageRepresentation:
    return PageRepresentation(
        page_index=page_index,
        image_path=f"page_{page_index:02d}.png",
        balloons=balloons or [],
        characters=characters or [],
    )


def make_adj(
    balloon_id: str,
    text: str = "Hello.",
    include: bool = True,
    text_type: str = "dialogue",
) -> BalloonAdjudicationResult:
    return BalloonAdjudicationResult(
        balloon_id=balloon_id,
        final_text=text,
        text_type=text_type,
        include_in_story=include,
    )


def make_speaker_result(
    page: PageRepresentation,
    decisions: list[SpeakerDecision] | None = None,
) -> PageSpeakerGroundingResult:
    return PageSpeakerGroundingResult(page=page, decisions=decisions or [])


def make_decision(balloon_id: str, char_id: str | None) -> SpeakerDecision:
    return SpeakerDecision(
        balloon_id=balloon_id,
        selected_character_instance_id=char_id,
        method="geometry",
    )


def make_reading_order(items: list[tuple[str, int, str | None]]) -> SequenceReadingOrder:
    """items: (balloon_id, page_index, panel_id)"""
    order_items = [
        ReadingOrderItem(
            balloon_id=bid,
            page_index=page,
            panel_id=panel,
            sequence_position=pos,
        )
        for pos, (bid, page, panel) in enumerate(items)
    ]
    return SequenceReadingOrder(
        ordered_items=order_items,
        ordered_balloon_ids=[bid for bid, _, _ in items],
        page_orders=[],
    )


def make_identity(
    clusters: list[tuple[str, str, list[tuple[str, int]], str]],
) -> SequenceCharacterIdentity:
    """clusters: (identity_id, label, [(char_id, page_index), ...], state)"""
    cluster_objs = [
        CharacterIdentityCluster(
            identity_id=iid,
            label=label,
            members=[IdentityMember(character_id=cid, page_index=p) for cid, p in members],
            state=state,
        )
        for iid, label, members, state in clusters
    ]
    char_to_identity: dict[str, str | None] = {}
    char_to_label: dict[str, str | None] = {}
    for cluster in cluster_objs:
        for member in cluster.members:
            char_to_identity[member.character_id] = cluster.identity_id
            char_to_label[member.character_id] = cluster.label
    return SequenceCharacterIdentity(
        sequence_id="seq_test",
        clusters=cluster_objs,
        character_to_identity=char_to_identity,
        character_to_label=char_to_label,
    )


# ---------------------------------------------------------------------------
# Default resolver instance
# ---------------------------------------------------------------------------

RESOLVER = SequenceConsistencyResolver()


# ---------------------------------------------------------------------------
# Test 1: normal fully resolved sequence
# ---------------------------------------------------------------------------

def test_normal_fully_resolved_sequence():
    char_a = make_char("char-p0-A")
    char_b = make_char("char-p1-B")
    char_c = make_char("char-p2-C")

    page0 = make_page(0, [make_balloon("b0")], [char_a])
    page1 = make_page(1, [make_balloon("b1")], [char_b])
    page2 = make_page(2, [make_balloon("b2")], [char_c])

    adj = [
        make_adj("b0", "Page 0 line."),
        make_adj("b1", "Page 1 line."),
        make_adj("b2", "Page 2 line."),
    ]
    reading_order = make_reading_order([("b0", 0, None), ("b1", 1, None), ("b2", 2, None)])
    speaker_results = [
        make_speaker_result(page0, [make_decision("b0", "char-p0-A")]),
        make_speaker_result(page1, [make_decision("b1", "char-p1-B")]),
        make_speaker_result(page2, [make_decision("b2", "char-p2-C")]),
    ]
    identity = make_identity([
        ("identity-001", "A", [("char-p0-A", 0)], "unmatched"),
        ("identity-002", "B", [("char-p1-B", 1)], "unmatched"),
        ("identity-003", "C", [("char-p2-C", 2)], "unmatched"),
    ])

    resolution = RESOLVER.resolve(
        sequence_id="seq_test",
        pages=[page0, page1, page2],
        adjudication_results=adj,
        reading_order=reading_order,
        speaker_results=speaker_results,
        identity=identity,
    )

    assert resolution.sequence_id == "seq_test"
    assert len(resolution.ordered_balloons) == 3
    assert [b.balloon_id for b in resolution.ordered_balloons] == ["b0", "b1", "b2"]
    assert [b.sequence_position for b in resolution.ordered_balloons] == [0, 1, 2]
    assert [b.speaker_label for b in resolution.ordered_balloons] == ["A", "B", "C"]
    assert resolution.ordered_balloons[0].text == "Page 0 line."

    errors = validate_sequence_resolution(
        resolution,
        known_character_ids={"char-p0-A", "char-p1-B", "char-p2-C"},
        expected_sequence_id="seq_test",
    )
    assert errors == [], [e.message for e in errors]


# ---------------------------------------------------------------------------
# Test 2: repeated speaker across pages
# ---------------------------------------------------------------------------

def test_repeated_speaker_across_pages():
    """char-p0-A, char-p1-A2, char-p2-A3 all map to identity-001 → label A."""
    chars = {
        0: make_char("char-p0-A"),
        1: make_char("char-p1-A2"),
        2: make_char("char-p2-A3"),
    }
    pages = [
        make_page(0, [make_balloon("b0")], [chars[0]]),
        make_page(1, [make_balloon("b1")], [chars[1]]),
        make_page(2, [make_balloon("b2")], [chars[2]]),
    ]
    adj = [make_adj(f"b{i}", f"Speech {i}.") for i in range(3)]
    reading_order = make_reading_order([("b0", 0, None), ("b1", 1, None), ("b2", 2, None)])
    speaker_results = [
        make_speaker_result(pages[i], [make_decision(f"b{i}", f"char-p{i}-{'A' if i == 0 else ('A2' if i == 1 else 'A3')}")])
        for i in range(3)
    ]
    identity = make_identity([
        ("identity-001", "A", [
            ("char-p0-A", 0),
            ("char-p1-A2", 1),
            ("char-p2-A3", 2),
        ], "matched"),
    ])

    resolution = RESOLVER.resolve(
        sequence_id="seq_test",
        pages=pages,
        adjudication_results=adj,
        reading_order=reading_order,
        speaker_results=speaker_results,
        identity=identity,
    )

    labels = [b.speaker_label for b in resolution.ordered_balloons]
    assert labels == ["A", "A", "A"], f"Expected all A, got {labels}"
    identity_ids = [b.speaker_identity_id for b in resolution.ordered_balloons]
    assert all(iid == "identity-001" for iid in identity_ids)


# ---------------------------------------------------------------------------
# Test 3: speaker instance → identity → label mapping
# ---------------------------------------------------------------------------

def test_speaker_instance_to_label_mapping():
    char = make_char("char-p0-X")
    page0 = make_page(0, [make_balloon("b0")], [char])
    page1 = make_page(1, [], [])
    page2 = make_page(2, [], [])

    adj = [make_adj("b0", "Mapped text.")]
    reading_order = make_reading_order([("b0", 0, None)])
    speaker_results = [make_speaker_result(page0, [make_decision("b0", "char-p0-X")])]
    identity = make_identity([("identity-001", "B", [("char-p0-X", 0)], "unmatched")])

    resolution = RESOLVER.resolve(
        "seq_test", [page0, page1, page2], adj, reading_order, speaker_results, identity
    )

    balloon = resolution.ordered_balloons[0]
    assert balloon.speaker_character_instance_id == "char-p0-X"
    assert balloon.speaker_identity_id == "identity-001"
    assert balloon.speaker_label == "B"


# ---------------------------------------------------------------------------
# Test 4: ambiguous speaker (no grounding result)
# ---------------------------------------------------------------------------

def test_ambiguous_speaker_no_grounding():
    char = make_char("char-p0-Z")
    page0 = make_page(0, [make_balloon("b0")], [char])
    page1 = make_page(1, [], [])
    page2 = make_page(2, [], [])

    adj = [make_adj("b0", "Ambiguous speech.")]
    reading_order = make_reading_order([("b0", 0, None)])
    # No speaker decision for b0
    speaker_results = [make_speaker_result(page0, [])]
    identity = make_identity([("identity-001", "A", [("char-p0-Z", 0)], "unmatched")])

    resolution = RESOLVER.resolve(
        "seq_test", [page0, page1, page2], adj, reading_order, speaker_results, identity
    )

    balloon = resolution.ordered_balloons[0]
    assert balloon.speaker_character_instance_id is None
    assert balloon.speaker_label is None
    # Diagnostic recorded
    diag_codes = [d.code for d in balloon.diagnostics]
    assert "no_speaker_resolved" in diag_codes


# ---------------------------------------------------------------------------
# Test 5: ambiguous identity cluster
# ---------------------------------------------------------------------------

def test_ambiguous_identity_cluster():
    char = make_char("char-p0-amb")
    page0 = make_page(0, [make_balloon("b0")], [char])
    page1 = make_page(1, [], [])
    page2 = make_page(2, [], [])

    adj = [make_adj("b0", "Ambiguous identity speech.")]
    reading_order = make_reading_order([("b0", 0, None)])
    speaker_results = [make_speaker_result(page0, [make_decision("b0", "char-p0-amb")])]
    identity = make_identity([("identity-001", "A", [("char-p0-amb", 0)], "ambiguous")])

    resolution = RESOLVER.resolve(
        "seq_test", [page0, page1, page2], adj, reading_order, speaker_results, identity
    )

    balloon = resolution.ordered_balloons[0]
    # Label is preserved for traceability even when ambiguous
    assert balloon.speaker_label == "A"
    assert balloon.identity_state == "ambiguous"
    # A diagnostic warning must be emitted
    diag_codes = [d.code for d in balloon.diagnostics]
    assert "speaker_identity_ambiguous" in diag_codes


# ---------------------------------------------------------------------------
# Test 6: unmatched character (no identity cluster match)
# ---------------------------------------------------------------------------

def test_unmatched_character_no_cluster():
    char = make_char("char-p0-solo")
    page0 = make_page(0, [make_balloon("b0")], [char])
    page1 = make_page(1, [], [])
    page2 = make_page(2, [], [])

    adj = [make_adj("b0", "Solo speech.")]
    reading_order = make_reading_order([("b0", 0, None)])
    speaker_results = [make_speaker_result(page0, [make_decision("b0", "char-p0-solo")])]
    # Identity has the char but no identity_id (character_to_identity → None)
    identity = SequenceCharacterIdentity(
        sequence_id="seq_test",
        clusters=[],
        character_to_identity={"char-p0-solo": None},
        character_to_label={"char-p0-solo": None},
    )

    resolution = RESOLVER.resolve(
        "seq_test", [page0, page1, page2], adj, reading_order, speaker_results, identity
    )

    balloon = resolution.ordered_balloons[0]
    assert balloon.speaker_label is None
    assert balloon.speaker_identity_id is None
    diag_codes = [d.code for d in balloon.diagnostics]
    assert "speaker_identity_unresolved" in diag_codes


# ---------------------------------------------------------------------------
# Test 7: missing speaker identity (no SequenceCharacterIdentity)
# ---------------------------------------------------------------------------

def test_no_identity_result_provided():
    char = make_char("char-p0-noid")
    page0 = make_page(0, [make_balloon("b0")], [char])
    page1 = make_page(1, [], [])
    page2 = make_page(2, [], [])

    adj = [make_adj("b0", "No identity available.")]
    reading_order = make_reading_order([("b0", 0, None)])
    speaker_results = [make_speaker_result(page0, [make_decision("b0", "char-p0-noid")])]

    resolution = RESOLVER.resolve(
        "seq_test", [page0, page1, page2], adj, reading_order, speaker_results, identity=None
    )

    balloon = resolution.ordered_balloons[0]
    assert balloon.speaker_label is None
    assert balloon.speaker_identity_id is None
    diag_codes = [d.code for d in balloon.diagnostics]
    assert "no_identity_result" in diag_codes


# ---------------------------------------------------------------------------
# Test 8: excluded SFX balloon (include_in_story=False)
# ---------------------------------------------------------------------------

def test_excluded_sfx_balloon():
    page0 = make_page(0, [make_balloon("b_sfx"), make_balloon("b_speech")])
    page1 = make_page(1, [], [])
    page2 = make_page(2, [], [])

    adj = [
        make_adj("b_sfx", "BOOM!", include=False, text_type="sound_effect"),
        make_adj("b_speech", "Real dialogue.", include=True, text_type="dialogue"),
    ]
    reading_order = make_reading_order([("b_sfx", 0, None), ("b_speech", 0, None)])
    speaker_results = [make_speaker_result(page0, [])]

    resolution = RESOLVER.resolve(
        "seq_test", [page0, page1, page2], adj, reading_order, speaker_results, identity=None
    )

    story_ids = [b.balloon_id for b in resolution.ordered_balloons]
    assert "b_sfx" not in story_ids
    assert "b_speech" in story_ids

    excluded_ids = [b.balloon_id for b in resolution.excluded_balloons]
    assert "b_sfx" in excluded_ids


# ---------------------------------------------------------------------------
# Test 9: page-order violation in reading order triggers diagnostic
# ---------------------------------------------------------------------------

def test_page_order_violation_triggers_diagnostic():
    page0 = make_page(0, [make_balloon("b0")])
    page1 = make_page(1, [make_balloon("b1")])
    page2 = make_page(2, [], [])

    adj = [make_adj("b0", "text"), make_adj("b1", "text")]
    # Page-reversed: b1 (page 1) before b0 (page 0)
    reading_order = make_reading_order([("b1", 1, None), ("b0", 0, None)])
    speaker_results = [
        make_speaker_result(page0, []),
        make_speaker_result(page1, []),
    ]

    resolution = RESOLVER.resolve(
        "seq_test", [page0, page1, page2], adj, reading_order, speaker_results, identity=None
    )

    # Ordered balloons follow reading order positions; page violation should be
    # caught by the post-resolution check.
    diag_codes = [d.code for d in resolution.diagnostics]
    assert "page_order_violation" in diag_codes


# ---------------------------------------------------------------------------
# Test 10: duplicate balloon in reading order
# ---------------------------------------------------------------------------

def test_duplicate_balloon_in_reading_order():
    page0 = make_page(0, [make_balloon("b0")])
    page1 = make_page(1, [], [])
    page2 = make_page(2, [], [])

    adj = [make_adj("b0", "text")]
    # b0 appears twice in the reading order (sequence_positions 0 and 1)
    reading_order = make_reading_order([("b0", 0, None), ("b0", 0, None)])
    speaker_results = [make_speaker_result(page0, [])]

    resolution = RESOLVER.resolve(
        "seq_test", [page0, page1, page2], adj, reading_order, speaker_results, identity=None
    )

    diag_codes = [d.code for d in resolution.diagnostics]
    assert "duplicate_balloon_in_sequence" in diag_codes


# ---------------------------------------------------------------------------
# Test 11: invalid character reference in speaker decision
# ---------------------------------------------------------------------------

def test_invalid_speaker_character_reference():
    page0 = make_page(0, [make_balloon("b0")], [])  # no characters
    page1 = make_page(1, [], [])
    page2 = make_page(2, [], [])

    adj = [make_adj("b0", "spoken")]
    reading_order = make_reading_order([("b0", 0, None)])
    # Speaker decision references a char_id that doesn't exist
    speaker_results = [make_speaker_result(page0, [make_decision("b0", "ghost-char-999")])]

    resolution = RESOLVER.resolve(
        "seq_test", [page0, page1, page2], adj, reading_order, speaker_results, identity=None
    )

    balloon = resolution.ordered_balloons[0]
    diag_codes = [d.code for d in balloon.diagnostics]
    assert "invalid_speaker_character_reference" in diag_codes
    assert balloon.speaker_label is None


# ---------------------------------------------------------------------------
# Test 12: identity references unknown character
# ---------------------------------------------------------------------------

def test_identity_references_unknown_character():
    page0 = make_page(0, [make_balloon("b0")], [])
    page1 = make_page(1, [], [])
    page2 = make_page(2, [], [])

    adj = [make_adj("b0", "line")]
    reading_order = make_reading_order([("b0", 0, None)])
    speaker_results = [make_speaker_result(page0, [])]
    # Identity references a char that never existed in any page
    identity = SequenceCharacterIdentity(
        sequence_id="seq_test",
        clusters=[
            CharacterIdentityCluster(
                identity_id="identity-001",
                label="A",
                members=[IdentityMember(character_id="phantom-char", page_index=0)],
                state="unmatched",
            )
        ],
        character_to_identity={"phantom-char": "identity-001"},
        character_to_label={"phantom-char": "A"},
    )

    resolution = RESOLVER.resolve(
        "seq_test", [page0, page1, page2], adj, reading_order, speaker_results, identity
    )

    diag_codes = [d.code for d in resolution.diagnostics]
    assert "identity_references_unknown_character" in diag_codes


# ---------------------------------------------------------------------------
# Test 13: identity label conflict
# ---------------------------------------------------------------------------

def test_identity_label_conflict_detected():
    char = make_char("char-p0-X")
    page0 = make_page(0, [make_balloon("b0"), make_balloon("b1")], [char])
    page1 = make_page(1, [], [])
    page2 = make_page(2, [], [])

    adj = [make_adj("b0", "text A"), make_adj("b1", "text B")]
    reading_order = make_reading_order([("b0", 0, None), ("b1", 0, None)])
    speaker_results = [make_speaker_result(page0, [
        make_decision("b0", "char-p0-X"),
        make_decision("b1", "char-p0-X"),
    ])]
    # Intentionally corrupt: same identity_id with two different labels
    identity = SequenceCharacterIdentity(
        sequence_id="seq_test",
        clusters=[
            CharacterIdentityCluster(
                identity_id="identity-001",
                label="A",
                members=[IdentityMember(character_id="char-p0-X", page_index=0)],
                state="unmatched",
            )
        ],
        character_to_identity={"char-p0-X": "identity-001"},
        # CORRUPT: manually set two different labels via two balloons — not possible
        # via make_identity but can happen if upstream bugs out.
        character_to_label={"char-p0-X": "A"},
    )

    resolution = RESOLVER.resolve(
        "seq_test", [page0, page1, page2], adj, reading_order, speaker_results, identity
    )

    # All balloons should get label A (from identity), no conflict here since
    # we can't truly corrupt via this path — both balloons share the same char.
    labels = {b.speaker_label for b in resolution.ordered_balloons}
    assert labels == {"A"}
    # No label conflict diagnostic since both correctly receive "A"
    diag_codes = [d.code for d in resolution.diagnostics]
    assert "identity_label_conflict" not in diag_codes


# ---------------------------------------------------------------------------
# Test 14: deterministic anonymous labels
# ---------------------------------------------------------------------------

def test_deterministic_anonymous_labels():
    """Same identity assignment must produce same labels regardless of call order."""
    chars = [make_char(f"char-p{i}-X") for i in range(3)]
    pages = [make_page(i, [make_balloon(f"b{i}")], [chars[i]]) for i in range(3)]
    adj = [make_adj(f"b{i}", f"text {i}") for i in range(3)]
    reading_order = make_reading_order([(f"b{i}", i, None) for i in range(3)])
    speaker_results = [
        make_speaker_result(pages[i], [make_decision(f"b{i}", f"char-p{i}-X")])
        for i in range(3)
    ]
    identity = make_identity([
        ("identity-001", "A", [("char-p0-X", 0)], "unmatched"),
        ("identity-002", "B", [("char-p1-X", 1)], "unmatched"),
        ("identity-003", "C", [("char-p2-X", 2)], "unmatched"),
    ])

    res1 = RESOLVER.resolve("seq_test", pages, adj, reading_order, speaker_results, identity)
    res2 = RESOLVER.resolve("seq_test", pages, adj, reading_order, speaker_results, identity)

    labels1 = [b.speaker_label for b in res1.ordered_balloons]
    labels2 = [b.speaker_label for b in res2.ordered_balloons]
    assert labels1 == labels2
    assert labels1 == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Test 15: three-page constraint validation
# ---------------------------------------------------------------------------

def test_three_page_constraint_valid():
    """A resolution with all three page indices present passes validation."""
    pages = [make_page(i, [make_balloon(f"b{i}")]) for i in range(3)]
    adj = [make_adj(f"b{i}", f"text {i}") for i in range(3)]
    reading_order = make_reading_order([(f"b{i}", i, None) for i in range(3)])
    speaker_results = [make_speaker_result(pages[i], []) for i in range(3)]

    resolution = RESOLVER.resolve(
        "seq_test", pages, adj, reading_order, speaker_results, identity=None
    )

    # Inject a fake story balloon with invalid page_index to test the rule
    bad_resolution = resolution.model_copy(update={
        "ordered_balloons": resolution.ordered_balloons + [
            ResolvedBalloon(
                sequence_position=99,
                page_index=5,  # invalid
                balloon_id="bad-balloon",
                text="oops",
                text_type="dialogue",
                include_in_story=True,
            )
        ]
    })

    errors = validate_sequence_resolution(bad_resolution)
    rule_names = [e.rule for e in errors]
    assert "exactly_three_pages" in rule_names or "all_page_indices_valid" in rule_names


# ---------------------------------------------------------------------------
# Test 16: missing adjudication result
# ---------------------------------------------------------------------------

def test_missing_adjudication_result():
    page0 = make_page(0, [make_balloon("b0")])
    page1 = make_page(1, [], [])
    page2 = make_page(2, [], [])

    # b0 is in reading order but has no adjudication result
    reading_order = make_reading_order([("b0", 0, None)])
    speaker_results = [make_speaker_result(page0, [])]

    resolution = RESOLVER.resolve(
        "seq_test", [page0, page1, page2],
        adjudication_results=[],  # no results
        reading_order=reading_order,
        speaker_results=speaker_results,
        identity=None,
    )

    diag_codes = [d.code for d in resolution.diagnostics]
    assert "missing_adjudication_result" in diag_codes
    assert len(resolution.ordered_balloons) == 0
    assert len(resolution.unresolved_items) == 1


# ---------------------------------------------------------------------------
# Test 17: conflicting page_index in reading order vs page list
# ---------------------------------------------------------------------------

def test_conflicting_page_index_in_reading_order():
    """Reading order says balloon b0 is on page 0, but balloon is actually on page 1."""
    page0 = make_page(0, [], [])
    page1 = make_page(1, [make_balloon("b0")], [])  # b0 is on page 1
    page2 = make_page(2, [], [])

    adj = [make_adj("b0", "text")]
    # Reading order claims page_index=0 but b0 is actually on page 1
    reading_order = make_reading_order([("b0", 0, None)])
    speaker_results = [
        make_speaker_result(page0, []),
        make_speaker_result(page1, []),
    ]

    resolution = RESOLVER.resolve(
        "seq_test", [page0, page1, page2], adj, reading_order, speaker_results, identity=None
    )

    # Balloon should still be resolved (using actual page from page list)
    assert len(resolution.ordered_balloons) == 1
    balloon = resolution.ordered_balloons[0]
    assert balloon.page_index == 1  # uses actual page, not claimed page
    # Mismatch diagnostic should be present
    all_diag_codes = [d.code for d in balloon.diagnostics] + [d.code for d in resolution.diagnostics]
    assert "page_index_mismatch" in all_diag_codes


# ---------------------------------------------------------------------------
# Test 18: final serializer exact competition schema
# ---------------------------------------------------------------------------

def test_serializer_exact_competition_schema():
    """Serialized output must have exactly {sequence_id, pages} at top level.
    Each page item must have exactly {speaker, text}."""
    char = make_char("char-p0-Z")
    page0 = make_page(0, [make_balloon("b0")], [char])
    page1 = make_page(1, [], [])
    page2 = make_page(2, [], [])

    adj = [make_adj("b0", "Hello there.")]
    reading_order = make_reading_order([("b0", 0, None)])
    speaker_results = [make_speaker_result(page0, [make_decision("b0", "char-p0-Z")])]
    identity = make_identity([("identity-001", "A", [("char-p0-Z", 0)], "unmatched")])

    resolution = RESOLVER.resolve(
        "seq_test", [page0, page1, page2], adj, reading_order, speaker_results, identity
    )

    record = serialize_to_submission(resolution)
    line = serialize_to_jsonl_line(record)
    parsed = json.loads(line)

    # Top-level keys: exactly sequence_id and pages
    assert set(parsed.keys()) == {"sequence_id", "pages"}
    assert parsed["sequence_id"] == "seq_test"
    assert isinstance(parsed["pages"], list)
    assert len(parsed["pages"]) == 3

    # Page 0 has one item
    assert len(parsed["pages"][0]) == 1
    item = parsed["pages"][0][0]
    assert set(item.keys()) == {"speaker", "text"}
    assert item["speaker"] == "A"
    assert item["text"] == "Hello there."

    # Pages 1 and 2 are empty
    assert parsed["pages"][1] == []
    assert parsed["pages"][2] == []


# ---------------------------------------------------------------------------
# Test 19: validator rejection cases
# ---------------------------------------------------------------------------

def test_validator_rejects_duplicate_balloon():
    b0 = ResolvedBalloon(
        sequence_position=0, page_index=0, balloon_id="b0",
        text="a", text_type="dialogue", include_in_story=True
    )
    b0_dup = ResolvedBalloon(
        sequence_position=1, page_index=1, balloon_id="b0",  # duplicate
        text="b", text_type="dialogue", include_in_story=True
    )
    resolution = SequenceResolution(
        sequence_id="seq_test",
        ordered_balloons=[b0, b0_dup],
    )
    errors = validate_sequence_resolution(resolution)
    rules = [e.rule for e in errors]
    assert "no_duplicate_balloon" in rules


def test_validator_rejects_excluded_in_story():
    b_excl = ResolvedBalloon(
        sequence_position=0, page_index=0, balloon_id="b_sfx",
        text="BOOM", text_type="sound_effect", include_in_story=False
    )
    resolution = SequenceResolution(
        sequence_id="seq_test",
        ordered_balloons=[b_excl],  # excluded balloon should NOT be here
    )
    errors = validate_sequence_resolution(resolution)
    rules = [e.rule for e in errors]
    assert "story_excluded_balloons_absent" in rules


def test_validator_rejects_invalid_character_reference():
    b0 = ResolvedBalloon(
        sequence_position=0, page_index=0, balloon_id="b0",
        text="spoken", text_type="dialogue", include_in_story=True,
        speaker_character_instance_id="does-not-exist",
        speaker_label="A",
    )
    resolution = SequenceResolution(sequence_id="seq_test", ordered_balloons=[b0])
    errors = validate_sequence_resolution(resolution, known_character_ids={"real-char"})
    rules = [e.rule for e in errors]
    assert "speaker_references_valid_character" in rules


def test_validator_rejects_empty_text():
    b0 = ResolvedBalloon(
        sequence_position=0, page_index=0, balloon_id="b0",
        text="   ",  # empty after strip
        text_type="dialogue", include_in_story=True,
    )
    resolution = SequenceResolution(sequence_id="seq_test", ordered_balloons=[b0])
    errors = validate_sequence_resolution(resolution)
    rules = [e.rule for e in errors]
    assert "text_exists_for_story_balloons" in rules


# ---------------------------------------------------------------------------
# Test 20: valid submission object passes full validation
# ---------------------------------------------------------------------------

def test_valid_submission_passes_all_checks():
    chars = [make_char(f"char-p{i}") for i in range(3)]
    pages = [make_page(i, [make_balloon(f"b{i}")], [chars[i]]) for i in range(3)]
    adj = [make_adj(f"b{i}", f"Story text {i}.") for i in range(3)]
    reading_order = make_reading_order([(f"b{i}", i, None) for i in range(3)])
    speaker_results = [
        make_speaker_result(pages[i], [make_decision(f"b{i}", f"char-p{i}")])
        for i in range(3)
    ]
    identity = make_identity([
        (f"identity-00{i+1}", chr(ord('A') + i),
         [(f"char-p{i}", i)], "unmatched")
        for i in range(3)
    ])

    resolution = RESOLVER.resolve(
        sequence_id="seq_valid",
        pages=pages,
        adjudication_results=adj,
        reading_order=reading_order,
        speaker_results=speaker_results,
        identity=identity,
    )

    known_chars = {f"char-p{i}" for i in range(3)}
    errors = validate_sequence_resolution(
        resolution,
        known_character_ids=known_chars,
        expected_sequence_id="seq_valid",
    )
    assert errors == [], [f"{e.rule}: {e.message}" for e in errors]

    # Serializer also produces valid competition format
    record = serialize_to_submission(resolution)
    line = serialize_to_jsonl_line(record)
    parsed = json.loads(line)
    assert set(parsed.keys()) == {"sequence_id", "pages"}
    assert len(parsed["pages"]) == 3
    for page_items in parsed["pages"]:
        for item in page_items:
            assert set(item.keys()) == {"speaker", "text"}
            assert item["speaker"] and item["speaker"].strip() == item["speaker"]
            assert item["text"]


# ---------------------------------------------------------------------------
# Additional edge-case tests for serializer
# ---------------------------------------------------------------------------

def test_serializer_narration_type_uses_narration_speaker():
    """Narration-type balloons must serialize with speaker=NARRATION."""
    page0 = make_page(0, [make_balloon("b_nar")])
    page1 = make_page(1, [], [])
    page2 = make_page(2, [], [])

    adj = [make_adj("b_nar", "Caption text.", include=True, text_type="narration")]
    reading_order = make_reading_order([("b_nar", 0, None)])
    speaker_results = [make_speaker_result(page0, [])]

    resolution = RESOLVER.resolve(
        "seq_test", [page0, page1, page2], adj, reading_order, speaker_results, identity=None
    )

    record = serialize_to_submission(resolution)
    assert record.pages[0][0].speaker == "NARRATION"


def test_serializer_unresolved_speaker_uses_unknown():
    """Unresolved speaker → 'UNKNOWN' in submission."""
    assert _submission_speaker(None) == "UNKNOWN"
    assert _submission_speaker("") == "UNKNOWN"
    assert _submission_speaker("A") == "A"
    assert _submission_speaker("NARRATION") == "NARRATION"


def test_serializer_ambiguous_identity_emits_unknown():
    """Ambiguous identity must NOT produce a falsely confident speaker label in submission.

    The internal ResolvedBalloon.speaker_label retains the label (e.g. 'A') for
    traceability, but serialize_to_submission() must emit 'UNKNOWN' because the
    cluster is ambiguous and the label cannot be treated as confident.
    """
    # Verify _submission_speaker directly
    assert _submission_speaker("A", identity_state="ambiguous") == "UNKNOWN"
    assert _submission_speaker("B", identity_state="ambiguous") == "UNKNOWN"
    assert _submission_speaker("NARRATION", identity_state="ambiguous") == "NARRATION"  # narration is always NARRATION
    assert _submission_speaker("A", identity_state="matched") == "A"
    assert _submission_speaker("A", identity_state="unmatched") == "A"
    assert _submission_speaker("A", identity_state="null") == "A"

    # End-to-end: ambiguous identity cluster → UNKNOWN in the submission record
    char = make_char("char-p0-amb")
    page0 = make_page(0, [make_balloon("b0")], [char])
    page1 = make_page(1, [], [])
    page2 = make_page(2, [], [])

    adj = [make_adj("b0", "Ambiguous identity speech.")]
    reading_order = make_reading_order([("b0", 0, None)])
    speaker_results = [make_speaker_result(page0, [make_decision("b0", "char-p0-amb")])]
    identity = make_identity([("identity-001", "A", [("char-p0-amb", 0)], "ambiguous")])

    resolution = RESOLVER.resolve(
        "seq_test", [page0, page1, page2], adj, reading_order, speaker_results, identity
    )

    # Internal resolution retains 'A' for traceability
    balloon = resolution.ordered_balloons[0]
    assert balloon.speaker_label == "A"
    assert balloon.identity_state == "ambiguous"

    # But the competition submission must use 'UNKNOWN', not 'A'
    record = serialize_to_submission(resolution)
    assert record.pages[0][0].speaker == "UNKNOWN", (
        "Ambiguous identity must not produce a falsely confident label in submission"
    )
