"""CPU-only tests for sequence-level reading order.

Coverage:
 1. three-page hard ordering
 2. page precedence (page 0 < page 1 < page 2)
 3. multiple panels
 4. multiple balloons inside one panel
 5. panelless fallback
 6. same-row balloons (right-to-left default)
 7. same-column balloons (top-to-bottom)
 8. overlapping balloons
 9. contained balloons
10. ambiguous pair (low-confidence decision)
11. cyclic pairwise graph
12. deterministic tie-breaking
13. excluded non-story balloon
14. missing panel data
15. speaker-independent ordering
"""

from __future__ import annotations

import pytest

from app.reading_order import (
    DeterministicGeometryRanker,
    PairwiseFeatures,
    _PrecedenceGraph,
    _order_balloons_within_group,
    _rank_panels,
    order_page,
    order_sequence,
)
from app.schemas.adjudication import BalloonAdjudicationResult
from app.schemas.page import Balloon, BoundingBox, PageRepresentation, Panel
from app.schemas.reading_order import ReadingOrderDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bb(x1, y1, x2, y2) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def make_balloon(bid: str, bbox: BoundingBox, panel_id: str | None = None) -> Balloon:
    return Balloon(id=bid, bbox=bbox, panel_id=panel_id)


def make_panel(pid: str, bbox: BoundingBox, source="layout_model") -> Panel:
    return Panel(id=pid, bbox=bbox, source=source)


def make_page(
    page_index: int,
    balloons: list[Balloon],
    panels: list[Panel] | None = None,
) -> PageRepresentation:
    return PageRepresentation(
        page_index=page_index,
        image_path="unused.png",
        balloons=balloons,
        panels=panels or [],
    )


def adjudication_map(balloon_ids: list[str], include: bool = True) -> dict[str, BalloonAdjudicationResult]:
    return {
        bid: BalloonAdjudicationResult(
            balloon_id=bid,
            final_text="text" if include else "",
            include_in_story=include,
        )
        for bid in balloon_ids
    }


# ---------------------------------------------------------------------------
# Test 1: three-page hard ordering — all page-0 before page-1 before page-2
# ---------------------------------------------------------------------------

def test_three_page_hard_ordering():
    """Page ordering is a hard constraint: page 0 always before page 1 before page 2."""
    pages = [
        make_page(0, [make_balloon("b0a", bb(10, 10, 50, 40)), make_balloon("b0b", bb(60, 10, 100, 40))]),
        make_page(1, [make_balloon("b1a", bb(10, 10, 50, 40))]),
        make_page(2, [make_balloon("b2a", bb(10, 10, 50, 40)), make_balloon("b2b", bb(10, 50, 50, 80))]),
    ]
    result = order_sequence(pages)
    ids = result.ordered_balloon_ids
    # page-0 balloons must precede page-1
    assert ids.index("b0a") < ids.index("b1a")
    assert ids.index("b0b") < ids.index("b1a")
    # page-1 must precede page-2
    assert ids.index("b1a") < ids.index("b2a")
    assert ids.index("b1a") < ids.index("b2b")
    # All IDs present
    assert set(ids) == {"b0a", "b0b", "b1a", "b2a", "b2b"}


# ---------------------------------------------------------------------------
# Test 2: page precedence — page 0 balloons always precede page 1
# ---------------------------------------------------------------------------

def test_page_0_balloons_always_precede_page_1():
    """Verify that every page-0 balloon appears before every page-1 balloon."""
    p0 = make_page(0, [make_balloon(f"p0-{i}", bb(10, 10 + i * 30, 50, 40 + i * 30)) for i in range(5)])
    p1 = make_page(1, [make_balloon(f"p1-{i}", bb(10, 10 + i * 30, 50, 40 + i * 30)) for i in range(5)])
    result = order_sequence([p0, p1])
    ids = result.ordered_balloon_ids
    p0_positions = [ids.index(f"p0-{i}") for i in range(5)]
    p1_positions = [ids.index(f"p1-{i}") for i in range(5)]
    assert max(p0_positions) < min(p1_positions)


def test_page_1_balloons_always_precede_page_2():
    """Verify that every page-1 balloon appears before every page-2 balloon."""
    p1 = make_page(1, [make_balloon(f"p1-{i}", bb(10, 10 + i * 30, 50, 40 + i * 30)) for i in range(3)])
    p2 = make_page(2, [make_balloon(f"p2-{i}", bb(10, 10 + i * 30, 50, 40 + i * 30)) for i in range(3)])
    result = order_sequence([p1, p2])
    ids = result.ordered_balloon_ids
    p1_positions = [ids.index(f"p1-{i}") for i in range(3)]
    p2_positions = [ids.index(f"p2-{i}") for i in range(3)]
    assert max(p1_positions) < min(p2_positions)


# ---------------------------------------------------------------------------
# Test 3: multiple panels — panel order respected
# ---------------------------------------------------------------------------

def test_multiple_panels_panel_order_respected():
    """Balloons in the top panel come before balloons in the bottom panel."""
    top_panel = make_panel("top", bb(0, 0, 200, 100))
    bot_panel = make_panel("bot", bb(0, 110, 200, 210))
    b_top = make_balloon("b_top", bb(10, 10, 80, 80), panel_id="top")
    b_bot = make_balloon("b_bot", bb(10, 130, 80, 190), panel_id="bot")
    page = make_page(0, [b_top, b_bot], panels=[top_panel, bot_panel])
    result = order_page(page, DeterministicGeometryRanker(), page_width=200, page_height=210)
    ids = result.balloon_ids_in_order
    assert ids.index("b_top") < ids.index("b_bot")


def test_two_panels_side_by_side_right_to_left():
    """Two side-by-side panels: right panel comes before left (manga RTL default)."""
    right_panel = make_panel("right", bb(110, 0, 200, 200))
    left_panel = make_panel("left", bb(0, 0, 100, 200))
    b_right = make_balloon("b_right", bb(120, 10, 190, 80), panel_id="right")
    b_left = make_balloon("b_left", bb(10, 10, 80, 80), panel_id="left")
    page = make_page(0, [b_right, b_left], panels=[right_panel, left_panel])
    result = order_page(page, DeterministicGeometryRanker(), page_width=200, page_height=200)
    ids = result.balloon_ids_in_order
    assert ids.index("b_right") < ids.index("b_left")


# ---------------------------------------------------------------------------
# Test 4: multiple balloons inside one panel
# ---------------------------------------------------------------------------

def test_multiple_balloons_in_one_panel_top_before_bottom():
    """Vertically stacked balloons inside one panel: top before bottom."""
    panel = make_panel("p1", bb(0, 0, 200, 300))
    top = make_balloon("top", bb(10, 10, 100, 80), panel_id="p1")
    mid = make_balloon("mid", bb(10, 100, 100, 170), panel_id="p1")
    bot = make_balloon("bot", bb(10, 190, 100, 260), panel_id="p1")
    page = make_page(0, [top, mid, bot], panels=[panel])
    result = order_page(page, DeterministicGeometryRanker(), page_width=200, page_height=300)
    ids = result.balloon_ids_in_order
    assert ids.index("top") < ids.index("mid") < ids.index("bot")


# ---------------------------------------------------------------------------
# Test 5: panelless fallback
# ---------------------------------------------------------------------------

def test_panelless_fallback_no_layout_panels():
    """When there are no layout_model panels, fallback uses page-level geometry."""
    fallback_panel = make_panel("fallback", bb(0, 0, 200, 400), source="page_fallback")
    top = make_balloon("top", bb(10, 10, 100, 80))
    bot = make_balloon("bot", bb(10, 200, 100, 280))
    page = make_page(0, [top, bot], panels=[fallback_panel])
    result = order_page(page, DeterministicGeometryRanker(), page_width=200, page_height=400)
    ids = result.balloon_ids_in_order
    assert ids.index("top") < ids.index("bot")
    # Diagnostic for no layout panels should be present
    assert any(d.code == "no_layout_panels" for d in result.diagnostics)


def test_panelless_fallback_completely_no_panels():
    """With zero panels at all, fallback also works."""
    top = make_balloon("top", bb(10, 10, 100, 80))
    bot = make_balloon("bot", bb(10, 200, 100, 280))
    page = make_page(0, [top, bot])
    result = order_page(page, DeterministicGeometryRanker(), page_width=200, page_height=400)
    ids = result.balloon_ids_in_order
    assert ids.index("top") < ids.index("bot")


# ---------------------------------------------------------------------------
# Test 6: same-row balloons (right-to-left default)
# ---------------------------------------------------------------------------

def test_same_row_balloons_rtl_default():
    """Same-row balloons: rightmost comes first (manga RTL default)."""
    right = make_balloon("right", bb(120, 10, 180, 60))
    left = make_balloon("left", bb(10, 15, 70, 55))
    features = PairwiseFeatures(right.bbox, left.bbox, page_width=200, page_height=100)
    ranker = DeterministicGeometryRanker(rtl=None)  # unknown → weak RTL default
    context: dict = {}
    dec = ranker.compare(right, left, features, context)
    # right has higher x, should come first in RTL
    assert dec.first_balloon_id == "right"
    assert dec.confidence is not None and dec.confidence < 0.8  # low confidence for unknown


def test_same_row_balloons_explicit_rtl():
    """Explicit RTL: rightmost balloon first, high confidence."""
    right = make_balloon("right", bb(120, 10, 180, 60))
    left = make_balloon("left", bb(10, 15, 70, 55))
    features = PairwiseFeatures(right.bbox, left.bbox, page_width=200, page_height=100)
    ranker = DeterministicGeometryRanker(rtl=True)
    dec = ranker.compare(right, left, features, {})
    assert dec.first_balloon_id == "right"
    assert dec.confidence is not None and dec.confidence >= 0.75


def test_same_row_balloons_explicit_ltr():
    """Explicit LTR: leftmost balloon first."""
    right = make_balloon("right", bb(120, 10, 180, 60))
    left = make_balloon("left", bb(10, 15, 70, 55))
    features = PairwiseFeatures(right.bbox, left.bbox, page_width=200, page_height=100)
    ranker = DeterministicGeometryRanker(rtl=False)
    dec = ranker.compare(right, left, features, {})
    assert dec.first_balloon_id == "left"


# ---------------------------------------------------------------------------
# Test 7: same-column balloons (top-to-bottom)
# ---------------------------------------------------------------------------

def test_same_column_balloons_top_before_bottom():
    """Vertically arranged balloons: top before bottom."""
    top = make_balloon("top", bb(10, 10, 80, 60))
    bot = make_balloon("bot", bb(15, 80, 75, 130))
    features = PairwiseFeatures(top.bbox, bot.bbox, page_width=200, page_height=200)
    ranker = DeterministicGeometryRanker()
    dec = ranker.compare(top, bot, features, {})
    assert dec.first_balloon_id == "top"


# ---------------------------------------------------------------------------
# Test 8: overlapping balloons
# ---------------------------------------------------------------------------

def test_overlapping_balloons_ordered_by_vertical_center():
    """Partially overlapping balloons are ordered by their vertical center."""
    higher = make_balloon("higher", bb(10, 10, 100, 70))
    lower = make_balloon("lower", bb(20, 50, 110, 110))
    features = PairwiseFeatures(higher.bbox, lower.bbox, page_width=200, page_height=200)
    ranker = DeterministicGeometryRanker()
    dec = ranker.compare(higher, lower, features, {})
    assert dec.first_balloon_id == "higher"


# ---------------------------------------------------------------------------
# Test 9: contained balloons
# ---------------------------------------------------------------------------

def test_contained_balloon_outer_precedes_inner():
    """If balloon A contains balloon B, A comes first."""
    outer = make_balloon("outer", bb(0, 0, 200, 200))
    inner = make_balloon("inner", bb(50, 50, 150, 150))
    features = PairwiseFeatures(outer.bbox, inner.bbox, page_width=200, page_height=200)
    ranker = DeterministicGeometryRanker()
    dec = ranker.compare(outer, inner, features, {})
    assert dec.first_balloon_id == "outer"
    assert dec.evidence["rule"] == "containment"


def test_contained_balloon_inner_is_b():
    """If balloon B contains balloon A, B still comes first."""
    outer = make_balloon("outer", bb(0, 0, 200, 200))
    inner = make_balloon("inner", bb(50, 50, 150, 150))
    features = PairwiseFeatures(inner.bbox, outer.bbox, page_width=200, page_height=200)
    ranker = DeterministicGeometryRanker()
    dec = ranker.compare(inner, outer, features, {})
    assert dec.first_balloon_id == "outer"


# ---------------------------------------------------------------------------
# Test 10: ambiguous pair
# ---------------------------------------------------------------------------

def test_ambiguous_pair_reports_low_confidence():
    """Same-row with unknown RTL direction → confidence < 0.7."""
    a = make_balloon("a", bb(10, 10, 80, 60))
    b = make_balloon("b", bb(90, 15, 160, 55))
    features = PairwiseFeatures(a.bbox, b.bbox, page_width=200, page_height=100)
    ranker = DeterministicGeometryRanker(rtl=None)
    dec = ranker.compare(a, b, features, {})
    assert dec.confidence is not None
    assert dec.confidence < 0.7


# ---------------------------------------------------------------------------
# Test 11: cyclic pairwise graph
# ---------------------------------------------------------------------------

def test_cyclic_graph_resolved_deterministically():
    """A cycle in the precedence graph is detected and resolved, producing a valid order."""
    graph = _PrecedenceGraph()
    fake_decision = ReadingOrderDecision(
        first_balloon_id="a",
        second_balloon_id="b",
        first_precedes_second=True,
        confidence=0.9,
        method="balloon_geometry",
    )
    graph.add_edge("a", "b", 0.9, fake_decision)
    graph.add_edge("b", "c", 0.8, fake_decision)
    graph.add_edge("c", "a", 0.5, fake_decision)  # back-edge forming cycle

    ordered, diag = graph.topological_sort()
    # All three nodes must be present
    assert set(ordered) == {"a", "b", "c"}
    # Cycle diagnostic must be reported
    assert any(d.code == "cycle_detected" for d in diag)


def test_cyclic_graph_in_order_sequence():
    """A set of balloons that form a cycle should still produce a valid complete order."""
    # Inject a custom ranker that creates a cycle
    class CyclicRanker:
        """Forces a→b, b→c, c→a cycle."""
        _decisions = {
            ("ba", "bb"): True,
            ("bb", "ba"): False,
            ("bb", "bc"): True,
            ("bc", "bb"): False,
            ("bc", "ba"): True,
            ("ba", "bc"): False,
        }

        def compare(self, a, b, features, context):
            key = (a.id, b.id)
            a_first = self._decisions.get(key, True)
            return ReadingOrderDecision(
                first_balloon_id=a.id if a_first else b.id,
                second_balloon_id=b.id if a_first else a.id,
                first_precedes_second=True,
                confidence=0.6 if key in self._decisions else 0.5,
                method="balloon_geometry",
            )

    balloons = [
        make_balloon("ba", bb(10, 10, 50, 50)),
        make_balloon("bb", bb(60, 10, 100, 50)),
        make_balloon("bc", bb(10, 60, 50, 100)),
    ]
    page = make_page(0, balloons)
    result = order_page(page, CyclicRanker(), page_width=120, page_height=120)
    assert set(result.balloon_ids_in_order) == {"ba", "bb", "bc"}
    assert len(result.balloon_ids_in_order) == 3


# ---------------------------------------------------------------------------
# Test 12: deterministic tie-breaking
# ---------------------------------------------------------------------------

def test_deterministic_tie_breaking_same_center():
    """Two balloons with identical centers produce a stable, id-based order."""
    a = make_balloon("aaa", bb(10, 10, 50, 50))
    b = make_balloon("bbb", bb(10, 10, 50, 50))
    features = PairwiseFeatures(a.bbox, b.bbox, page_width=200, page_height=200)
    ranker = DeterministicGeometryRanker()
    dec1 = ranker.compare(a, b, features, {})
    dec2 = ranker.compare(a, b, features, {})
    assert dec1.first_balloon_id == dec2.first_balloon_id  # deterministic


def test_deterministic_order_independent_of_balloon_list_order():
    """Reversing the input list produces the same final order."""
    top = make_balloon("top", bb(10, 10, 80, 60))
    mid = make_balloon("mid", bb(10, 80, 80, 130))
    bot = make_balloon("bot", bb(10, 160, 80, 210))
    page_a = make_page(0, [top, mid, bot])
    page_b = make_page(0, [bot, mid, top])
    res_a = order_page(page_a, DeterministicGeometryRanker(), page_width=200, page_height=250)
    res_b = order_page(page_b, DeterministicGeometryRanker(), page_width=200, page_height=250)
    assert res_a.balloon_ids_in_order == res_b.balloon_ids_in_order


# ---------------------------------------------------------------------------
# Test 13: excluded non-story balloon
# ---------------------------------------------------------------------------

def test_non_story_balloon_excluded_from_order():
    """Balloons with include_in_story=False are excluded from reading order."""
    story = make_balloon("story", bb(10, 10, 80, 60))
    excluded = make_balloon("sfx", bb(10, 80, 80, 130))
    page = make_page(0, [story, excluded])
    adj = {
        "story": BalloonAdjudicationResult(
            balloon_id="story", final_text="Hello", include_in_story=True
        ),
        "sfx": BalloonAdjudicationResult(
            balloon_id="sfx", final_text="", include_in_story=False
        ),
    }
    result = order_page(page, DeterministicGeometryRanker(), adjudication_results=adj)
    assert "story" in result.balloon_ids_in_order
    assert "sfx" not in result.balloon_ids_in_order


def test_excluded_balloon_not_in_sequence_order():
    """End-to-end: excluded balloon absent from SequenceReadingOrder."""
    story = make_balloon("story", bb(10, 10, 80, 60))
    sfx = make_balloon("sfx", bb(10, 80, 80, 130))
    page = make_page(0, [story, sfx])
    adj = {
        "story": BalloonAdjudicationResult(
            balloon_id="story", final_text="Hello", include_in_story=True
        ),
        "sfx": BalloonAdjudicationResult(
            balloon_id="sfx", final_text="", include_in_story=False
        ),
    }
    result = order_sequence([page], adjudication_results=adj)
    assert "story" in result.ordered_balloon_ids
    assert "sfx" not in result.ordered_balloon_ids


# ---------------------------------------------------------------------------
# Test 14: missing panel data
# ---------------------------------------------------------------------------

def test_missing_panel_data_gracefully_falls_back():
    """Balloons with panel_id that doesn't exist in page.panels get geometric assignment."""
    panel = make_panel("real-panel", bb(0, 0, 200, 200))
    b_with_panel = make_balloon("b_in_panel", bb(10, 10, 90, 90), panel_id="real-panel")
    b_orphan = make_balloon("b_orphan", bb(10, 110, 90, 190), panel_id="ghost-panel")  # non-existent
    page = make_page(0, [b_with_panel, b_orphan], panels=[panel])
    result = order_page(page, DeterministicGeometryRanker(), page_width=200, page_height=200)
    # Both should appear
    assert "b_in_panel" in result.balloon_ids_in_order
    assert "b_orphan" in result.balloon_ids_in_order


def test_balloon_with_no_panel_and_no_geometric_overlap_still_ordered():
    """A balloon with no panel_id and no overlap is added to page-level fallback."""
    panel = make_panel("p1", bb(0, 0, 100, 100))
    b_in = make_balloon("b_in", bb(10, 10, 80, 80), panel_id="p1")
    b_out = make_balloon("b_out", bb(150, 150, 190, 190))  # no overlap with any panel
    page = make_page(0, [b_in, b_out], panels=[panel])
    result = order_page(page, DeterministicGeometryRanker(), page_width=200, page_height=200)
    assert "b_in" in result.balloon_ids_in_order
    assert "b_out" in result.balloon_ids_in_order


# ---------------------------------------------------------------------------
# Test 15: speaker-independent ordering
# ---------------------------------------------------------------------------

def test_ordering_does_not_require_speaker_information():
    """Reading order must not require candidate_character_ids or speaker decisions."""
    # Balloons with empty candidate_character_ids (unknown speakers)
    b1 = Balloon(id="b1", bbox=bb(10, 10, 80, 60), candidate_character_ids=[])
    b2 = Balloon(id="b2", bbox=bb(10, 80, 80, 130), candidate_character_ids=[])
    b3 = Balloon(id="b3", bbox=bb(10, 150, 80, 200), candidate_character_ids=[])
    page = make_page(0, [b1, b2, b3])
    result = order_page(page, DeterministicGeometryRanker(), page_width=200, page_height=250)
    assert set(result.balloon_ids_in_order) == {"b1", "b2", "b3"}
    # Top-to-bottom order
    ids = result.balloon_ids_in_order
    assert ids.index("b1") < ids.index("b2") < ids.index("b3")


# ---------------------------------------------------------------------------
# Panel ordering unit tests
# ---------------------------------------------------------------------------

def test_rank_panels_top_row_before_bottom_row():
    """Panels in the top row should come before panels in the bottom row."""
    p_top_left = make_panel("tl", bb(0, 0, 100, 100))
    p_top_right = make_panel("tr", bb(110, 0, 200, 100))
    p_bot = make_panel("b", bb(0, 120, 200, 220))
    ordered = _rank_panels([p_top_left, p_top_right, p_bot])
    ordered_ids = [p.id for p in ordered]
    assert ordered_ids.index("tl") < ordered_ids.index("b")
    assert ordered_ids.index("tr") < ordered_ids.index("b")


def test_rank_panels_same_row_right_before_left():
    """Panels in the same row: right panel before left panel (manga RTL)."""
    left = make_panel("left", bb(0, 0, 90, 100))
    right = make_panel("right", bb(110, 0, 200, 100))
    ordered = _rank_panels([left, right])
    ids = [p.id for p in ordered]
    assert ids[0] == "right"
    assert ids[1] == "left"


def test_rank_panels_single_panel_returned():
    single = make_panel("solo", bb(0, 0, 200, 200))
    assert _rank_panels([single]) == [single]


def test_rank_panels_empty():
    assert _rank_panels([]) == []


# ---------------------------------------------------------------------------
# SequenceReadingOrder completeness tests
# ---------------------------------------------------------------------------

def test_sequence_reading_order_items_cover_all_story_balloons():
    """All story balloon_ids appear exactly once in ordered_items."""
    pages = [
        make_page(0, [make_balloon(f"p0b{i}", bb(10, 10 + i * 50, 80, 50 + i * 50)) for i in range(3)]),
        make_page(1, [make_balloon(f"p1b{i}", bb(10, 10 + i * 50, 80, 50 + i * 50)) for i in range(2)]),
    ]
    result = order_sequence(pages)
    ids = result.ordered_balloon_ids
    all_balloon_ids = {b.id for p in pages for b in p.balloons}
    assert set(ids) == all_balloon_ids
    assert len(ids) == len(all_balloon_ids)


def test_sequence_ordered_items_sequence_positions_are_contiguous():
    """sequence_position values should be 0, 1, 2, ... without gaps."""
    pages = [
        make_page(0, [make_balloon(f"a{i}", bb(10, 10 + i * 40, 80, 40 + i * 40)) for i in range(2)]),
        make_page(1, [make_balloon(f"b{i}", bb(10, 10 + i * 40, 80, 40 + i * 40)) for i in range(3)]),
    ]
    result = order_sequence(pages)
    positions = [item.sequence_position for item in result.ordered_items]
    assert positions == list(range(len(positions)))


def test_sequence_page_order_results_match_page_count():
    """page_orders should have one entry per input page."""
    pages = [make_page(i, [make_balloon(f"b{i}", bb(10, 10, 50, 50))]) for i in range(3)]
    result = order_sequence(pages)
    assert len(result.page_orders) == 3
    assert result.page_orders[0].page_index == 0
    assert result.page_orders[1].page_index == 1
    assert result.page_orders[2].page_index == 2


def test_empty_page_produces_empty_balloon_order():
    """An empty page contributes no balloons but doesn't break the sequence."""
    p0 = make_page(0, [make_balloon("b0", bb(10, 10, 80, 60))])
    p1 = make_page(1, [])  # empty
    p2 = make_page(2, [make_balloon("b2", bb(10, 10, 80, 60))])
    result = order_sequence([p0, p1, p2])
    ids = result.ordered_balloon_ids
    assert "b0" in ids
    assert "b2" in ids
    assert ids.index("b0") < ids.index("b2")
    assert len(ids) == 2


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------

def test_pairwise_features_same_row_detection():
    """Balloons with high vertical overlap are classified as same-row."""
    a = bb(10, 10, 80, 60)
    b = bb(90, 15, 160, 55)
    f = PairwiseFeatures(a, b, page_width=200, page_height=100)
    assert f.same_row is True
    assert f.same_column is False


def test_pairwise_features_same_column_detection():
    """Balloons with high horizontal overlap are classified as same-column."""
    a = bb(10, 10, 80, 60)
    b = bb(15, 80, 75, 130)
    f = PairwiseFeatures(a, b, page_width=200, page_height=200)
    assert f.same_column is True
    assert f.same_row is False


def test_pairwise_features_containment():
    outer = bb(0, 0, 200, 200)
    inner = bb(50, 50, 150, 150)
    f = PairwiseFeatures(outer, inner, page_width=200, page_height=200)
    assert f.a_contains_b is True
    assert f.b_contains_a is False
