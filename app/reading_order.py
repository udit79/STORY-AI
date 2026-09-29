"""Sequence-level reading order for three consecutive manga pages.

Hierarchy:
    PAGE  (hard constraint: page 0 → page 1 → page 2)
      ↓
    PANEL ORDER  (geometry-derived pairwise decisions)
      ↓
    BALLOON ORDER WITHIN PANEL  (geometry-derived pairwise decisions)
      ↓
    TEXT WITHIN BALLOON  (already resolved by adjudicator; not re-ordered here)

Design:
- Pairwise decisions are made by an injectable ReadingOrderRanker.
- Default implementation: DeterministicGeometryRanker (CPU-only, no model).
- Pairwise decisions are accumulated into a directed graph.
- Topological sort resolves acyclic graphs.
- Cycles are detected and resolved via strongest-evidence edge removal.
- Ambiguous pairs carry confidence < 1.0 and explicit evidence.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Protocol, Sequence

from app.schemas.adjudication import BalloonAdjudicationResult
from app.schemas.page import Balloon, BoundingBox, PageRepresentation, Panel
from app.schemas.reading_order import (
    PageReadingOrder,
    ReadingOrderDecision,
    ReadingOrderDiagnostic,
    ReadingOrderItem,
    SequenceReadingOrder,
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _area(bbox: BoundingBox) -> float:
    return max(0.0, bbox.x2 - bbox.x1) * max(0.0, bbox.y2 - bbox.y1)


def _center(bbox: BoundingBox) -> tuple[float, float]:
    return ((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0)


def _intersection_area(a: BoundingBox, b: BoundingBox) -> float:
    w = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    h = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    return w * h


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    inter = _intersection_area(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _valid_bbox(bbox: BoundingBox) -> bool:
    vals = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
    return all(math.isfinite(v) for v in vals) and bbox.x2 > bbox.x1 and bbox.y2 > bbox.y1


def _contains(outer: BoundingBox, inner: BoundingBox) -> bool:
    """Return True if outer fully contains inner (with 1-pixel tolerance)."""
    return (
        outer.x1 <= inner.x1 + 1
        and outer.y1 <= inner.y1 + 1
        and outer.x2 >= inner.x2 - 1
        and outer.y2 >= inner.y2 - 1
    )


def _overlap_ratio(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection / min(area_a, area_b)."""
    inter = _intersection_area(a, b)
    smaller = min(_area(a), _area(b))
    return inter / smaller if smaller > 0 else 0.0


# ---------------------------------------------------------------------------
# Pairwise geometry features for two balloons
# ---------------------------------------------------------------------------

class PairwiseFeatures:
    """Geometry features for an (a, b) balloon pair within a shared context."""

    __slots__ = (
        "cx_a", "cy_a", "cx_b", "cy_b",
        "w_a", "h_a", "w_b", "h_b",
        "iou", "overlap_ratio", "a_contains_b", "b_contains_a",
        "dx", "dy", "horizontal_gap", "vertical_gap",
        "same_row", "same_column",
        "panel_rel_cx_a", "panel_rel_cy_a",
        "panel_rel_cx_b", "panel_rel_cy_b",
        "page_height", "page_width",
    )

    def __init__(
        self,
        a: BoundingBox,
        b: BoundingBox,
        *,
        panel_bbox: BoundingBox | None = None,
        page_width: float = 0.0,
        page_height: float = 0.0,
        same_row_threshold: float = 0.6,
    ) -> None:
        self.cx_a, self.cy_a = _center(a)
        self.cx_b, self.cy_b = _center(b)
        self.w_a = a.x2 - a.x1
        self.h_a = a.y2 - a.y1
        self.w_b = b.x2 - b.x1
        self.h_b = b.y2 - b.y1
        self.iou = _iou(a, b)
        self.overlap_ratio = _overlap_ratio(a, b)
        self.a_contains_b = _contains(a, b)
        self.b_contains_a = _contains(b, a)
        self.dx = self.cx_b - self.cx_a
        self.dy = self.cy_b - self.cy_a
        self.horizontal_gap = max(0.0, max(a.x1, b.x1) - min(a.x2, b.x2))
        self.vertical_gap = max(0.0, max(a.y1, b.y1) - min(a.y2, b.y2))
        # Row/column classification based on vertical overlap
        min_height = min(self.h_a, self.h_b)
        vertical_overlap = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
        self.same_row = (
            min_height > 0
            and vertical_overlap / min_height >= same_row_threshold
        )
        min_width = min(self.w_a, self.w_b)
        horizontal_overlap = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
        self.same_column = (
            min_width > 0
            and horizontal_overlap / min_width >= same_row_threshold
        )
        # Panel-relative coordinates (normalized to [0, 1] if panel available)
        ref = panel_bbox or BoundingBox(x1=0, y1=0, x2=max(page_width, 1), y2=max(page_height, 1))
        pw = max(ref.x2 - ref.x1, 1.0)
        ph = max(ref.y2 - ref.y1, 1.0)
        self.panel_rel_cx_a = (self.cx_a - ref.x1) / pw
        self.panel_rel_cy_a = (self.cy_a - ref.y1) / ph
        self.panel_rel_cx_b = (self.cx_b - ref.x1) / pw
        self.panel_rel_cy_b = (self.cy_b - ref.y1) / ph
        self.page_width = page_width
        self.page_height = page_height


# ---------------------------------------------------------------------------
# Ranker protocol + deterministic baseline
# ---------------------------------------------------------------------------

class ReadingOrderRanker(Protocol):
    """Injectable pairwise ranker: does balloon *a* come before balloon *b*?"""

    def compare(
        self,
        a: Balloon,
        b: Balloon,
        features: PairwiseFeatures,
        context: dict[str, Any],
    ) -> ReadingOrderDecision: ...


class DeterministicGeometryRanker:
    """Deterministic pairwise ranker based purely on spatial geometry.

    Decision logic (in priority order):
    1. If b is fully contained in a → a comes first (container precedes content).
    2. If a is fully contained in b → b comes first.
    3. If balloons are in the same row (significant vertical overlap):
       → use x-center order (right-to-left for manga).
       Tie-break: smaller x1.
    4. If balloons are in the same column (significant horizontal overlap):
       → use y-center order (top to bottom).
    5. General: top-then-right-then-bottom precedence.
       Use weighted y+x score with right-to-left x preference.
    6. Absolute tie-break on balloon id for determinism.

    The `rtl` parameter controls whether same-row balloons read right-to-left.
    It defaults to None (unknown), in which case the ranker uses a weak
    rightward heuristic and marks the decision as low-confidence.
    """

    def __init__(
        self,
        rtl: bool | None = None,
        *,
        same_row_threshold: float = 0.6,
        row_margin_px: float = 10.0,
    ) -> None:
        self.rtl = rtl
        self.same_row_threshold = same_row_threshold
        self.row_margin_px = row_margin_px

    def compare(
        self,
        a: Balloon,
        b: Balloon,
        features: PairwiseFeatures,
        context: dict[str, Any],
    ) -> ReadingOrderDecision:
        def decision(
            a_first: bool,
            confidence: float,
            evidence: dict[str, Any],
        ) -> ReadingOrderDecision:
            return ReadingOrderDecision(
                first_balloon_id=a.id if a_first else b.id,
                second_balloon_id=b.id if a_first else a.id,
                first_precedes_second=True,
                confidence=confidence,
                evidence=evidence,
                method="balloon_geometry",
            )

        # --- containment ---
        if features.b_contains_a:
            return decision(False, 0.85, {"rule": "containment", "outer": b.id, "inner": a.id})
        if features.a_contains_b:
            return decision(True, 0.85, {"rule": "containment", "outer": a.id, "inner": b.id})

        # --- heavy overlap (non-containment): treat as same-row ---
        if features.overlap_ratio >= 0.6:
            return self._same_row_decision(a, b, features, reason="heavy_overlap")

        # --- same row ---
        if features.same_row:
            return self._same_row_decision(a, b, features, reason="same_row")

        # --- general: top-to-bottom precedence, with column tie-break ---
        dy = features.cy_b - features.cy_a
        if abs(dy) > self.row_margin_px:
            a_first = dy > 0  # a is higher (smaller y) → a comes first
            return decision(
                a_first,
                0.82,
                {
                    "rule": "vertical_position",
                    "dy": dy,
                    "cy_a": features.cy_a,
                    "cy_b": features.cy_b,
                },
            )

        # --- within row margin: same-row fallback ---
        return self._same_row_decision(a, b, features, reason="row_margin")

    def _same_row_decision(
        self,
        a: Balloon,
        b: Balloon,
        features: PairwiseFeatures,
        reason: str,
    ) -> ReadingOrderDecision:
        """Order balloons within a row; defaults to right-to-left if rtl is unknown."""
        dx = features.cx_b - features.cx_a
        if self.rtl is True:
            # right-to-left: higher x → comes first
            a_first = dx < 0  # a is to the right of b
            confidence = 0.78
        elif self.rtl is False:
            # left-to-right: lower x → comes first
            a_first = dx > 0  # a is to the left of b
            confidence = 0.78
        else:
            # unknown direction: use right-to-left as weak default (common for manga)
            a_first = dx < 0
            confidence = 0.52

        if dx == 0:
            # deterministic tie-break on id
            a_first = a.id < b.id
            confidence = 0.40

        return ReadingOrderDecision(
            first_balloon_id=a.id if a_first else b.id,
            second_balloon_id=b.id if a_first else a.id,
            first_precedes_second=True,
            confidence=confidence,
            evidence={
                "rule": reason,
                "dx": dx,
                "cx_a": features.cx_a,
                "cx_b": features.cx_b,
                "rtl": self.rtl,
            },
            method="balloon_geometry",
        )


# ---------------------------------------------------------------------------
# Panel ordering (geometry-based)
# ---------------------------------------------------------------------------

def _panel_order_score(panel: Panel) -> tuple[float, float, str]:
    """Stable sort key: top-row panels first, then right-to-left within row."""
    cx, cy = _center(panel.bbox)
    # Using negative cx to sort right-to-left (common for manga)
    return (cy, -cx, panel.id)


def _rank_panels(panels: Sequence[Panel]) -> list[Panel]:
    """Derive pairwise panel order from geometry.

    Strategy:
    1. Partition panels into rows by y-center clustering (gap-based).
    2. Within each row, order by x-center (right-to-left for manga).
    3. Rows are ordered top-to-bottom.

    Returns panels in their estimated reading order.
    """
    if not panels:
        return []
    if len(panels) == 1:
        return [panels[0]]

    valid = [p for p in panels if _valid_bbox(p.bbox)]
    if not valid:
        return list(panels)

    # Sort by y-center then -x for right-to-left
    sorted_panels = sorted(valid, key=_panel_order_score)

    # Group into rows: a new row starts when the vertical gap between consecutive
    # panel y-centers is large relative to the panels' heights.
    rows: list[list[Panel]] = []
    current_row: list[Panel] = [sorted_panels[0]]
    for panel in sorted_panels[1:]:
        prev = current_row[-1]
        prev_cy = (prev.bbox.y1 + prev.bbox.y2) / 2.0
        curr_cy = (panel.bbox.y1 + panel.bbox.y2) / 2.0
        prev_h = prev.bbox.y2 - prev.bbox.y1
        curr_h = panel.bbox.y2 - panel.bbox.y1
        row_height = (prev_h + curr_h) / 2.0
        if row_height > 0 and (curr_cy - prev_cy) / row_height > 0.5:
            rows.append(current_row)
            current_row = [panel]
        else:
            current_row.append(panel)
    rows.append(current_row)

    ordered: list[Panel] = []
    for row in rows:
        # Within each row: right-to-left by x-center (manga default)
        row_sorted = sorted(row, key=lambda p: -_center(p.bbox)[0])
        ordered.extend(row_sorted)
    return ordered


# ---------------------------------------------------------------------------
# Directed precedence graph + topological sort
# ---------------------------------------------------------------------------

class _PrecedenceGraph:
    """Directed graph of balloon IDs with edge confidence for cycle resolution."""

    def __init__(self) -> None:
        self._edges: dict[str, dict[str, float]] = defaultdict(dict)
        self._nodes: set[str] = set()
        self._decisions: list[ReadingOrderDecision] = []

    def add_node(self, node_id: str) -> None:
        self._nodes.add(node_id)

    def add_edge(self, src: str, dst: str, confidence: float, decision: ReadingOrderDecision) -> None:
        self._nodes.add(src)
        self._nodes.add(dst)
        # Only keep the strongest-confidence edge between the same pair
        existing = self._edges[src].get(dst)
        if existing is None or confidence > existing:
            self._edges[src][dst] = confidence
        self._decisions.append(decision)

    @property
    def nodes(self) -> frozenset[str]:
        return frozenset(self._nodes)

    def successors(self, node: str) -> dict[str, float]:
        return dict(self._edges.get(node, {}))

    def predecessors(self, node: str) -> list[str]:
        return [src for src, dsts in self._edges.items() if node in dsts]

    def remove_edge(self, src: str, dst: str) -> None:
        if src in self._edges and dst in self._edges[src]:
            del self._edges[src][dst]

    def in_degree(self, node: str) -> int:
        return sum(1 for dsts in self._edges.values() if node in dsts)

    def topological_sort(self) -> tuple[list[str], list[ReadingOrderDiagnostic]]:
        """Kahn's algorithm. Returns (order, diagnostics)."""
        diagnostics: list[ReadingOrderDiagnostic] = []
        in_deg: dict[str, int] = {n: self.in_degree(n) for n in self._nodes}
        queue: deque[str] = deque(
            sorted(n for n, d in in_deg.items() if d == 0)  # stable: sort for determinism
        )
        result: list[str] = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for successor in sorted(self._edges.get(node, {})):  # sorted for determinism
                in_deg[successor] -= 1
                if in_deg[successor] == 0:
                    queue.append(successor)

        unresolved = [n for n in self._nodes if n not in result]
        if unresolved:
            diagnostics.append(
                ReadingOrderDiagnostic(
                    component="topological_sort",
                    code="cycle_detected",
                    message=(
                        f"Cycle detected among {len(unresolved)} balloon(s); "
                        "resolved by removing weakest-confidence edges."
                    ),
                    balloon_ids=sorted(unresolved),
                )
            )
            # Resolve cycles by removing the weakest edges until acyclic
            result_with_unresolved = result + _resolve_cycles(self, unresolved)
            return result_with_unresolved, diagnostics
        return result, diagnostics


def _resolve_cycles(
    graph: _PrecedenceGraph,
    cycle_nodes: list[str],
) -> list[str]:
    """Break cycles by repeatedly removing the weakest-confidence back-edge."""
    remaining = set(cycle_nodes)
    ordered: list[str] = []
    max_iterations = len(cycle_nodes) * len(cycle_nodes) + 1
    iterations = 0

    while remaining and iterations < max_iterations:
        iterations += 1
        # Find weakest edge among cycle nodes
        weakest_confidence = float("inf")
        weakest_src: str | None = None
        weakest_dst: str | None = None
        for src in remaining:
            for dst, conf in graph.successors(src).items():
                if dst in remaining and conf < weakest_confidence:
                    weakest_confidence = conf
                    weakest_src = src
                    weakest_dst = dst

        if weakest_src is None:
            # No cross edges among remaining — just sort by id for determinism
            break

        graph.remove_edge(weakest_src, weakest_dst)

        # Retry topological sort on remaining nodes
        in_deg: dict[str, int] = {
            n: sum(1 for s in remaining if weakest_dst in graph.successors(s))
            for n in remaining
        }
        queue: deque[str] = deque(sorted(n for n, d in in_deg.items() if d == 0))
        while queue:
            node = queue.popleft()
            if node in remaining:
                ordered.append(node)
                remaining.discard(node)
                for successor in sorted(graph.successors(node)):
                    if successor in remaining:
                        in_deg[successor] = in_deg.get(successor, 1) - 1
                        if in_deg[successor] <= 0:
                            queue.append(successor)

    # Any still-remaining nodes: sort deterministically by id
    ordered.extend(sorted(remaining))
    return ordered


# ---------------------------------------------------------------------------
# Story filter
# ---------------------------------------------------------------------------

def _story_balloon_ids(
    balloons: Sequence[Balloon],
    adjudication_results: dict[str, BalloonAdjudicationResult] | None,
) -> set[str]:
    """Return IDs of balloons that are story-visible.

    When adjudication results are available, use include_in_story.
    When they are not, all balloons are included.
    """
    if adjudication_results is None:
        return {b.id for b in balloons}
    included: set[str] = set()
    for balloon in balloons:
        result = adjudication_results.get(balloon.id)
        if result is None or result.include_in_story:
            included.add(balloon.id)
    return included


# ---------------------------------------------------------------------------
# Per-page reading order
# ---------------------------------------------------------------------------

def _order_balloons_within_group(
    balloons: list[Balloon],
    ranker: ReadingOrderRanker,
    *,
    panel_bbox: BoundingBox | None = None,
    page_width: float = 0.0,
    page_height: float = 0.0,
) -> tuple[list[str], list[ReadingOrderDecision], list[ReadingOrderDiagnostic]]:
    """Build a pairwise graph for the given balloons and return a topological order."""
    if not balloons:
        return [], [], []
    if len(balloons) == 1:
        return [balloons[0].id], [], []

    graph = _PrecedenceGraph()
    for balloon in balloons:
        graph.add_node(balloon.id)

    decisions: list[ReadingOrderDecision] = []
    for i in range(len(balloons)):
        for j in range(i + 1, len(balloons)):
            a = balloons[i]
            b = balloons[j]
            features = PairwiseFeatures(
                a.bbox,
                b.bbox,
                panel_bbox=panel_bbox,
                page_width=page_width,
                page_height=page_height,
            )
            context: dict[str, Any] = {
                "panel_bbox": panel_bbox,
                "page_width": page_width,
                "page_height": page_height,
            }
            dec = ranker.compare(a, b, features, context)
            decisions.append(dec)
            confidence = dec.confidence if dec.confidence is not None else 0.5
            graph.add_edge(dec.first_balloon_id, dec.second_balloon_id, confidence, dec)

    ordered_ids, diag = graph.topological_sort()
    return ordered_ids, decisions, diag


def order_page(
    page: PageRepresentation,
    ranker: ReadingOrderRanker,
    *,
    adjudication_results: dict[str, BalloonAdjudicationResult] | None = None,
    page_width: float = 0.0,
    page_height: float = 0.0,
) -> PageReadingOrder:
    """Derive reading order for all story balloons on one page."""
    diagnostics: list[ReadingOrderDiagnostic] = []
    all_decisions: list[ReadingOrderDecision] = []

    story_ids = _story_balloon_ids(page.balloons, adjudication_results)
    story_balloons = [b for b in page.balloons if b.id in story_ids]

    layout_panels = [p for p in page.panels if p.source == "layout_model" and _valid_bbox(p.bbox)]
    has_panels = bool(layout_panels)

    if not has_panels:
        diagnostics.append(
            ReadingOrderDiagnostic(
                component="panel_order",
                code="no_layout_panels",
                message="No layout-model panels found; using page-level geometric fallback.",
                page_index=page.page_index,
            )
        )
        ordered_ids, decisions, diag = _order_balloons_within_group(
            story_balloons,
            ranker,
            panel_bbox=None,
            page_width=page_width,
            page_height=page_height,
        )
        all_decisions.extend(decisions)
        diagnostics.extend(diag)
        return PageReadingOrder(
            page_index=page.page_index,
            panel_ids_in_order=[None],
            balloon_ids_in_order=ordered_ids,
            decisions=all_decisions,
            diagnostics=diagnostics,
        )

    # --- Panel-aware mode ---
    ordered_panels = _rank_panels(layout_panels)
    panel_ids_in_order: list[str | None] = [p.id for p in ordered_panels]
    panel_map = {p.id: p for p in layout_panels}

    # Assign each story balloon to a panel (or None for no-panel)
    panels_to_balloons: dict[str | None, list[Balloon]] = defaultdict(list)
    for balloon in story_balloons:
        if balloon.panel_id in panel_map:
            panels_to_balloons[balloon.panel_id].append(balloon)
        else:
            # Geometrically assign unattached balloons to the panel with max overlap
            best_panel_id = _best_panel_for_balloon(balloon, layout_panels)
            panels_to_balloons[best_panel_id].append(balloon)
            if best_panel_id is None:
                diagnostics.append(
                    ReadingOrderDiagnostic(
                        component="panel_assignment",
                        code="balloon_unassigned",
                        message=(
                            f"Balloon {balloon.id!r} has no panel_id and no geometric assignment; "
                            "placed in page-level fallback group."
                        ),
                        page_index=page.page_index,
                        balloon_ids=[balloon.id],
                    )
                )

    ordered_balloon_ids: list[str] = []
    # Process in panel order, then unassigned
    panel_order = panel_ids_in_order + ([None] if None in panels_to_balloons else [])
    for panel_id in panel_order:
        group = panels_to_balloons.get(panel_id, [])
        if not group:
            continue
        panel_bbox = panel_map[panel_id].bbox if panel_id is not None else None
        group_ids, decisions, diag = _order_balloons_within_group(
            group,
            ranker,
            panel_bbox=panel_bbox,
            page_width=page_width,
            page_height=page_height,
        )
        all_decisions.extend(decisions)
        diagnostics.extend(diag)
        ordered_balloon_ids.extend(group_ids)

    return PageReadingOrder(
        page_index=page.page_index,
        panel_ids_in_order=panel_ids_in_order,
        balloon_ids_in_order=ordered_balloon_ids,
        decisions=all_decisions,
        diagnostics=diagnostics,
    )


def _best_panel_for_balloon(
    balloon: Balloon,
    panels: list[Panel],
) -> str | None:
    """Find the panel with greatest overlap with the balloon; return None if < 0.1."""
    if not _valid_bbox(balloon.bbox):
        return None
    balloon_area = _area(balloon.bbox)
    if balloon_area == 0:
        return None
    best_ratio = 0.0
    best_id: str | None = None
    for panel in panels:
        ratio = _intersection_area(balloon.bbox, panel.bbox) / balloon_area
        if ratio > best_ratio:
            best_ratio = ratio
            best_id = panel.id
    return best_id if best_ratio >= 0.1 else None


# ---------------------------------------------------------------------------
# Multi-page SequenceReadingOrder
# ---------------------------------------------------------------------------

def order_sequence(
    pages: Sequence[PageRepresentation],
    *,
    ranker: ReadingOrderRanker | None = None,
    adjudication_results: dict[str, BalloonAdjudicationResult] | None = None,
    page_sizes: Sequence[tuple[int, int]] | None = None,
) -> SequenceReadingOrder:
    """Produce a reading order for three (or more) consecutive pages.

    Page order is a HARD constraint: all page 0 balloons precede page 1, etc.

    Args:
        pages: PageRepresentation list in their narrative order (index 0 first).
        ranker: Injectable pairwise ranker; defaults to DeterministicGeometryRanker.
        adjudication_results: Optional map of balloon_id → BalloonAdjudicationResult
            used to filter non-story balloons via include_in_story.
        page_sizes: Optional (width, height) per page for normalization.
    """
    if ranker is None:
        ranker = DeterministicGeometryRanker()

    sequence_diagnostics: list[ReadingOrderDiagnostic] = []
    page_orders: list[PageReadingOrder] = []
    all_items: list[ReadingOrderItem] = []
    global_position = 0

    for page_idx, page in enumerate(pages):
        pw, ph = (page_sizes[page_idx] if page_sizes and page_idx < len(page_sizes) else (0, 0))
        page_order = order_page(
            page,
            ranker,
            adjudication_results=adjudication_results,
            page_width=float(pw),
            page_height=float(ph),
        )
        page_orders.append(page_order)

        for balloon_id in page_order.balloon_ids_in_order:
            # Determine panel_id for this balloon
            panel_id = _balloon_panel_id(balloon_id, page.balloons)
            all_items.append(
                ReadingOrderItem(
                    page_index=page.page_index,
                    panel_id=panel_id,
                    balloon_id=balloon_id,
                    sequence_position=global_position,
                )
            )
            global_position += 1

    return SequenceReadingOrder(
        ordered_items=all_items,
        ordered_balloon_ids=[item.balloon_id for item in all_items],
        page_orders=page_orders,
        diagnostics=sequence_diagnostics,
    )


def _balloon_panel_id(balloon_id: str, balloons: Sequence[Balloon]) -> str | None:
    for b in balloons:
        if b.id == balloon_id:
            return b.panel_id
    return None
