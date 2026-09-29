"""Panel context and uncertain spatial grouping of CTD regions into balloons.

The grouper never changes a TextRegion or its page-coordinate bbox. Without a
layout provider, panel geometry falls back to the full page and balloon
membership uses conservative spatial adjacency, not OCR text similarity.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import quote

import cv2
from pydantic import BaseModel, Field

from app.schemas.page import (
    Balloon,
    BoundingBox,
    PageRepresentation,
    Panel,
    Point2D,
    TextRegion,
)

GroupingMethod = Literal[
    "layout_model",
    "geometry",
    "containment",
    "hybrid",
    "visual_fallback",
]


class PanelProposal(BaseModel):
    bbox: BoundingBox
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class BalloonProposal(BaseModel):
    bbox: BoundingBox
    mask_polygon: list[Point2D] | None = None
    kind: Literal["speech", "thought", "narration", "unknown"] = "unknown"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: dict[str, Any] = Field(default_factory=dict)


class VisualGroupingProposal(BaseModel):
    text_region_ids: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    target_balloon_id: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class GroupingDecision(BaseModel):
    text_region_ids: list[str]
    target_balloon_id: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    method: GroupingMethod
    status: Literal["assigned", "ambiguous", "unassigned"]
    candidate_balloon_ids: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)


class LayoutDiagnostic(BaseModel):
    component: str
    error_type: str
    message: str


class LayoutGroupingResult(BaseModel):
    page: PageRepresentation
    decisions: list[GroupingDecision]
    unassigned_text_region_ids: list[str]
    panel_proposal_count: int = 0
    balloon_proposal_count: int = 0
    diagnostics: list[LayoutDiagnostic] = Field(default_factory=list)


class PanelProvider(Protocol):
    def detect_panels(
        self,
        image_path: Path,
        page_width: int,
        page_height: int,
    ) -> Sequence[PanelProposal]: ...


class BalloonLayoutProvider(Protocol):
    def propose_balloons(
        self,
        image_path: Path,
        page_width: int,
        page_height: int,
    ) -> Sequence[BalloonProposal]: ...


class VisualGroupingProvider(Protocol):
    def resolve_ambiguous(
        self,
        image_path: Path,
        regions: Sequence[TextRegion],
        candidate_region_ids: Sequence[str],
        candidate_balloon_ids: Mapping[str, Sequence[str]],
        panels: Sequence[Panel],
    ) -> Sequence[VisualGroupingProposal]: ...


class _RegionMembership(BaseModel):
    panel_id: str | None = None
    confidence: float | None = None
    status: Literal["assigned", "ambiguous", "unassigned"]
    candidate_panel_ids: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)


class _BalloonGroup(BaseModel):
    region_ids: list[str]
    bbox: BoundingBox
    panel_id: str | None
    confidence: float | None
    method: GroupingMethod
    balloon_id: str | None = None
    mask_polygon: list[Point2D] | None = None
    kind: Literal["speech", "thought", "narration", "unknown"] = "unknown"
    evidence: dict[str, Any] = Field(default_factory=dict)


def _valid_bbox(box: BoundingBox) -> bool:
    values = (box.x1, box.y1, box.x2, box.y2)
    return (
        all(math.isfinite(value) for value in values)
        and box.x2 > box.x1
        and box.y2 > box.y1
    )


def _area(box: BoundingBox) -> float:
    return max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1)


def _intersection_area(first: BoundingBox, second: BoundingBox) -> float:
    width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    return width * height


def _iou(first: BoundingBox, second: BoundingBox) -> float:
    intersection = _intersection_area(first, second)
    union = _area(first) + _area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _contains(outer: BoundingBox, inner: BoundingBox) -> bool:
    return (
        outer.x1 <= inner.x1
        and outer.y1 <= inner.y1
        and outer.x2 >= inner.x2
        and outer.y2 >= inner.y2
    )


def _union_bbox(boxes: Sequence[BoundingBox]) -> BoundingBox:
    return BoundingBox(
        x1=min(box.x1 for box in boxes),
        y1=min(box.y1 for box in boxes),
        x2=max(box.x2 for box in boxes),
        y2=max(box.y2 for box in boxes),
    )


def _as_panel_proposal(raw: Any) -> PanelProposal:
    return raw if isinstance(raw, PanelProposal) else PanelProposal.model_validate(raw)


def _as_balloon_proposal(raw: Any) -> BalloonProposal:
    return raw if isinstance(raw, BalloonProposal) else BalloonProposal.model_validate(raw)


def _deterministic_id(kind: str, sequence_id: str, page_index: int, ordinal: int) -> str:
    safe_sequence_id = quote(sequence_id, safe="-_.") or "_"
    return f"{kind}-{safe_sequence_id}-p{page_index:02d}-{ordinal:04d}"


def _spatial_relation(
    first: BoundingBox,
    second: BoundingBox,
) -> tuple[Literal["strong", "weak"] | None, GroupingMethod | None, float, dict[str, Any]]:
    intersection = _intersection_area(first, second)
    smaller_area = min(_area(first), _area(second))
    overlap_of_smaller = intersection / smaller_area if smaller_area else 0.0
    if overlap_of_smaller >= 0.8:
        method: GroupingMethod = "containment" if _contains(first, second) or _contains(second, first) else "geometry"
        return "strong", method, 0.88, {"overlap_of_smaller": overlap_of_smaller}
    if overlap_of_smaller >= 0.35:
        return "strong", "geometry", 0.76, {"overlap_of_smaller": overlap_of_smaller}

    width_a, height_a = first.x2 - first.x1, first.y2 - first.y1
    width_b, height_b = second.x2 - second.x1, second.y2 - second.y1
    overlap_width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    overlap_height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    horizontal_alignment = overlap_width / min(width_a, width_b)
    vertical_alignment = overlap_height / min(height_a, height_b)
    gap_x = max(0.0, max(first.x1, second.x1) - min(first.x2, second.x2))
    gap_y = max(0.0, max(first.y1, second.y1) - min(first.y2, second.y2))
    scale = max(1.0, min(height_a, height_b), min(width_a, width_b))
    normalized_gap = math.hypot(gap_x, gap_y) / scale

    aligned_neighbors = (
        horizontal_alignment >= 0.45 and gap_y <= 0.55 * max(height_a, height_b)
    ) or (
        vertical_alignment >= 0.45 and gap_x <= 0.4 * max(width_a, width_b)
    )
    evidence = {
        "horizontal_alignment": horizontal_alignment,
        "vertical_alignment": vertical_alignment,
        "normalized_gap": normalized_gap,
        "overlap_of_smaller": overlap_of_smaller,
    }
    if aligned_neighbors:
        return "strong", "geometry", 0.72, evidence
    if normalized_gap <= 1.6 and (
        horizontal_alignment >= 0.2 or vertical_alignment >= 0.2
    ):
        return "weak", "geometry", 0.4, evidence
    return None, None, 0.0, evidence


class PageLayoutGrouper:
    """Add panel and provisional balloon ownership without editing CTD regions."""

    def __init__(
        self,
        panel_provider: PanelProvider | None = None,
        balloon_provider: BalloonLayoutProvider | None = None,
        visual_grouping_provider: VisualGroupingProvider | None = None,
        *,
        duplicate_iou_threshold: float = 0.85,
        visual_acceptance_threshold: float = 0.65,
    ) -> None:
        self.panel_provider = panel_provider
        self.balloon_provider = balloon_provider
        self.visual_grouping_provider = visual_grouping_provider
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.visual_acceptance_threshold = visual_acceptance_threshold

    def group_page(
        self,
        page: PageRepresentation,
        sequence_id: str,
        image_path: str | Path | None = None,
        *,
        page_size: tuple[int, int] | None = None,
    ) -> LayoutGroupingResult:
        source_path = Path(image_path or page.image_path)
        diagnostics: list[LayoutDiagnostic] = []
        width, height = self._page_size(source_path, page, page_size, diagnostics)
        panels = self._detect_panels(source_path, width, height, page.page_index, sequence_id, diagnostics)
        memberships = self._assign_panels(page.text_regions, panels)

        groups: list[_BalloonGroup] = []
        decisions: list[GroupingDecision] = []
        explicitly_claimed: set[str] = set()
        explicitly_ambiguous: set[str] = set()
        explicit_ambiguity: dict[str, tuple[list[str], dict[str, Any]]] = {}
        layout_provider_succeeded = False
        balloon_proposal_count = 0

        if self.balloon_provider is not None:
            (
                explicit_groups,
                claimed,
                ambiguous,
                layout_provider_succeeded,
                balloon_proposal_count,
            ) = self._layout_groups(
                source_path,
                width,
                height,
                page.text_regions,
                panels,
                memberships,
                sequence_id,
                page.page_index,
                diagnostics,
            )
            groups.extend(explicit_groups)
            explicitly_claimed.update(claimed)
            explicit_ambiguity.update(ambiguous)
            explicitly_ambiguous.update(ambiguous)

        if layout_provider_succeeded:
            geometry_groups, weak_regions, relation_evidence = [], set(), {}
        else:
            geometry_groups, weak_regions, relation_evidence = self._geometry_groups(
                page.text_regions,
                memberships,
                explicitly_claimed | explicitly_ambiguous,
            )
        groups.extend(geometry_groups)

        unresolved = {
            region.id
            for region in page.text_regions
            if region.id not in explicitly_claimed
            and region.id not in explicitly_ambiguous
            and all(region.id not in group.region_ids for group in geometry_groups)
        }
        unresolved.update(explicitly_ambiguous)
        ambiguous_region_ids = sorted((unresolved & weak_regions) | set(explicit_ambiguity))
        visual_groups: list[_BalloonGroup] = []
        if ambiguous_region_ids and self.visual_grouping_provider is not None:
            visual_groups, _ = self._visual_groups(
                source_path,
                page.text_regions,
                panels,
                memberships,
                ambiguous_region_ids,
                explicit_ambiguity,
                groups,
                sequence_id,
                page.page_index,
                diagnostics,
            )
            groups.extend(visual_groups)

        grouped_regions = {region_id for group in groups for region_id in group.region_ids}
        for region in page.text_regions:
            if region.id in explicitly_claimed or region.id in grouped_regions:
                continue
            membership = memberships[region.id]
            region_box = region.bbox
            if not _valid_bbox(region_box):
                decisions.append(
                    GroupingDecision(
                        text_region_ids=[region.id],
                        method="geometry",
                        status="unassigned",
                        evidence={"reason": "malformed_text_region_bbox"},
                    )
                )
                continue
            if layout_provider_succeeded and region.id not in explicit_ambiguity:
                decisions.append(
                    GroupingDecision(
                        text_region_ids=[region.id],
                        method="layout_model",
                        status="unassigned",
                        evidence={"reason": "no_balloon_proposal_overlap"},
                    )
                )
                continue
            if region.id in ambiguous_region_ids or region.id in explicitly_ambiguous:
                if region.id in explicit_ambiguity:
                    candidate_balloon_ids, evidence = explicit_ambiguity[region.id]
                    decisions.append(
                        GroupingDecision(
                            text_region_ids=[region.id],
                            method="layout_model",
                            status="ambiguous",
                            candidate_balloon_ids=candidate_balloon_ids,
                            evidence=evidence,
                        )
                    )
                    continue
                possible = [
                    other_id
                    for other_id in relation_evidence.get(region.id, {})
                    if other_id not in grouped_regions
                ]
                decisions.append(
                    GroupingDecision(
                        text_region_ids=[region.id],
                        method="geometry",
                        status="ambiguous",
                        candidate_balloon_ids=[],
                        evidence={
                            "reason": "weak_spatial_relationship",
                            "candidate_region_ids": possible,
                        },
                    )
                )
                continue
            if membership.status != "assigned":
                decisions.append(
                    GroupingDecision(
                        text_region_ids=[region.id],
                        method="geometry",
                        status="ambiguous" if membership.status == "ambiguous" else "unassigned",
                        evidence={
                            "reason": "panel_ownership_uncertain",
                            "candidate_panel_ids": membership.candidate_panel_ids,
                            **membership.evidence,
                        },
                    )
                )
                continue

            singleton = _BalloonGroup(
                region_ids=[region.id],
                bbox=region_box,
                panel_id=membership.panel_id,
                confidence=0.5,
                method="geometry",
                evidence={"relation": "singleton_ctd_region", "category_preserved": region.category},
            )
            groups.append(singleton)

        balloon_records = self._materialize_balloons(groups, sequence_id, page.page_index)
        balloons = [balloon for balloon, _ in balloon_records]
        for balloon, region_ids in balloon_records:
            decisions.append(
                GroupingDecision(
                    text_region_ids=region_ids,
                    target_balloon_id=balloon.id,
                    confidence=balloon.confidence,
                    method=balloon.grouping_method,
                    status="assigned",
                    candidate_balloon_ids=[balloon.id],
                    evidence=balloon.grouping_evidence,
                )
            )

        grouped_page = page.model_copy(update={"panels": panels, "balloons": balloons})
        unassigned = sorted(
            region_id
            for decision in decisions
            if decision.status != "assigned"
            for region_id in decision.text_region_ids
        )
        return LayoutGroupingResult(
            page=grouped_page,
            decisions=decisions,
            unassigned_text_region_ids=list(dict.fromkeys(unassigned)),
            panel_proposal_count=sum(panel.source == "layout_model" for panel in panels),
            balloon_proposal_count=balloon_proposal_count,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _page_size(
        image_path: Path,
        page: PageRepresentation,
        requested_size: tuple[int, int] | None,
        diagnostics: list[LayoutDiagnostic],
    ) -> tuple[int, int]:
        if requested_size is not None:
            width, height = requested_size
            if width > 0 and height > 0:
                return width, height
            diagnostics.append(
                LayoutDiagnostic(
                    component="page_dimensions",
                    error_type="InvalidPageSize",
                    message="Supplied page dimensions must be positive.",
                )
            )
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            height, width = image.shape[:2]
            return width, height
        valid_boxes = [region.bbox for region in page.text_regions if _valid_bbox(region.bbox)]
        width = max((math.ceil(box.x2) for box in valid_boxes), default=1)
        height = max((math.ceil(box.y2) for box in valid_boxes), default=1)
        diagnostics.append(
            LayoutDiagnostic(
                component="page_dimensions",
                error_type="ImageReadError",
                message=f"Could not read {image_path}; using CTD extent fallback dimensions.",
            )
        )
        return max(1, width), max(1, height)

    def _detect_panels(
        self,
        image_path: Path,
        width: int,
        height: int,
        page_index: int,
        sequence_id: str,
        diagnostics: list[LayoutDiagnostic],
    ) -> list[Panel]:
        proposals: list[PanelProposal] = []
        if self.panel_provider is not None:
            try:
                raw_proposals = self.panel_provider.detect_panels(image_path, width, height)
                proposals = [_as_panel_proposal(raw) for raw in raw_proposals]
            except Exception as exc:  # noqa: BLE001 - panel detection is optional
                diagnostics.append(
                    LayoutDiagnostic(
                        component="panel_provider",
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )

        normalized: list[PanelProposal] = []
        for proposal in proposals:
            if not _valid_bbox(proposal.bbox):
                diagnostics.append(
                    LayoutDiagnostic(
                        component="panel_provider",
                        error_type="InvalidPanelBBox",
                        message="Discarded panel proposal with malformed bbox.",
                    )
                )
                continue
            clipped = BoundingBox(
                x1=max(0.0, min(float(width), proposal.bbox.x1)),
                y1=max(0.0, min(float(height), proposal.bbox.y1)),
                x2=max(0.0, min(float(width), proposal.bbox.x2)),
                y2=max(0.0, min(float(height), proposal.bbox.y2)),
            )
            if not _valid_bbox(clipped):
                continue
            candidate = proposal.model_copy(update={"bbox": clipped})
            if any(
                _iou(candidate.bbox, previous.bbox) >= self.duplicate_iou_threshold
                for previous in normalized
            ):
                continue
            normalized.append(candidate)

        if not normalized:
            return [
                Panel(
                    id=_deterministic_id("panel", sequence_id, page_index, 1),
                    bbox=BoundingBox(x1=0, y1=0, x2=width, y2=height),
                    source="page_fallback",
                )
            ]

        normalized.sort(key=lambda proposal: (
            proposal.bbox.y1,
            proposal.bbox.x1,
            proposal.bbox.y2,
            proposal.bbox.x2,
        ))
        return [
            Panel(
                id=_deterministic_id("panel", sequence_id, page_index, index),
                bbox=proposal.bbox,
                confidence=proposal.confidence,
                source="layout_model",
            )
            for index, proposal in enumerate(normalized, start=1)
        ]

    @staticmethod
    def _assign_panels(
        regions: Sequence[TextRegion],
        panels: Sequence[Panel],
    ) -> dict[str, _RegionMembership]:
        memberships: dict[str, _RegionMembership] = {}
        for region in regions:
            box = region.bbox
            if not _valid_bbox(box):
                memberships[region.id] = _RegionMembership(
                    status="unassigned",
                    evidence={"reason": "malformed_text_region_bbox"},
                )
                continue
            containing = [panel for panel in panels if _contains(panel.bbox, box)]
            if containing:
                panel = min(containing, key=lambda item: _area(item.bbox))
                memberships[region.id] = _RegionMembership(
                    panel_id=panel.id,
                    confidence=0.95 if panel.source == "layout_model" else 0.5,
                    status="assigned",
                    candidate_panel_ids=[panel.id],
                    evidence={"relationship": "bbox_containment"},
                )
                continue

            overlaps = [
                (
                    _intersection_area(panel.bbox, box) / _area(box),
                    panel,
                )
                for panel in panels
                if _intersection_area(panel.bbox, box) > 0
            ]
            overlaps.sort(key=lambda pair: pair[0], reverse=True)
            if overlaps and overlaps[0][0] >= 0.8 and (
                len(overlaps) == 1 or overlaps[0][0] - overlaps[1][0] >= 0.2
            ):
                coverage, panel = overlaps[0]
                memberships[region.id] = _RegionMembership(
                    panel_id=panel.id,
                    confidence=0.65,
                    status="assigned",
                    candidate_panel_ids=[panel.id],
                    evidence={"relationship": "partial_overlap", "region_coverage": coverage},
                )
            elif overlaps:
                memberships[region.id] = _RegionMembership(
                    status="ambiguous",
                    candidate_panel_ids=[panel.id for _, panel in overlaps],
                    evidence={
                        "relationship": "ambiguous_panel_overlap",
                        "panel_coverages": {
                            panel.id: coverage for coverage, panel in overlaps
                        },
                    },
                )
            else:
                memberships[region.id] = _RegionMembership(
                    status="unassigned",
                    evidence={"relationship": "outside_detected_panels"},
                )
        return memberships

    def _layout_groups(
        self,
        image_path: Path,
        page_width: int,
        page_height: int,
        regions: Sequence[TextRegion],
        panels: Sequence[Panel],
        memberships: Mapping[str, _RegionMembership],
        sequence_id: str,
        page_index: int,
        diagnostics: list[LayoutDiagnostic],
    ) -> tuple[
        list[_BalloonGroup],
        set[str],
        dict[str, tuple[list[str], dict[str, Any]]],
        bool,
        int,
    ]:
        assert self.balloon_provider is not None
        try:
            raw_proposals = self.balloon_provider.propose_balloons(
                image_path, page_width, page_height
            )
            proposals = [_as_balloon_proposal(raw) for raw in raw_proposals]
        except Exception as exc:  # noqa: BLE001 - geometry grouping remains available
            diagnostics.append(
                LayoutDiagnostic(
                    component="balloon_layout_provider",
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            return [], set(), {}, False, 0

        normalized: list[BalloonProposal] = []
        for proposal in proposals:
            if not _valid_bbox(proposal.bbox):
                diagnostics.append(
                    LayoutDiagnostic(
                        component="balloon_layout_provider",
                        error_type="InvalidBalloonBBox",
                        message="Discarded balloon proposal with malformed bbox.",
                    )
                )
                continue
            normalized.append(proposal)

        normalized.sort(
            key=lambda proposal: (
                proposal.bbox.y1,
                proposal.bbox.x1,
                proposal.bbox.y2,
                proposal.bbox.x2,
                -(proposal.confidence or 0.0),
            )
        )
        deduplicated: list[BalloonProposal] = []
        for proposal in normalized:
            duplicate = next(
                (
                    previous
                    for previous in deduplicated
                    if _iou(proposal.bbox, previous.bbox) >= self.duplicate_iou_threshold
                ),
                None,
            )
            if duplicate is None:
                deduplicated.append(proposal)

        proposal_ids = [
            _deterministic_id("balloon-layout", sequence_id, page_index, index)
            for index in range(1, len(deduplicated) + 1)
        ]
        assigned_by_proposal: dict[int, list[str]] = defaultdict(list)
        ambiguous: dict[str, tuple[list[str], dict[str, Any]]] = {}
        claimed: set[str] = set()
        for region in regions:
            if not _valid_bbox(region.bbox):
                continue
            region_area = _area(region.bbox)
            overlaps = [
                (
                    _intersection_area(region.bbox, proposal.bbox) / region_area,
                    index,
                )
                for index, proposal in enumerate(deduplicated)
                if region_area > 0
                and _intersection_area(region.bbox, proposal.bbox) > 0
            ]
            overlaps.sort(key=lambda item: (-item[0], item[1]))
            compatible = [(score, index) for score, index in overlaps if score >= 0.25]
            if not compatible:
                continue
            best_score, best_index = compatible[0]
            next_score = compatible[1][0] if len(compatible) > 1 else 0.0
            if best_score >= 0.5 and best_score - next_score >= 0.15:
                assigned_by_proposal[best_index].append(region.id)
                claimed.add(region.id)
            else:
                ambiguous[region.id] = (
                    [proposal_ids[index] for _, index in compatible],
                    {
                        "reason": "competing_or_partial_balloon_overlap",
                        "overlap_by_candidate_balloon": {
                            proposal_ids[index]: score for score, index in compatible
                        },
                    },
                )

        groups: list[_BalloonGroup] = []
        for proposal_index, proposal in enumerate(deduplicated):
            members = list(assigned_by_proposal.get(proposal_index, []))
            panel_ids = {
                memberships[region_id].panel_id for region_id in members
                if memberships[region_id].panel_id is not None
            }
            if len(panel_ids) > 1:
                for region_id in members:
                    ambiguous[region_id] = (
                        [proposal_ids[proposal_index]],
                        {"reason": "ctd_fragments_cross_panel_ownership"},
                    )
                    claimed.discard(region_id)
                members = []
            groups.append(
                _BalloonGroup(
                    region_ids=members,
                    bbox=proposal.bbox,
                    panel_id=next(iter(panel_ids)) if len(panel_ids) == 1 else None,
                    confidence=proposal.confidence,
                    method="layout_model",
                    balloon_id=proposal_ids[proposal_index],
                    mask_polygon=proposal.mask_polygon,
                    kind=proposal.kind,
                    evidence={**proposal.evidence, "proposal_index": proposal_index},
                )
            )
        return groups, claimed, ambiguous, True, len(deduplicated)

    def _geometry_groups(
        self,
        regions: Sequence[TextRegion],
        memberships: Mapping[str, _RegionMembership],
        excluded: set[str],
    ) -> tuple[list[_BalloonGroup], set[str], dict[str, dict[str, Any]]]:
        usable = [
            region for region in regions
            if region.id not in excluded
            and _valid_bbox(region.bbox)
            and memberships[region.id].status == "assigned"
            and memberships[region.id].panel_id is not None
        ]
        strong_edges: dict[str, set[str]] = defaultdict(set)
        weak_regions: set[str] = set()
        relation_evidence: dict[str, dict[str, Any]] = defaultdict(dict)
        relation_methods: dict[tuple[str, str], GroupingMethod] = {}
        relation_confidence: dict[tuple[str, str], float] = {}
        for index, first in enumerate(usable):
            for second in usable[index + 1 :]:
                if memberships[first.id].panel_id != memberships[second.id].panel_id:
                    continue
                strength, method, confidence, evidence = _spatial_relation(
                    first.bbox, second.bbox
                )
                if strength is None:
                    continue
                relation_evidence[first.id][second.id] = evidence
                relation_evidence[second.id][first.id] = evidence
                if strength == "strong":
                    strong_edges[first.id].add(second.id)
                    strong_edges[second.id].add(first.id)
                    relation_methods[(first.id, second.id)] = method or "geometry"
                    relation_confidence[(first.id, second.id)] = confidence
                else:
                    weak_regions.update((first.id, second.id))

        by_id = {region.id: region for region in usable}
        remaining = set(by_id)
        components: list[list[str]] = []
        while remaining:
            root = min(remaining)
            component: set[str] = set()
            stack = [root]
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                stack.extend(strong_edges[current] - component)
            remaining.difference_update(component)
            components.append(sorted(component, key=lambda region_id: next(
                index for index, region in enumerate(regions) if region.id == region_id
            )))

        groups: list[_BalloonGroup] = []
        for component in components:
            panel_id = memberships[component[0]].panel_id
            if len(component) == 1 and component[0] in weak_regions:
                continue
            pairs = [
                relation_confidence.get((first, second), relation_confidence.get((second, first), 0.5))
                for index, first in enumerate(component)
                for second in component[index + 1 :]
            ]
            methods = {
                relation_methods.get((first, second), relation_methods.get((second, first), "geometry"))
                for index, first in enumerate(component)
                for second in component[index + 1 :]
            }
            method: GroupingMethod = next(iter(methods)) if len(methods) == 1 else "hybrid"
            group_confidence = min(pairs) if pairs else 0.5
            groups.append(
                _BalloonGroup(
                    region_ids=component,
                    bbox=_union_bbox([by_id[region_id].bbox for region_id in component]),
                    panel_id=panel_id,
                    confidence=group_confidence,
                    method=method,
                    evidence={
                        "relationship": "spatial_component",
                        "pair_count": len(pairs),
                        "text_similarity_used": False,
                    },
                )
            )
        return groups, weak_regions, relation_evidence

    def _visual_groups(
        self,
        image_path: Path,
        regions: Sequence[TextRegion],
        panels: Sequence[Panel],
        memberships: Mapping[str, _RegionMembership],
        ambiguous_ids: Sequence[str],
        layout_ambiguity: Mapping[str, tuple[list[str], dict[str, Any]]],
        existing_groups: list[_BalloonGroup],
        sequence_id: str,
        page_index: int,
        diagnostics: list[LayoutDiagnostic],
    ) -> tuple[list[_BalloonGroup], set[str]]:
        assert self.visual_grouping_provider is not None
        by_id = {region.id: region for region in regions}
        try:
            proposals = self.visual_grouping_provider.resolve_ambiguous(
                image_path,
                [by_id[region_id] for region_id in ambiguous_ids],
                ambiguous_ids,
                {
                    region_id: layout_ambiguity[region_id][0]
                    for region_id in ambiguous_ids
                    if region_id in layout_ambiguity
                },
                panels,
            )
        except Exception as exc:  # noqa: BLE001 - visual confirmation is an optional fallback
            diagnostics.append(
                LayoutDiagnostic(
                    component="visual_grouping_provider",
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            return [], set()

        groups: list[_BalloonGroup] = []
        accepted_ids: set[str] = set()
        resolved_ids: set[str] = set()
        for proposal_index, raw in enumerate(proposals, start=1):
            try:
                proposal = raw if isinstance(raw, VisualGroupingProposal) else VisualGroupingProposal.model_validate(raw)
            except Exception as exc:  # noqa: BLE001 - reject malformed visual suggestions individually
                diagnostics.append(
                    LayoutDiagnostic(
                        component="visual_grouping_provider",
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                continue
            members = list(dict.fromkeys(proposal.text_region_ids))
            if (
                proposal.confidence < self.visual_acceptance_threshold
                or any(member not in ambiguous_ids for member in members)
                or accepted_ids.intersection(members)
            ):
                continue

            if proposal.target_balloon_id is not None:
                allowed_targets = [
                    set(layout_ambiguity.get(member, ([], {}))[0])
                    for member in members
                ]
                target_group = next(
                    (
                        candidate
                        for candidate in existing_groups
                        if candidate.balloon_id == proposal.target_balloon_id
                    ),
                    None,
                )
                if (
                    target_group is None
                    or any(proposal.target_balloon_id not in targets for targets in allowed_targets)
                ):
                    continue
                panel_ids = {
                    memberships[member].panel_id
                    for member in members
                    if memberships[member].panel_id is not None
                }
                if (
                    target_group.panel_id is not None
                    and panel_ids
                    and panel_ids != {target_group.panel_id}
                ):
                    continue
                target_group_index = existing_groups.index(target_group)
                existing_groups[target_group_index] = target_group.model_copy(update={
                    "region_ids": list(dict.fromkeys(target_group.region_ids + members)),
                    "panel_id": (
                        next(iter(panel_ids))
                        if target_group.panel_id is None and len(panel_ids) == 1
                        else target_group.panel_id
                    ),
                    "bbox": _union_bbox(
                        [target_group.bbox] + [by_id[member].bbox for member in members]
                    ),
                    "confidence": proposal.confidence,
                    "evidence": {
                        **target_group.evidence,
                        "visual_membership_confirmation": proposal.evidence,
                    },
                })
                accepted_ids.update(members)
                resolved_ids.update(members)
                continue

            if len(members) < 2:
                continue
            panel_ids = {
                memberships[member].panel_id for member in members
                if memberships[member].panel_id is not None
            }
            if len(panel_ids) != 1:
                continue
            accepted_ids.update(members)
            groups.append(
                _BalloonGroup(
                    region_ids=members,
                    bbox=_union_bbox([by_id[member].bbox for member in members]),
                    panel_id=next(iter(panel_ids)),
                    confidence=proposal.confidence,
                    method="visual_fallback",
                    evidence=proposal.evidence,
                )
            )
        return groups, resolved_ids

    @staticmethod
    def _materialize_balloons(
        groups: Sequence[_BalloonGroup],
        sequence_id: str,
        page_index: int,
    ) -> list[tuple[Balloon, list[str]]]:
        ordered = sorted(
            (group for group in groups if group.region_ids),
            key=lambda group: (
                group.bbox.y1,
                group.bbox.x1,
                group.bbox.y2,
                group.bbox.x2,
                tuple(group.region_ids),
            ),
        )
        records: list[tuple[Balloon, list[str]]] = []
        for index, group in enumerate(ordered, start=1):
            records.append((
                Balloon(
                    id=group.balloon_id
                    or _deterministic_id("balloon-geometry", sequence_id, page_index, index),
                    bbox=group.bbox,
                    panel_id=group.panel_id,
                    confidence=group.confidence,
                    grouping_method=group.method,
                    grouping_evidence=group.evidence,
                    mask_polygon=group.mask_polygon,
                    kind=group.kind,
                    text_region_ids=group.region_ids,
                ),
                group.region_ids,
            ))
        return records