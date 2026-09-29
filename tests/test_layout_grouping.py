from app.layout_grouping import (
    BalloonProposal,
    PageLayoutGrouper,
    PanelProposal,
    VisualGroupingProposal,
)
from app.schemas.page import BoundingBox, PageRepresentation, TextRegion


def make_region(
    region_id: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> TextRegion:
    return TextRegion(
        id=region_id,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        raw_text="",
    )


def make_page(*regions: TextRegion, page_index: int = 0) -> PageRepresentation:
    return PageRepresentation(
        page_index=page_index,
        image_path="unused-page.png",
        text_regions=list(regions),
    )


def group(page: PageRepresentation, grouper: PageLayoutGrouper | None = None):
    return (grouper or PageLayoutGrouper()).group_page(
        page,
        "sequence-test",
        page_size=(200, 160),
    )


def test_one_ctd_region_creates_one_provisional_balloon():
    region = make_region("ctd-region-0001", 10, 10, 40, 25)
    result = group(make_page(region))

    assert len(result.page.balloons) == 1
    assert result.page.balloons[0].text_region_ids == [region.id]
    assert result.page.balloons[0].kind == "unknown"
    assert result.page.balloons[0].confidence == 0.5


def test_multiple_ctd_regions_group_by_spatial_relationship():
    first = make_region("ctd-region-0001", 20, 20, 55, 35)
    second = make_region("ctd-region-0002", 21, 38, 56, 53)
    third = make_region("ctd-region-0003", 22, 56, 57, 71)
    result = group(make_page(first, second, third))

    assert len(result.page.balloons) == 1
    assert result.page.balloons[0].text_region_ids == [first.id, second.id, third.id]
    assert result.page.balloons[0].grouping_method == "geometry"


def test_multiple_balloons_can_share_one_panel():
    upper = make_region("ctd-upper", 10, 10, 35, 25)
    lower = make_region("ctd-lower", 120, 110, 150, 125)
    result = group(make_page(upper, lower))

    assert len(result.page.panels) == 1
    assert result.page.panels[0].source == "page_fallback"
    assert len(result.page.balloons) == 2
    assert {balloon.panel_id for balloon in result.page.balloons} == {
        result.page.panels[0].id
    }


def test_regions_in_different_detected_panels_are_not_merged():
    first = make_region("ctd-left", 10, 20, 50, 40)
    second = make_region("ctd-right", 55, 20, 95, 40)

    class Panels:
        def detect_panels(self, image_path, page_width, page_height):
            return [
                PanelProposal(bbox=BoundingBox(x1=0, y1=0, x2=50, y2=100)),
                PanelProposal(bbox=BoundingBox(x1=50, y1=0, x2=100, y2=100)),
            ]

    result = group(make_page(first, second), PageLayoutGrouper(panel_provider=Panels()))

    assert len(result.page.panels) == 2
    assert len(result.page.balloons) == 2
    assert result.page.balloons[0].panel_id != result.page.balloons[1].panel_id


def test_ctd_region_outside_detected_panels_remains_unassigned():
    region = make_region("ctd-outside", 130, 20, 150, 40)

    class Panels:
        def detect_panels(self, image_path, page_width, page_height):
            return [PanelProposal(bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100))]

    result = group(make_page(region), PageLayoutGrouper(panel_provider=Panels()))

    assert result.page.balloons == []
    assert result.unassigned_text_region_ids == [region.id]
    assert result.decisions[0].status == "unassigned"


def test_panel_detection_failure_falls_back_without_losing_regions():
    region = make_region("ctd-region", 20, 20, 40, 35)

    class BrokenPanels:
        def detect_panels(self, image_path, page_width, page_height):
            raise RuntimeError("detector unavailable")

    result = group(make_page(region), PageLayoutGrouper(panel_provider=BrokenPanels()))

    assert len(result.page.panels) == 1
    assert result.page.panels[0].source == "page_fallback"
    assert result.page.text_regions == [region]
    assert len(result.page.balloons) == 1
    assert result.diagnostics[0].component == "panel_provider"


def test_duplicate_panel_and_balloon_proposals_are_suppressed_or_merged():
    first = make_region("ctd-1", 20, 20, 40, 35)
    second = make_region("ctd-2", 20, 38, 40, 53)

    class Panels:
        def detect_panels(self, image_path, page_width, page_height):
            bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)
            return [PanelProposal(bbox=bbox), PanelProposal(bbox=bbox)]

    class Balloons:
        def propose_balloons(self, image_path, regions, panels):
            bbox = BoundingBox(x1=15, y1=15, x2=50, y2=60)
            return [
                BalloonProposal(bbox=bbox, text_region_ids=[first.id]),
                BalloonProposal(bbox=bbox, text_region_ids=[second.id]),
            ]

    result = group(
        make_page(first, second),
        PageLayoutGrouper(panel_provider=Panels(), balloon_provider=Balloons()),
    )

    assert len(result.page.panels) == 1
    assert len(result.page.balloons) == 1
    assert set(result.page.balloons[0].text_region_ids) == {first.id, second.id}
    assert result.page.balloons[0].grouping_method == "layout_model"


def test_bbox_containment_assigns_region_to_the_smallest_panel():
    region = make_region("ctd-inner", 25, 25, 35, 35)

    class Panels:
        def detect_panels(self, image_path, page_width, page_height):
            return [
                PanelProposal(bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100)),
                PanelProposal(bbox=BoundingBox(x1=20, y1=20, x2=50, y2=50)),
            ]

    result = group(make_page(region), PageLayoutGrouper(panel_provider=Panels()))

    assert len(result.page.panels) == 2
    assigned_panel = next(
        panel for panel in result.page.panels
        if panel.id == result.page.balloons[0].panel_id
    )
    assert assigned_panel.bbox == BoundingBox(x1=20, y1=20, x2=50, y2=50)


def test_partial_panel_overlap_is_kept_uncertain():
    region = make_region("ctd-border", 90, 20, 110, 40)

    class Panels:
        def detect_panels(self, image_path, page_width, page_height):
            return [PanelProposal(bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100))]

    result = group(make_page(region), PageLayoutGrouper(panel_provider=Panels()))

    assert result.page.balloons == []
    assert result.decisions[0].status == "ambiguous"
    assert result.decisions[0].evidence["relationship"] == "ambiguous_panel_overlap"


def test_partial_bbox_overlap_can_group_regions_with_geometry_evidence():
    first = make_region("ctd-overlap-a", 20, 20, 40, 40)
    second = make_region("ctd-overlap-b", 32, 20, 52, 40)
    result = group(make_page(first, second))

    assert len(result.page.balloons) == 1
    assert result.page.balloons[0].text_region_ids == [first.id, second.id]
    assert result.page.balloons[0].grouping_evidence["text_similarity_used"] is False


def test_weak_spatial_relationship_is_represented_as_ambiguous():
    first = make_region("ctd-near-a", 10, 20, 30, 40)
    second = make_region("ctd-near-b", 50, 20, 70, 40)
    result = group(make_page(first, second))

    assert result.page.balloons == []
    assert result.unassigned_text_region_ids == [first.id, second.id]
    assert all(decision.status == "ambiguous" for decision in result.decisions)
    assert all(decision.evidence["reason"] == "weak_spatial_relationship" for decision in result.decisions)


def test_visual_grouping_provider_can_resolve_ambiguous_geometry():
    first = make_region("ctd-near-a", 10, 20, 30, 40)
    second = make_region("ctd-near-b", 50, 20, 70, 40)

    class VisualGrouper:
        def resolve_ambiguous(self, image_path, regions, candidate_region_ids, panels):
            assert set(candidate_region_ids) == {first.id, second.id}
            return [VisualGroupingProposal(
                text_region_ids=[first.id, second.id],
                confidence=0.82,
                evidence={"visual_relation": "same enclosure"},
            )]

    result = group(
        make_page(first, second),
        PageLayoutGrouper(visual_grouping_provider=VisualGrouper()),
    )

    assert len(result.page.balloons) == 1
    assert result.page.balloons[0].grouping_method == "visual_fallback"
    assert result.page.balloons[0].confidence == 0.82
    assert not result.unassigned_text_region_ids


def test_ctd_region_ids_and_source_boxes_remain_unchanged():
    first = make_region("ctd-region-0001", 20, 20, 55, 35)
    second = make_region("ctd-region-0002", 21, 38, 56, 53)
    original = [(region.id, region.bbox.model_copy(deep=True)) for region in (first, second)]
    result = group(make_page(first, second))

    assert [(region.id, region.bbox) for region in result.page.text_regions] == original
    assert set(result.page.balloons[0].text_region_ids) == {first.id, second.id}
    assert result.page.balloons[0].id not in {first.id, second.id}


def test_panel_and_balloon_ids_are_stable_and_sequence_scoped():
    region = make_region("ctd-region-0001", 10, 10, 40, 25)
    page = make_page(region, page_index=2)

    first = group(page)
    second = group(page)
    other_sequence = PageLayoutGrouper().group_page(
        page,
        "another-sequence",
        page_size=(200, 160),
    )

    assert first.page.panels[0].id == second.page.panels[0].id
    assert first.page.balloons[0].id == second.page.balloons[0].id
    assert first.page.panels[0].id != other_sequence.page.panels[0].id
    assert first.page.balloons[0].id != other_sequence.page.balloons[0].id


def test_empty_ctd_output_returns_panel_context_without_balloons():
    result = group(make_page())

    assert len(result.page.panels) == 1
    assert result.page.balloons == []
    assert result.decisions == []
    assert result.unassigned_text_region_ids == []


def test_malformed_region_bbox_is_preserved_but_unassigned():
    region = make_region("ctd-malformed", 30, 30, 20, 20)
    result = group(make_page(region))

    assert result.page.text_regions[0] == region
    assert result.page.balloons == []
    assert result.unassigned_text_region_ids == [region.id]
    assert result.decisions[0].evidence["reason"] == "malformed_text_region_bbox"


def test_conflicting_explicit_balloon_membership_remains_ambiguous():
    first = make_region("ctd-conflict", 20, 20, 40, 35)
    second = make_region("ctd-neighbor", 20, 38, 40, 53)

    class Balloons:
        def propose_balloons(self, image_path, regions, panels):
            return [
                BalloonProposal(
                    bbox=BoundingBox(x1=15, y1=15, x2=45, y2=40),
                    text_region_ids=[first.id],
                ),
                BalloonProposal(
                    bbox=BoundingBox(x1=15, y1=25, x2=45, y2=55),
                    text_region_ids=[first.id],
                ),
            ]

    result = group(make_page(first, second), PageLayoutGrouper(balloon_provider=Balloons()))

    assert all(first.id not in balloon.text_region_ids for balloon in result.page.balloons)
    first_decisions = [decision for decision in result.decisions if first.id in decision.text_region_ids]
    assert len(first_decisions) == 1
    assert first_decisions[0].status == "ambiguous"