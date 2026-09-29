import sys
import types
from types import SimpleNamespace

import pytest

from app.layout_grouping import BalloonProposal, PageLayoutGrouper
from app.models.manga_layout import (
    MANGALENS_IMAGE_SIZE,
    Manga109YoloBalloonProvider,
    load_manga109_balloon_provider,
)
from app.schemas.page import BoundingBox, PageRepresentation, TextRegion


def fake_box(class_id=0, confidence=0.91, bbox=(25.0, 20.0, 115.0, 90.0)):
    return SimpleNamespace(cls=[class_id], conf=[confidence], xyxy=[list(bbox)])


def fake_result(boxes=None, names=None):
    return SimpleNamespace(
        names=names or {0: "speech_bubble"},
        boxes=list(boxes if boxes is not None else [fake_box()]),
    )


class FakeModel:
    def __init__(self, result=None):
        self.names = {0: "speech_bubble"}
        self.result = fake_result() if result is None else result
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [self.result]


def make_region(region_id: str, bbox: BoundingBox) -> TextRegion:
    return TextRegion(id=region_id, bbox=bbox, raw_text="")


def test_provider_normalizes_balloon_bbox_confidence_and_class(tmp_path):
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"mock page")
    model = FakeModel()
    provider = Manga109YoloBalloonProvider(model)

    proposals = provider.propose_balloons(image_path, 200, 100)

    assert proposals == [
        BalloonProposal(
            bbox=BoundingBox(x1=25, y1=20, x2=115, y2=90),
            confidence=0.91,
            kind="unknown",
            evidence={
                "provider": "huyvux3005/manga109-segmentation-bubble",
                "model_class": "speech_bubble",
                "class_id": 0,
            },
        )
    ]
    assert model.calls == [{
        "source": str(image_path),
        "imgsz": MANGALENS_IMAGE_SIZE,
        "conf": 0.25,
        "verbose": False,
        "retina_masks": True,
    }]


def test_provider_clips_model_boxes_to_page_coordinates(tmp_path):
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"mock page")

    result = fake_result(
        [fake_box(confidence=0.6, bbox=(-10, -5, 250, 140))],
        {0: "balloon"},
    )
    proposals = Manga109YoloBalloonProvider(FakeModel(result)).propose_balloons(
        image_path, 200, 100
    )

    assert proposals[0].bbox == BoundingBox(x1=0, y1=0, x2=200, y2=100)


def test_provider_ignores_non_balloon_classes_and_has_no_panel_class(tmp_path):
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"mock page")

    result = fake_result(
        [fake_box(class_id=1)],
        {0: "speech_bubble", 1: "text"},
    )
    provider = Manga109YoloBalloonProvider(FakeModel(result))

    assert provider.propose_balloons(image_path, 200, 100) == []
    assert provider.detect_panels(image_path, 200, 100) == ()


def test_provider_preserves_segmentation_mask_polygon_in_page_coordinates(tmp_path):
    import numpy as np

    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"mock page")
    result = fake_result()
    result.masks = types.SimpleNamespace(
        xy=[np.asarray([[25, 20], [115, 20], [115, 90], [25, 90]], dtype=float)]
    )
    provider = Manga109YoloBalloonProvider(FakeModel(result))

    proposal = provider.propose_balloons(image_path, 200, 100)[0]

    assert proposal.mask_polygon is not None
    assert [(point.x, point.y) for point in proposal.mask_polygon] == [
        (25, 20), (115, 20), (115, 90), (25, 90)
    ]


def test_local_loader_requires_explicit_local_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="Local manga layout weights"):
        load_manga109_balloon_provider(tmp_path / "missing.pt")


def test_local_loader_initializes_ultralytics_lazily_from_local_weights(
    tmp_path, monkeypatch
):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"mock weights")
    loaded = []

    class FakeYOLO:
        def __init__(self, path, *, task):
            loaded.append((path, task))

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    provider = load_manga109_balloon_provider(weights)

    assert isinstance(provider, Manga109YoloBalloonProvider)
    assert loaded == [(str(weights.resolve()), "segment")]


def test_one_balloon_proposal_groups_multiple_ctd_fragments(tmp_path):
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"mock page")
    regions = [
        make_region("ctd-01", BoundingBox(x1=35, y1=30, x2=60, y2=45)),
        make_region("ctd-02", BoundingBox(x1=65, y1=47, x2=92, y2=61)),
        make_region("ctd-03", BoundingBox(x1=40, y1=64, x2=70, y2=80)),
    ]
    page = PageRepresentation(page_index=0, image_path=str(image_path), text_regions=regions)
    provider = Manga109YoloBalloonProvider(FakeModel())

    result = PageLayoutGrouper(
        panel_provider=provider,
        balloon_provider=provider,
    ).group_page(page, "seq-layout", page_size=(200, 100))

    assert result.panel_proposal_count == 0
    assert result.balloon_proposal_count == 1
    assert len(result.page.panels) == 1
    assert result.page.panels[0].source == "page_fallback"
    assert len(result.page.balloons) == 1
    assert result.page.balloons[0].text_region_ids == ["ctd-01", "ctd-02", "ctd-03"]
    assert result.page.balloons[0].id.startswith("balloon-layout-seq-layout")
    assert result.page.text_regions == regions


def test_partial_ctd_balloon_overlap_assigns_when_overlap_is_unambiguous(tmp_path):
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"mock page")
    region = make_region("ctd-partial", BoundingBox(x1=25, y1=20, x2=45, y2=40))
    page = PageRepresentation(page_index=0, image_path=str(image_path), text_regions=[region])
    provider = Manga109YoloBalloonProvider(FakeModel())

    result = PageLayoutGrouper(balloon_provider=provider).group_page(
        page, "seq-layout", page_size=(200, 100)
    )

    assert result.page.balloons[0].text_region_ids == [region.id]


def test_successful_provider_leaves_nonoverlapping_ctd_region_unassigned(tmp_path):
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"mock page")
    region = make_region("ctd-outside", BoundingBox(x1=150, y1=10, x2=180, y2=30))
    page = PageRepresentation(page_index=0, image_path=str(image_path), text_regions=[region])
    provider = Manga109YoloBalloonProvider(FakeModel())

    result = PageLayoutGrouper(balloon_provider=provider).group_page(
        page, "seq-layout", page_size=(200, 100)
    )

    assert result.page.balloons == []
    assert result.unassigned_text_region_ids == [region.id]
    assert result.decisions[0].evidence["reason"] == "no_balloon_proposal_overlap"


def test_provider_failure_falls_back_and_records_diagnostic(tmp_path):
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"mock page")
    region = make_region("ctd-fallback", BoundingBox(x1=20, y1=20, x2=45, y2=40))

    class BrokenProvider:
        def propose_balloons(self, image_path, page_width, page_height):
            raise RuntimeError("local inference failed")

    result = PageLayoutGrouper(balloon_provider=BrokenProvider()).group_page(
        PageRepresentation(page_index=0, image_path=str(image_path), text_regions=[region]),
        "seq-layout",
        page_size=(200, 100),
    )

    assert len(result.page.balloons) == 1
    assert result.page.balloons[0].text_region_ids == [region.id]
    assert result.diagnostics[0].component == "balloon_layout_provider"


def test_provider_ambiguity_is_preserved_as_candidate_balloon_ids(tmp_path):
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"mock page")
    region = make_region("ctd-competing", BoundingBox(x1=40, y1=20, x2=60, y2=40))
    page = PageRepresentation(page_index=0, image_path=str(image_path), text_regions=[region])

    class CompetingModel(FakeModel):
        def predict(self, **kwargs):
            return [
                fake_result(
                    [
                        fake_box(confidence=0.91, bbox=(30, 10, 60, 50)),
                        fake_box(confidence=0.88, bbox=(40, 10, 70, 50)),
                    ]
                )
            ]

    result = PageLayoutGrouper(
        balloon_provider=Manga109YoloBalloonProvider(CompetingModel())
    ).group_page(page, "seq-layout", page_size=(200, 100))

    decision = result.decisions[0]
    assert decision.status == "ambiguous"
    assert len(decision.candidate_balloon_ids) == 2
    assert result.page.balloons == []