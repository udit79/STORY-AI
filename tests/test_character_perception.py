from pathlib import Path

import cv2
import numpy as np

from app.character_perception import (
    CharacterPerceptionService,
    CharacterProposal,
)
from app.models.character_crops import CharacterCropStore
from app.schemas.page import (
    Balloon,
    BoundingBox,
    CharacterInstance,
    PageRepresentation,
    Panel,
)


def write_page(path: Path, width=120, height=100):
    image = np.full((height, width, 3), 190, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return image


def test_provider_normalizes_dict_proposals_and_preserves_visual_metadata(tmp_path):
    image_path = tmp_path / "page.png"
    write_page(image_path)

    class Provider:
        def detect(self, page_image_path, panel_bbox=None):
            assert panel_bbox is None
            return [{
                "bbox": {"x1": 20, "y1": 15, "x2": 55, "y2": 80},
                "confidence": 0.77,
                "face_bbox": {"x1": 27, "y1": 18, "x2": 45, "y2": 40},
                "visual_embedding": [0.1, 0.2, 0.3],
                "visual_metadata": {"detector": "mock"},
            }]

    result = CharacterPerceptionService(
        Provider(), CharacterCropStore(tmp_path / "crops")
    ).process_page(
        PageRepresentation(page_index=0, image_path=str(image_path)), "seq-a"
    )
    character = result.page.characters[0]

    assert character.id == "char-seq-a-p01-001"
    assert character.character_instance_id == character.id
    assert character.bbox == BoundingBox(x1=20, y1=15, x2=55, y2=80)
    assert character.face_bbox == BoundingBox(x1=27, y1=18, x2=45, y2=40)
    assert character.visual_embedding == [0.1, 0.2, 0.3]
    assert character.visual_metadata == {"detector": "mock"}


def test_character_ids_are_deterministic_and_independent_of_provider_order(tmp_path):
    image_path = tmp_path / "page.png"
    write_page(image_path)
    boxes = [
        CharacterProposal(bbox=BoundingBox(x1=70, y1=10, x2=90, y2=55)),
        CharacterProposal(bbox=BoundingBox(x1=10, y1=20, x2=35, y2=70)),
    ]

    class Provider:
        def __init__(self, reverse):
            self.reverse = reverse

        def detect(self, page_image_path, panel_bbox=None):
            return list(reversed(boxes)) if self.reverse else boxes

    first = CharacterPerceptionService(
        Provider(False), CharacterCropStore(tmp_path / "crops-a")
    ).process_page(PageRepresentation(page_index=2, image_path=str(image_path)), "seq")
    second = CharacterPerceptionService(
        Provider(True), CharacterCropStore(tmp_path / "crops-b")
    ).process_page(PageRepresentation(page_index=2, image_path=str(image_path)), "seq")

    assert [(char.id, char.bbox) for char in first.page.characters] == [
        (char.id, char.bbox) for char in second.page.characters
    ]
    assert first.page.characters[0].id == "char-seq-p03-001"


def test_panel_association_uses_layout_panel_and_preserves_source_bbox(tmp_path):
    image_path = tmp_path / "page.png"
    write_page(image_path)
    panel = Panel(
        id="panel-seq-p01-0001",
        bbox=BoundingBox(x1=0, y1=0, x2=60, y2=100),
        source="layout_model",
    )
    inside_bbox = BoundingBox(x1=10, y1=10, x2=30, y2=60)
    outside_bbox = BoundingBox(x1=80, y1=10, x2=105, y2=60)

    class Provider:
        def detect(self, page_image_path, panel_bbox=None):
            assert panel_bbox == panel.bbox
            return [CharacterProposal(bbox=inside_bbox), CharacterProposal(bbox=outside_bbox)]

    page = PageRepresentation(page_index=0, image_path=str(image_path), panels=[panel])
    result = CharacterPerceptionService(
        Provider(), CharacterCropStore(tmp_path / "crops")
    ).process_page(page, "seq")

    assert result.page.characters[0].panel_id == panel.id
    assert result.page.characters[0].bbox == inside_bbox
    assert result.page.characters[1].panel_id is None
    assert result.page.characters[1].bbox == outside_bbox


def test_no_panel_context_uses_one_page_level_provider_call(tmp_path):
    image_path = tmp_path / "page.png"
    write_page(image_path)
    seen_panel_contexts = []

    class Provider:
        def detect(self, page_image_path, panel_bbox=None):
            seen_panel_contexts.append(panel_bbox)
            return [CharacterProposal(bbox=BoundingBox(x1=5, y1=5, x2=30, y2=50))]

    page = PageRepresentation(
        page_index=0,
        image_path=str(image_path),
        panels=[Panel(
            id="fallback-panel",
            bbox=BoundingBox(x1=0, y1=0, x2=120, y2=100),
            source="page_fallback",
        )],
    )
    result = CharacterPerceptionService(
        Provider(), CharacterCropStore(tmp_path / "crops")
    ).process_page(page, "seq")

    assert seen_panel_contexts == [None]
    assert len(result.page.characters) == 1
    assert result.page.characters[0].panel_id is None


def test_character_face_body_crops_are_deterministic_and_do_not_change_boxes(tmp_path):
    image_path = tmp_path / "page.png"
    write_page(image_path, width=40, height=30)
    character = CharacterInstance(
        id="char-seq-p01-001",
        bbox=BoundingBox(x1=-2, y1=2, x2=25, y2=35),
        face_bbox=BoundingBox(x1=5, y1=3, x2=15, y2=14),
        body_bbox=BoundingBox(x1=2, y1=12, x2=22, y2=30),
    )
    original_bbox = character.bbox.model_copy(deep=True)
    store = CharacterCropStore(tmp_path / "crops")

    first = store.generate(image_path, "seq", 0, [character])
    second = store.generate(image_path, "seq", 0, [character])

    assert first.crop_paths[character.id] == second.crop_paths[character.id]
    paths = first.crop_paths[character.id]
    assert paths.character.name == "character.png"
    assert paths.face.name == "face.png"
    assert paths.body.name == "body.png"
    assert cv2.imread(str(paths.character)).shape[:2] == (28, 25)
    assert character.bbox == original_bbox


def test_provider_failure_keeps_balloons_and_other_panel_detections(tmp_path):
    image_path = tmp_path / "page.png"
    write_page(image_path)
    left = Panel(
        id="panel-left", bbox=BoundingBox(x1=0, y1=0, x2=55, y2=100), source="layout_model"
    )
    right = Panel(
        id="panel-right", bbox=BoundingBox(x1=60, y1=0, x2=120, y2=100), source="layout_model"
    )
    balloon = Balloon(
        id="balloon-01",
        bbox=BoundingBox(x1=5, y1=5, x2=45, y2=40),
        panel_id=left.id,
        text_region_ids=["ctd-01"],
    )

    class Provider:
        def detect(self, page_image_path, panel_bbox=None):
            if panel_bbox == left.bbox:
                raise RuntimeError("left panel detector failed")
            return [CharacterProposal(bbox=BoundingBox(x1=75, y1=10, x2=100, y2=65))]

    page = PageRepresentation(
        page_index=0,
        image_path=str(image_path),
        panels=[left, right],
        balloons=[balloon],
    )
    result = CharacterPerceptionService(
        Provider(), CharacterCropStore(tmp_path / "crops")
    ).process_page(page, "seq")

    assert result.page.balloons == [balloon]
    assert len(result.page.characters) == 1
    assert result.page.characters[0].panel_id == right.id
    assert result.diagnostics[0].panel_id == left.id


def test_unavailable_character_provider_preserves_existing_page_and_characters(tmp_path):
    image_path = tmp_path / "page.png"
    write_page(image_path)
    character = CharacterInstance(
        id="char-existing",
        bbox=BoundingBox(x1=2, y1=2, x2=20, y2=30),
    )
    balloon = Balloon(
        id="balloon-existing",
        bbox=BoundingBox(x1=30, y1=20, x2=50, y2=45),
    )
    page = PageRepresentation(
        page_index=0,
        image_path=str(image_path),
        characters=[character],
        balloons=[balloon],
    )

    result = CharacterPerceptionService(
        None, CharacterCropStore(tmp_path / "crops")
    ).process_page(page, "seq")

    assert result.page.characters == [character]
    assert result.page.balloons == [balloon]
    assert result.crop_paths[character.id].character.is_file()
    assert result.diagnostics[0].code == "provider_unavailable"


def test_face_crop_failure_does_not_discard_character_crop(tmp_path):
    image_path = tmp_path / "page.png"
    write_page(image_path, width=40, height=30)
    character = CharacterInstance(
        id="char-seq-p01-001",
        bbox=BoundingBox(x1=2, y1=2, x2=20, y2=28),
        face_bbox=BoundingBox(x1=80, y1=80, x2=90, y2=90),
    )

    result = CharacterCropStore(tmp_path / "crops").generate(
        image_path, "seq", 0, [character]
    )

    assert character.id in result.crop_paths
    assert result.crop_paths[character.id].character.is_file()
    assert result.failures[character.id]["face"]