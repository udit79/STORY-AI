from pathlib import Path

import cv2
import numpy as np

from app.models.candidate_adapters import (
    NemotronCandidateAdapter,
    PaddleOCRCandidateAdapter,
    Qwen3VLCandidateAdapter,
)
from app.models.crops import CTDCropStore
from app.models.ctd import (
    CTDPageLocalizer,
    ctd_block_bbox,
    suppress_duplicate_ctd_blocks,
)
from app.page_perception import PagePerceptionService
from app.schemas.page import BoundingBox, TextRegion


class FakeCTDBlock:
    def __init__(self, x: int, y: int, width: int, height: int, **attributes):
        self._rect = [x, y, width, height]
        self.__dict__.update(attributes)

    def bounding_rect(self):
        return self._rect


def write_test_image(path: Path, width: int = 32, height: int = 24) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(width, dtype=np.uint8)[None, :]
    assert cv2.imwrite(str(path), image)
    return image


def test_ctd_bbox_conversion_preserves_page_coordinates():
    bbox = ctd_block_bbox(FakeCTDBlock(3, 5, 12, 8))

    assert bbox == BoundingBox(x1=3, y1=5, x2=15, y2=13)


def test_ctd_duplicate_suppression_preserves_original_detector_indices():
    blocks = [
        FakeCTDBlock(0, 0, 20, 20),
        FakeCTDBlock(0, 0, 20, 20),
        FakeCTDBlock(40, 10, 12, 8),
    ]

    retained = suppress_duplicate_ctd_blocks(blocks, iou_threshold=0.85)

    assert [source_index for source_index, _ in retained] == [0, 2]


def test_localizer_keeps_language_unfiltered_and_missing_confidence_unset(tmp_path):
    image_path = tmp_path / "page.png"
    image = write_test_image(image_path)
    detector_inputs = []
    block = FakeCTDBlock(2, 3, 10, 8, language="ja", prob=1.0)

    def detector(input_image):
        detector_inputs.append(input_image.shape)
        return None, None, [block]

    regions = CTDPageLocalizer(detector).localize(image_path)

    assert detector_inputs == [image.shape]
    assert len(regions) == 1
    assert regions[0].id == "ctd-region-0000"
    assert regions[0].bbox == BoundingBox(x1=2, y1=3, x2=12, y2=11)
    assert regions[0].confidence is None
    assert regions[0].category == "unknown"


def test_crop_clamps_partial_out_of_bounds_bbox_without_mutating_region(tmp_path):
    image_path = tmp_path / "page.png"
    write_test_image(image_path, width=8, height=6)
    region = TextRegion(
        id="region:one",
        bbox=BoundingBox(x1=-2.2, y1=1.0, x2=4.2, y2=9.0),
        raw_text="",
    )

    result = CTDCropStore(tmp_path / "artifacts").generate(
        image_path, "sequence-a", 0, [region]
    )
    crop = cv2.imread(str(result.crop_paths[region.id]["base"]))

    assert crop.shape[:2] == (5, 5)
    assert region.bbox == BoundingBox(x1=-2.2, y1=1.0, x2=4.2, y2=9.0)
    assert not result.failures


def test_crop_names_are_deterministic_and_id_is_path_safe(tmp_path):
    image_path = tmp_path / "page.png"
    write_test_image(image_path)
    region = TextRegion(
        id="ctd:4/part",
        bbox=BoundingBox(x1=1, y1=2, x2=7, y2=9),
        raw_text="",
    )
    store = CTDCropStore(tmp_path / "artifacts")

    first = store.generate(image_path, "seq/alpha", 2, [region])
    second = store.generate(image_path, "seq/alpha", 2, [region])
    expected = (
        tmp_path / "artifacts" / "seq%2Falpha" / "page_02"
        / "ctd%3A4%2Fpart" / "base.png"
    )

    assert first.crop_paths[region.id]["base"] == expected
    assert second.crop_paths[region.id]["base"] == expected
    assert expected.is_file()


def test_preprocessing_crops_only_generated_when_configured(tmp_path):
    image_path = tmp_path / "page.png"
    write_test_image(image_path)
    region = TextRegion(
        id="region-1",
        bbox=BoundingBox(x1=1, y1=2, x2=8, y2=10),
        raw_text="",
    )
    store = CTDCropStore(tmp_path / "artifacts")

    base_only = store.generate(image_path, "seq", 0, [region])
    with_variant = store.generate(
        image_path,
        "seq",
        0,
        [region],
        variants={"upscale": lambda crop: cv2.resize(crop, None, fx=2, fy=2)},
    )

    assert set(base_only.crop_paths[region.id]) == {"base"}
    assert set(with_variant.crop_paths[region.id]) == {"base", "upscale"}
    assert with_variant.crop_paths[region.id]["upscale"].name == "upscale.png"


def test_page_service_routes_preprocessing_adapter_to_matching_crop(tmp_path):
    image_path = tmp_path / "page.png"
    write_test_image(image_path)
    region = TextRegion(
        id="region-1",
        bbox=BoundingBox(x1=1, y1=2, x2=8, y2=10),
        raw_text="",
    )
    received_paths = []

    class Localizer:
        def localize(self, path):
            return [region]

    class Recognizer:
        def predict(self, *, input: str, batch_size: int):
            received_paths.append(Path(input))
            return [{"rec_text": "upscaled evidence", "rec_score": 0.8}]

    service = PagePerceptionService(
        Localizer(),
        CTDCropStore(tmp_path / "artifacts"),
        {"paddle_upscale": PaddleOCRCandidateAdapter(
            Recognizer(), preprocessing="upscale"
        )},
        crop_variants={
            "upscale": lambda crop: cv2.resize(crop, None, fx=2, fy=2)
        },
    )
    result = service.process_page(image_path, "seq", 0)

    assert received_paths == [
        tmp_path / "artifacts" / "seq" / "page_00" / "region-1" / "upscale.png"
    ]
    assert result.candidate_bank.groups[0].candidates[0].source == "ocr_upscale"


def test_crop_reports_invalid_and_non_intersecting_boxes(tmp_path):
    image_path = tmp_path / "page.png"
    write_test_image(image_path)
    regions = [
        TextRegion(
            id="outside",
            bbox=BoundingBox(x1=100, y1=100, x2=110, y2=110),
            raw_text="",
        ),
        TextRegion(
            id="inverted",
            bbox=BoundingBox(x1=5, y1=5, x2=2, y2=2),
            raw_text="",
        ),
    ]

    result = CTDCropStore(tmp_path / "artifacts").generate(image_path, "seq", 0, regions)

    assert set(result.failures) == {"outside", "inverted"}
    assert not result.crop_paths


def test_page_service_isolates_a_failed_producer_and_keeps_other_evidence(tmp_path):
    image_path = tmp_path / "page.png"
    write_test_image(image_path)
    region = TextRegion(
        id="ctd-region-0000",
        bbox=BoundingBox(x1=1, y1=2, x2=10, y2=12),
        raw_text="",
    )

    class Localizer:
        def localize(self, path):
            return [region]

    class Recognizer:
        def predict(self, **kwargs):
            return [{"rec_text": "Paddle evidence", "rec_score": 0.8}]

    service = PagePerceptionService(
        Localizer(),
        CTDCropStore(tmp_path / "artifacts"),
        {
            "paddle": PaddleOCRCandidateAdapter(Recognizer()),
            "nemotron": None,
            "qwen3_vl": Qwen3VLCandidateAdapter(
                lambda path: (_ for _ in ()).throw(RuntimeError("local Qwen unavailable"))
            ),
        },
        backend_errors={"nemotron": "WSL bridge is not configured"},
    )

    result = service.process_page(image_path, "seq", 0)

    assert result.page.text_regions == [region]
    assert len(result.candidate_bank.groups[0].candidates) == 1
    assert result.candidate_bank.groups[0].candidates[0].source == "ocr_base"
    assert {(item.component, item.status) for item in result.diagnostics} == {
        ("nemotron", "unavailable"),
        ("qwen3_vl", "failed"),
    }


def test_zero_region_page_returns_empty_candidate_bank(tmp_path):
    image_path = tmp_path / "page.png"
    write_test_image(image_path)

    class Localizer:
        def localize(self, path):
            return []

    result = PagePerceptionService(
        Localizer(), CTDCropStore(tmp_path / "artifacts"), {}
    ).process_page(image_path, "seq", 1)

    assert result.page.page_index == 1
    assert result.page.text_regions == []
    assert result.candidate_bank.groups == []
    assert result.diagnostics == []


def test_end_to_end_mocked_page_to_three_source_candidate_bank(tmp_path):
    image_path = tmp_path / "page.png"
    write_test_image(image_path)
    blocks = [
        FakeCTDBlock(2, 3, 12, 8),
        FakeCTDBlock(30, 4, 10, 8),
    ]

    class Detector:
        def __call__(self, image):
            return None, None, blocks

    localizer = CTDPageLocalizer(Detector())

    class Recognizer:
        def predict(self, *, input: str, batch_size: int):
            assert Path(input).is_file()
            return [{"rec_text": "Paddle", "rec_score": 0.8}]

    def nemotron(path: str, *, merge_level: str):
        assert Path(path).is_file()
        return [{"text": "Nemotron", "confidence": 0.7}]

    def qwen(path: Path):
        assert path.is_file()
        return {
            "transcription": "Qwen",
            "visual_confidence": 0.6,
            "candidate_supported": True,
        }

    service = PagePerceptionService(
        localizer,
        CTDCropStore(tmp_path / "artifacts"),
        {
            "paddle": PaddleOCRCandidateAdapter(Recognizer()),
            "nemotron": NemotronCandidateAdapter(nemotron),
            "qwen3_vl": Qwen3VLCandidateAdapter(qwen),
        },
    )
    result = service.process_page(image_path, "seq-dev", 2)

    assert len(result.page.text_regions) == 2
    assert len(result.candidate_bank.groups) == 2
    assert all(len(group.candidates) == 3 for group in result.candidate_bank.groups)
    assert all(
        {candidate.source for candidate in group.candidates}
        == {"ocr_base", "nemotron", "qwen3_vl"}
        for group in result.candidate_bank.groups
    )
    assert not result.diagnostics