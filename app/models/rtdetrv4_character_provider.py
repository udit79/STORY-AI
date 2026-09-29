"""Local ONNX Runtime adapter for RT-DETRv4-X Manga109-s v2 character detections.

The checkpoint is model-only; Manga109-s data is neither fetched nor bundled.
The model card documents fixed 1280-square RGB input normalized to [0, 1],
`orig_target_sizes` as [width, height], and output boxes in original-image xyxy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.character_perception import CharacterDetection
from app.schemas.page import BoundingBox

MODEL_ID = "tori29umai/rtdetrv4-x-manga109s_v2"
MODEL_CARD_URL = "https://huggingface.co/tori29umai/rtdetrv4-x-manga109s_v2"
MODEL_LICENSE = "Apache-2.0"
MODEL_INPUT_SIZE = 1280
CLASS_NAMES = {0: "body", 1: "text", 2: "frame", 3: "face"}


class RTDETRv4CharacterProvider:
    """Run an injected ONNX session; retain only body and face detections."""

    def __init__(
        self,
        session: Any,
        *,
        confidence_threshold: float = 0.5,
        input_size: int = MODEL_INPUT_SIZE,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if input_size != MODEL_INPUT_SIZE:
            raise ValueError("RT-DETRv4 checkpoint requires 1280x1280 input")
        input_names = {item.name for item in session.get_inputs()}
        output_names = {item.name for item in session.get_outputs()}
        if not {"images", "orig_target_sizes"}.issubset(input_names):
            raise ValueError(f"unexpected RT-DETRv4 ONNX inputs: {sorted(input_names)}")
        if not {"labels", "boxes", "scores"}.issubset(output_names):
            raise ValueError(f"unexpected RT-DETRv4 ONNX outputs: {sorted(output_names)}")
        self.session = session
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size

    def detect(
        self,
        page_image_path: Path,
        panel_bbox: BoundingBox | None = None,
    ) -> list[CharacterDetection]:
        image_path = Path(page_image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read character detection image: {image_path}")
        page_height, page_width = image.shape[:2]

        offset_x = 0
        offset_y = 0
        if panel_bbox is not None:
            if not self._valid_bbox(panel_bbox):
                raise ValueError("panel bbox is malformed")
            offset_x = max(0, math.floor(panel_bbox.x1))
            offset_y = max(0, math.floor(panel_bbox.y1))
            end_x = min(page_width, math.ceil(panel_bbox.x2))
            end_y = min(page_height, math.ceil(panel_bbox.y2))
            offset_x = min(page_width, offset_x)
            offset_y = min(page_height, offset_y)
            if end_x <= offset_x or end_y <= offset_y:
                return []
            image = image[offset_y:end_y, offset_x:end_x]

        crop_height, crop_width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        images = np.ascontiguousarray(
            (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        )
        original_size = np.asarray([[crop_width, crop_height]], dtype=np.int64)
        labels, boxes, scores = self.session.run(
            ["labels", "boxes", "scores"],
            {"images": images, "orig_target_sizes": original_size},
        )
        labels = np.asarray(labels)
        boxes = np.asarray(boxes)
        scores = np.asarray(scores)
        if labels.ndim == 2:
            labels = labels[0]
        if boxes.ndim == 3:
            boxes = boxes[0]
        if scores.ndim == 2:
            scores = scores[0]
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError(f"unexpected RT-DETRv4 boxes shape: {boxes.shape}")
        if len(labels) != len(boxes) or len(scores) != len(boxes):
            raise ValueError("RT-DETRv4 labels, boxes, and scores have different lengths")

        detections: list[CharacterDetection] = []
        for detection_index, (raw_label, raw_box, raw_score) in enumerate(
            zip(labels, boxes, scores, strict=True)
        ):
            class_id = int(raw_label)
            category = CLASS_NAMES.get(class_id)
            if category not in {"body", "face"}:
                continue
            confidence = float(raw_score)
            if (
                not math.isfinite(confidence)
                or confidence < self.confidence_threshold
                or confidence > 1.0
            ):
                continue
            x1, y1, x2, y2 = map(float, raw_box)
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                continue
            x1, x2 = sorted((x1 + offset_x, x2 + offset_x))
            y1, y2 = sorted((y1 + offset_y, y2 + offset_y))
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            if not self._valid_bbox(bbox):
                continue
            detection_id = (
                f"{category}-{detection_index:03d}-"
                f"x{round(x1)}-y{round(y1)}"
            )
            detections.append(
                CharacterDetection(
                    detection_id=detection_id,
                    category=category,
                    bbox=bbox,
                    confidence=confidence,
                    visual_metadata={
                        "model_id": MODEL_ID,
                        "class_id": class_id,
                        "coordinate_frame": "page",
                    },
                )
            )
        return detections

    @staticmethod
    def _valid_bbox(bbox: BoundingBox) -> bool:
        return (
            all(math.isfinite(value) for value in (bbox.x1, bbox.y1, bbox.x2, bbox.y2))
            and bbox.x2 > bbox.x1
            and bbox.y2 > bbox.y1
        )


def load_rtdetrv4_character_provider(
    model_path: str | Path,
    *,
    providers: Sequence[str] | None = None,
    confidence_threshold: float = 0.5,
) -> RTDETRv4CharacterProvider:
    """Initialize the local ONNX session explicitly without downloading weights."""
    local_model = Path(model_path).resolve()
    if not local_model.is_file():
        raise FileNotFoundError(f"RT-DETRv4 ONNX model not found: {local_model}")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("ONNX Runtime is not installed") from exc

    available = set(ort.get_available_providers())
    if providers is None:
        selected_providers = [
            provider
            for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if provider in available
        ]
    else:
        selected_providers = list(providers)
        unavailable = set(selected_providers) - available
        if unavailable:
            raise RuntimeError(f"Requested ONNX Runtime providers unavailable: {sorted(unavailable)}")
    if not selected_providers:
        raise RuntimeError("No usable ONNX Runtime execution provider is available")

    session = ort.InferenceSession(str(local_model), providers=selected_providers)
    return RTDETRv4CharacterProvider(
        session,
        confidence_threshold=confidence_threshold,
    )
