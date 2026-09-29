"""Lazy local adapter for the Apache-2.0-declared MangaLens YOLO11 bubble model.

The upstream checkpoint provides only a speech-bubble class, not panels. Its
training card cites MangaSegmentation and Manga109; their separate access and
use conditions remain applicable. This adapter loads only an explicit local
checkpoint and makes no hosted inference calls or implicit downloads.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from app.layout_grouping import BalloonProposal, PanelProposal
from app.schemas.page import BoundingBox, Point2D

MANGALENS_MODEL_ID = "huyvux3005/manga109-segmentation-bubble"
MANGALENS_MODEL_CARD = (
    "https://huggingface.co/huyvux3005/manga109-segmentation-bubble"
)
MANGALENS_LICENSE = "Apache-2.0 (model-card metadata)"
MANGALENS_TRAINING_DATA = ("MS92/MangaSegmentation", "Manga109")
MANGALENS_IMAGE_SIZE = 1600
MANGALENS_BALLOON_CLASSES = frozenset(
    {"balloon", "bubble", "speech_bubble", "speech bubble", "speech-balloon"}
)


def _python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def _class_name(names: Any, class_id: int) -> str | None:
    if isinstance(names, Mapping):
        name = names.get(class_id, names.get(str(class_id)))
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        name = names[class_id] if 0 <= class_id < len(names) else None
    else:
        name = None
    return str(name).strip().lower() if name is not None else None


def _mask_polygon(result: Any, index: int, width: int, height: int) -> list[Point2D] | None:
    masks = getattr(result, "masks", None)
    polygons = getattr(masks, "xy", None)
    if polygons is None or index >= len(polygons):
        return None
    try:
        raw_polygon = polygons[index]
        raw_polygon = (
            raw_polygon.tolist()
            if callable(getattr(raw_polygon, "tolist", None))
            else list(raw_polygon)
        )
        points: list[Point2D] = []
        for raw_point in raw_polygon:
            point = (
                raw_point.tolist()
                if callable(getattr(raw_point, "tolist", None))
                else list(raw_point)
            )
            point_x, point_y = map(float, point[:2])
            if math.isfinite(point_x) and math.isfinite(point_y):
                points.append(
                    Point2D(
                        x=max(0.0, min(float(width), point_x)),
                        y=max(0.0, min(float(height), point_y)),
                    )
                )
        return points if len(points) >= 3 else None
    except (TypeError, ValueError, IndexError, AttributeError):
        return None


class Manga109YoloBalloonProvider:
    """Adapt the local one-class YOLO segmentation checkpoint to layout proposals."""

    def __init__(
        self,
        model: Any,
        *,
        image_size: int = MANGALENS_IMAGE_SIZE,
        confidence_threshold: float = 0.25,
        balloon_classes: frozenset[str] = MANGALENS_BALLOON_CLASSES,
    ) -> None:
        if image_size < 32:
            raise ValueError("image_size must be at least 32 pixels")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.model = model
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.balloon_classes = balloon_classes

    def detect_panels(
        self,
        image_path: Path,
        page_width: int,
        page_height: int,
    ) -> Sequence[PanelProposal]:
        """This checkpoint has no panel class; the grouper uses its page fallback."""
        return ()

    def propose_balloons(
        self,
        image_path: Path,
        page_width: int,
        page_height: int,
    ) -> Sequence[BalloonProposal]:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Manga layout input must be a local image: {path}")
        if page_width <= 0 or page_height <= 0:
            raise ValueError("page dimensions must be positive")

        results = self.model.predict(
            source=str(path),
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            verbose=False,
            retina_masks=True,
        )
        proposals: list[BalloonProposal] = []
        for result in results or []:
            names = getattr(result, "names", getattr(self.model, "names", None))
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            single_class = len(names) == 1 if isinstance(names, (Mapping, Sequence)) else False
            for box_index, box in enumerate(boxes):
                class_id_value = getattr(box, "cls", None)
                confidence_value = getattr(box, "conf", None)
                coordinates_value = getattr(box, "xyxy", None)
                if class_id_value is None or confidence_value is None or coordinates_value is None:
                    continue
                try:
                    class_id = int(_python_scalar(class_id_value[0]))
                    confidence = float(_python_scalar(confidence_value[0]))
                    coordinate_row = coordinates_value[0]
                    coordinates = (
                        coordinate_row.tolist()
                        if callable(getattr(coordinate_row, "tolist", None))
                        else list(coordinate_row)
                    )
                    x1, y1, x2, y2 = map(float, coordinates)
                except (IndexError, TypeError, ValueError, AttributeError):
                    continue

                name = _class_name(names, class_id)
                is_balloon = name in self.balloon_classes or (name is None and single_class)
                if not is_balloon:
                    continue
                if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                    continue
                if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                    continue

                x1, x2 = sorted(
                    (
                        max(0.0, min(float(page_width), x1)),
                        max(0.0, min(float(page_width), x2)),
                    )
                )
                y1, y2 = sorted(
                    (
                        max(0.0, min(float(page_height), y1)),
                        max(0.0, min(float(page_height), y2)),
                    )
                )
                if x2 <= x1 or y2 <= y1:
                    continue
                mask_polygon = _mask_polygon(result, box_index, page_width, page_height)
                proposals.append(
                    BalloonProposal(
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        mask_polygon=mask_polygon,
                        kind="unknown",
                        confidence=confidence,
                        evidence={
                            "provider": MANGALENS_MODEL_ID,
                            "model_class": name or str(class_id),
                            "class_id": class_id,
                        },
                    )
                )
        return proposals


def load_manga109_balloon_provider(
    weights_path: str | Path,
    *,
    image_size: int = MANGALENS_IMAGE_SIZE,
    confidence_threshold: float = 0.25,
) -> Manga109YoloBalloonProvider:
    """Load explicitly supplied local weights; never resolve/download model IDs."""
    local_weights = Path(weights_path).resolve()
    if not local_weights.is_file():
        raise FileNotFoundError(f"Local manga layout weights not found: {local_weights}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional story-ai[manga-layout] dependencies to load YOLO weights"
        ) from exc
    model = YOLO(str(local_weights), task="segment")
    return Manga109YoloBalloonProvider(
        model,
        image_size=image_size,
        confidence_threshold=confidence_threshold,
    )