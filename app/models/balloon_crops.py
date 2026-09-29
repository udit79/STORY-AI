"""Persist crops for semantic balloons without changing their source geometry."""

from __future__ import annotations

import math
from pathlib import Path
from urllib.parse import quote

import cv2

from app.schemas.page import Balloon


class BalloonCropStore:
    """Save a deterministic, pixel-clamped crop for one grouped balloon."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def create(
        self,
        page_image_path: str | Path,
        sequence_id: str,
        page_index: int,
        balloon: Balloon,
    ) -> Path:
        if page_index < 0:
            raise ValueError("page_index must be zero-based and non-negative")
        page_image_path = Path(page_image_path)
        image = cv2.imread(str(page_image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read page image: {page_image_path}")

        image_height, image_width = image.shape[:2]
        bbox = balloon.bbox
        coordinates = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("balloon bbox contains non-finite coordinates")
        if bbox.x2 <= bbox.x1 or bbox.y2 <= bbox.y1:
            raise ValueError("balloon bbox must have positive width and height")

        x1 = max(0, min(image_width, math.floor(bbox.x1)))
        y1 = max(0, min(image_height, math.floor(bbox.y1)))
        x2 = max(0, min(image_width, math.ceil(bbox.x2)))
        y2 = max(0, min(image_height, math.ceil(bbox.y2)))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("balloon bbox does not intersect the page image")
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError("balloon bbox produced an empty crop")

        sequence_component = quote(sequence_id, safe="-_.") or "_"
        balloon_component = quote(balloon.id, safe="-_.") or "_"
        path = (
            self.root
            / sequence_component
            / f"page_{page_index:02d}"
            / balloon_component
            / "balloon.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), crop):
            raise OSError(f"could not write balloon crop: {path}")
        return path
