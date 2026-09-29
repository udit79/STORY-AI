"""Persist deterministic, page-coordinate-preserving CTD region crops."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2

from app.schemas.page import TextRegion

CropTransform = Callable[[Any], Any]


@dataclass
class CropGenerationResult:
    crop_paths: dict[str, dict[str, Path]] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)


def _safe_component(value: str) -> str:
    return quote(value, safe="-_.") or "_"


class CTDCropStore:
    """Create exact bbox crops and only the explicitly configured variants."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def generate(
        self,
        image_path: str | Path,
        sequence_id: str,
        page_index: int,
        regions: list[TextRegion],
        *,
        variants: Mapping[str, CropTransform] | None = None,
    ) -> CropGenerationResult:
        if page_index < 0:
            raise ValueError("page_index must be zero-based and non-negative")
        image_path = Path(image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read page image for crops: {image_path}")

        configured_variants = dict(variants or {})
        if "base" in configured_variants:
            raise ValueError("base is reserved for the unmodified localized crop")
        if any(not name.strip() or "/" in name or "\\" in name for name in configured_variants):
            raise ValueError("preprocessing variant names must be simple path components")

        result = CropGenerationResult()
        image_height, image_width = image.shape[:2]
        for region in regions:
            try:
                crop = self._slice_region(image, region, image_width, image_height)
                region_dir = (
                    self.root
                    / _safe_component(sequence_id)
                    / f"page_{page_index:02d}"
                    / _safe_component(region.id)
                )
                region_dir.mkdir(parents=True, exist_ok=True)
                paths = {"base": self._write(region_dir / "base.png", crop)}
                for name, transform in configured_variants.items():
                    variant = transform(crop.copy())
                    if variant is None or getattr(variant, "size", 0) == 0:
                        raise ValueError(f"preprocessing variant {name!r} produced an empty crop")
                    paths[name] = self._write(region_dir / f"{_safe_component(name)}.png", variant)
                result.crop_paths[region.id] = paths
            except Exception as exc:  # noqa: BLE001 - isolate per-region crop failures
                result.failures[region.id] = f"{type(exc).__name__}: {exc}"
        return result

    @staticmethod
    def _slice_region(image: Any, region: TextRegion, width: int, height: int) -> Any:
        bbox = region.bbox
        coordinates = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("region bbox contains non-finite coordinates")
        if bbox.x2 <= bbox.x1 or bbox.y2 <= bbox.y1:
            raise ValueError("region bbox must have positive width and height")

        x1 = max(0, min(width, math.floor(bbox.x1)))
        y1 = max(0, min(height, math.floor(bbox.y1)))
        x2 = max(0, min(width, math.ceil(bbox.x2)))
        y2 = max(0, min(height, math.ceil(bbox.y2)))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("region bbox does not intersect the page image")
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError("region bbox produced an empty crop")
        return crop

    @staticmethod
    def _write(path: Path, image: Any) -> Path:
        if not cv2.imwrite(str(path), image):
            raise OSError(f"could not write crop: {path}")
        return path