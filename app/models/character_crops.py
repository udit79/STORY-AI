"""Deterministic page-coordinate character, face, and body crop persistence."""

from __future__ import annotations

import math
from pathlib import Path
from urllib.parse import quote

import cv2
from pydantic import BaseModel, Field

from app.schemas.page import BoundingBox, CharacterInstance


class CharacterCropPaths(BaseModel):
    character: Path
    face: Path | None = None
    face_crops: list[Path] = Field(default_factory=list)
    body: Path | None = None


class CharacterCropResult(BaseModel):
    crop_paths: dict[str, CharacterCropPaths]
    failures: dict[str, dict[str, str]]


class CharacterCropStore:
    """Write available crops while leaving CharacterInstance bboxes untouched."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def generate(
        self,
        page_image_path: str | Path,
        sequence_id: str,
        page_index: int,
        characters: list[CharacterInstance],
    ) -> CharacterCropResult:
        if page_index < 0:
            raise ValueError("page_index must be zero-based and non-negative")
        page_image_path = Path(page_image_path)
        image = cv2.imread(str(page_image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read page image: {page_image_path}")
        image_height, image_width = image.shape[:2]
        sequence_component = quote(sequence_id, safe="-_.") or "_"

        crop_paths: dict[str, CharacterCropPaths] = {}
        failures: dict[str, dict[str, str]] = {}
        for character in characters:
            character_component = quote(character.id, safe="-_.") or "_"
            character_dir = (
                self.root
                / sequence_component
                / f"page_{page_index:02d}"
                / character_component
            )
            paths: dict[str, Path | None] = {"face": None, "body": None}
            face_paths: list[Path] = []
            errors: dict[str, str] = {}
            crop_targets: list[tuple[str, BoundingBox]] = [("character", character.bbox)]
            face_boxes = character.face_bboxes or (
                [character.face_bbox] if character.face_bbox is not None else []
            )
            deduplicated_faces: list[BoundingBox] = []
            for face_box in face_boxes:
                if face_box not in deduplicated_faces:
                    deduplicated_faces.append(face_box)
            crop_targets.extend(
                (f"face_{index:02d}", face_box)
                for index, face_box in enumerate(deduplicated_faces, start=1)
            )
            if character.body_bbox is not None:
                crop_targets.append(("body", character.body_bbox))
            for crop_kind, bbox in crop_targets:
                if bbox is None:
                    continue
                try:
                    crop = self._crop(image, bbox, image_width, image_height)
                    crop_filename = (
                        "face.png"
                        if crop_kind == "face_01"
                        else f"{crop_kind}.png"
                    )
                    crop_path = character_dir / crop_filename
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(crop_path), crop):
                        raise OSError(f"could not write {crop_kind} crop: {crop_path}")
                    if crop_kind.startswith("face_"):
                        face_paths.append(crop_path)
                        if paths["face"] is None:
                            paths["face"] = crop_path
                    else:
                        paths[crop_kind] = crop_path
                except Exception as exc:  # noqa: BLE001 - isolate component crop failures
                    error_key = "face" if crop_kind.startswith("face_") else crop_kind
                    errors[error_key] = f"{type(exc).__name__}: {exc}"
            if paths["character"] is None:
                failures[character.id] = errors
                continue
            crop_paths[character.id] = CharacterCropPaths(
                character=paths["character"],
                face=paths["face"],
                face_crops=face_paths,
                body=paths["body"],
            )
            if errors:
                failures[character.id] = errors
        return CharacterCropResult(crop_paths=crop_paths, failures=failures)

    @staticmethod
    def _crop(image, bbox: BoundingBox, width: int, height: int):
        values = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("character bbox contains non-finite coordinates")
        if bbox.x2 <= bbox.x1 or bbox.y2 <= bbox.y1:
            raise ValueError("character bbox must have positive width and height")
        x1 = max(0, min(width, math.floor(bbox.x1)))
        y1 = max(0, min(height, math.floor(bbox.y1)))
        x2 = max(0, min(width, math.ceil(bbox.x2)))
        y2 = max(0, min(height, math.ceil(bbox.y2)))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("character bbox does not intersect the page image")
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError("character bbox produced an empty crop")
        return crop
