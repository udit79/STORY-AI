"""Page-scoped character instance perception with an injectable local provider."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import quote

import cv2
from pydantic import BaseModel, Field

from app.models.character_crops import CharacterCropPaths, CharacterCropStore
from app.schemas.page import (
    BoundingBox,
    CharacterInstance,
    PageRepresentation,
    Panel,
)


class CharacterProposal(BaseModel):
    """Page-coordinate character detection; it is not a cross-page identity."""

    bbox: BoundingBox
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    face_bbox: BoundingBox | None = None
    face_bboxes: list[BoundingBox] = Field(default_factory=list)
    face_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    body_bbox: BoundingBox | None = None
    visual_embedding: list[float] | None = None
    visual_metadata: dict[str, Any] = Field(default_factory=dict)


class CharacterDetection(BaseModel):
    detection_id: str
    category: Literal["body", "face"]
    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    visual_metadata: dict[str, Any] = Field(default_factory=dict)


class CharacterProvider(Protocol):
    def detect(
        self,
        page_image_path: Path,
        panel_bbox: BoundingBox | None = None,
    ) -> Sequence[CharacterDetection | CharacterProposal]: ...


class CharacterPerceptionDiagnostic(BaseModel):
    component: str
    code: str
    message: str
    panel_id: str | None = None
    character_instance_id: str | None = None


class CharacterPerceptionResult(BaseModel):
    page: PageRepresentation
    crop_paths: dict[str, CharacterCropPaths] = Field(default_factory=dict)
    diagnostics: list[CharacterPerceptionDiagnostic] = Field(default_factory=list)


def _valid_bbox(bbox: BoundingBox) -> bool:
    values = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
    return (
        all(math.isfinite(value) for value in values)
        and bbox.x2 > bbox.x1
        and bbox.y2 > bbox.y1
    )


def _area(bbox: BoundingBox) -> float:
    return max(0.0, bbox.x2 - bbox.x1) * max(0.0, bbox.y2 - bbox.y1)


def _intersection_area(first: BoundingBox, second: BoundingBox) -> float:
    width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    return width * height


def _iou(first: BoundingBox, second: BoundingBox) -> float:
    intersection = _intersection_area(first, second)
    union = _area(first) + _area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _as_proposal(raw: Any) -> CharacterProposal:
    return raw if isinstance(raw, CharacterProposal) else CharacterProposal.model_validate(raw)


def _as_detection(raw: Any) -> CharacterDetection:
    return raw if isinstance(raw, CharacterDetection) else CharacterDetection.model_validate(raw)


def _stable_character_id(sequence_id: str, page_index: int, ordinal: int) -> str:
    safe_sequence = quote(sequence_id, safe="-_.") or "_"
    return f"char-{safe_sequence}-p{page_index + 1:02d}-{ordinal:03d}"


class CharacterPerceptionService:
    """Run an injected detector per real panel, or once with page-level context."""

    def __init__(
        self,
        provider: CharacterProvider | None,
        crop_store: CharacterCropStore,
        *,
        duplicate_iou_threshold: float = 0.9,
    ) -> None:
        if not 0.0 <= duplicate_iou_threshold <= 1.0:
            raise ValueError("duplicate_iou_threshold must be between 0 and 1")
        self.provider = provider
        self.crop_store = crop_store
        self.duplicate_iou_threshold = duplicate_iou_threshold

    def process_page(
        self,
        page: PageRepresentation,
        sequence_id: str,
        page_image_path: str | Path | None = None,
    ) -> CharacterPerceptionResult:
        image_path = Path(page_image_path or page.image_path)
        diagnostics: list[CharacterPerceptionDiagnostic] = []
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            diagnostics.append(
                CharacterPerceptionDiagnostic(
                    component="character_image",
                    code="image_unreadable",
                    message=f"Could not read page image: {image_path}",
                )
            )
            return CharacterPerceptionResult(page=page, diagnostics=diagnostics)
        if self.provider is None:
            diagnostics.append(
                CharacterPerceptionDiagnostic(
                    component="character_provider",
                    code="provider_unavailable",
                    message="No suitable licensed local character detector is configured.",
                )
            )
            crop_result = self.crop_store.generate(
                image_path, sequence_id, page.page_index, page.characters
            )
            self._add_crop_diagnostics(crop_result.failures, diagnostics)
            return CharacterPerceptionResult(
                page=page,
                crop_paths=crop_result.crop_paths,
                diagnostics=diagnostics,
            )

        detected_panels = [
            panel for panel in page.panels
            if panel.source == "layout_model" and _valid_bbox(panel.bbox)
        ]
        panel_targets: list[Panel | None] = detected_panels or [None]
        proposals: list[tuple[CharacterProposal, str | None]] = []
        detections: list[tuple[CharacterDetection, str | None]] = []
        for panel in panel_targets:
            try:
                raw_proposals = self.provider.detect(
                    image_path,
                    panel.bbox if panel is not None else None,
                )
            except Exception as exc:  # noqa: BLE001 - preserve other panels and page content
                diagnostics.append(
                    CharacterPerceptionDiagnostic(
                        component="character_provider",
                        code="provider_failure",
                        message=f"{type(exc).__name__}: {exc}",
                        panel_id=panel.id if panel is not None else None,
                    )
                )
                continue

            for raw_proposal in raw_proposals:
                if isinstance(raw_proposal, CharacterDetection) or (
                    isinstance(raw_proposal, Mapping) and "category" in raw_proposal
                ):
                    try:
                        detection = _as_detection(raw_proposal)
                    except Exception as exc:  # noqa: BLE001 - reject malformed model outputs individually
                        diagnostics.append(
                            CharacterPerceptionDiagnostic(
                                component="character_provider",
                                code="invalid_detection",
                                message=f"{type(exc).__name__}: {exc}",
                                panel_id=panel.id if panel is not None else None,
                            )
                        )
                        continue
                    if not _valid_bbox(detection.bbox):
                        diagnostics.append(
                            CharacterPerceptionDiagnostic(
                                component="character_provider",
                                code="invalid_bbox",
                                message="Discarded character detection with malformed bbox.",
                                panel_id=panel.id if panel is not None else None,
                            )
                        )
                        continue
                    detections.append((
                        detection,
                        self._associate_panel(detection.bbox, page.panels),
                    ))
                    continue
                try:
                    proposal = _as_proposal(raw_proposal)
                except Exception as exc:  # noqa: BLE001 - reject malformed detections individually
                    diagnostics.append(
                        CharacterPerceptionDiagnostic(
                            component="character_provider",
                            code="invalid_proposal",
                            message=f"{type(exc).__name__}: {exc}",
                            panel_id=panel.id if panel is not None else None,
                        )
                    )
                    continue
                if not _valid_bbox(proposal.bbox):
                    diagnostics.append(
                        CharacterPerceptionDiagnostic(
                            component="character_provider",
                            code="invalid_bbox",
                            message="Discarded character proposal with malformed bbox.",
                            panel_id=panel.id if panel is not None else None,
                        )
                    )
                    continue
                associated_panel_id = self._associate_panel(
                    proposal.bbox,
                    page.panels,
                )
                proposals.append((proposal, associated_panel_id))

        proposals.extend(self._associate_faces_with_bodies(detections, diagnostics))

        proposals.sort(
            key=lambda item: (
                item[0].bbox.y1,
                item[0].bbox.x1,
                item[0].bbox.y2,
                item[0].bbox.x2,
                item[1] or "",
                -(item[0].confidence or 0.0),
            )
        )
        unique_proposals: list[tuple[CharacterProposal, str | None]] = []
        for proposal, panel_id in proposals:
            duplicate_index = next(
                (
                    index
                    for index, (existing, existing_panel_id) in enumerate(unique_proposals)
                    if panel_id == existing_panel_id
                    and _iou(proposal.bbox, existing.bbox) >= self.duplicate_iou_threshold
                ),
                None,
            )
            if duplicate_index is None:
                unique_proposals.append((proposal, panel_id))

        instances = list(page.characters)
        existing_ids = {character.id for character in instances}
        new_ordinal = 1
        for proposal, panel_id in unique_proposals:
            while _stable_character_id(sequence_id, page.page_index, new_ordinal) in existing_ids:
                new_ordinal += 1
            character_id = _stable_character_id(sequence_id, page.page_index, new_ordinal)
            new_ordinal += 1
            instance = CharacterInstance(
                id=character_id,
                bbox=proposal.bbox,
                panel_id=panel_id,
                confidence=proposal.confidence,
                face_bbox=proposal.face_bbox,
                face_bboxes=proposal.face_bboxes,
                face_confidence=proposal.face_confidence,
                body_bbox=proposal.body_bbox,
                visual_embedding=proposal.visual_embedding,
                visual_metadata=proposal.visual_metadata,
            )
            instances.append(instance)
            existing_ids.add(character_id)

        crop_result = self.crop_store.generate(
            image_path,
            sequence_id,
            page.page_index,
            instances,
        )
        self._add_crop_diagnostics(crop_result.failures, diagnostics)
        page_with_characters = page.model_copy(update={"characters": instances})
        return CharacterPerceptionResult(
            page=page_with_characters,
            crop_paths=crop_result.crop_paths,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _associate_faces_with_bodies(
        detections: Sequence[tuple[CharacterDetection, str | None]],
        diagnostics: list[CharacterPerceptionDiagnostic],
    ) -> list[tuple[CharacterProposal, str | None]]:
        bodies = [item for item in detections if item[0].category == "body"]
        faces = [item for item in detections if item[0].category == "face"]
        faces_by_body: dict[str, list[CharacterDetection]] = {}
        orphan_faces: list[tuple[CharacterDetection, str | None, list[str]]] = []

        for face, face_panel_id in faces:
            face_area = _area(face.bbox)
            overlaps = [
                (
                    _intersection_area(face.bbox, body.bbox) / face_area,
                    body,
                    body_panel_id,
                )
                for body, body_panel_id in bodies
                if face_area > 0
                and (face_panel_id is None or body_panel_id is None or face_panel_id == body_panel_id)
                and _intersection_area(face.bbox, body.bbox) > 0
            ]
            overlaps.sort(key=lambda item: (-item[0], item[1].detection_id))
            if overlaps and overlaps[0][0] >= 0.55 and (
                len(overlaps) == 1 or overlaps[0][0] - overlaps[1][0] >= 0.15
            ):
                faces_by_body.setdefault(overlaps[0][1].detection_id, []).append(face)
                continue
            possible_bodies = [
                body.detection_id for coverage, body, _ in overlaps if coverage >= 0.25
            ]
            orphan_faces.append((face, face_panel_id, possible_bodies))

        proposals: list[tuple[CharacterProposal, str | None]] = []
        for body, panel_id in bodies:
            body_faces = sorted(
                faces_by_body.get(body.detection_id, []),
                key=lambda face: (
                    -(face.confidence or 0.0),
                    face.bbox.y1,
                    face.bbox.x1,
                ),
            )
            face_boxes = [face.bbox for face in body_faces]
            proposals.append((
                CharacterProposal(
                    bbox=body.bbox,
                    confidence=body.confidence,
                    face_bbox=face_boxes[0] if face_boxes else None,
                    face_bboxes=face_boxes,
                    face_confidence=body_faces[0].confidence if body_faces else None,
                    body_bbox=body.bbox,
                    visual_metadata={
                        **body.visual_metadata,
                        "detection_category": "body",
                        "body_detection_id": body.detection_id,
                        "associated_face_detection_ids": [
                            face.detection_id for face in body_faces
                        ],
                    },
                ),
                panel_id,
            ))

        for face, panel_id, possible_bodies in orphan_faces:
            diagnostics.append(
                CharacterPerceptionDiagnostic(
                    component="face_body_association",
                    code=("ambiguous_face_association" if possible_bodies else "face_without_body"),
                    message="Face evidence was preserved without forcing body ownership.",
                    panel_id=panel_id,
                )
            )
            proposals.append((
                CharacterProposal(
                    bbox=face.bbox,
                    confidence=face.confidence,
                    face_bbox=face.bbox,
                    face_bboxes=[face.bbox],
                    face_confidence=face.confidence,
                    visual_metadata={
                        **face.visual_metadata,
                        "detection_category": "face_only",
                        "face_detection_id": face.detection_id,
                        "candidate_body_detection_ids": possible_bodies,
                    },
                ),
                panel_id,
            ))
        return proposals

    @staticmethod
    def _associate_panel(
        character_bbox: BoundingBox,
        panels: Sequence[Panel],
    ) -> str | None:
        real_panels = [
            panel for panel in panels
            if panel.source == "layout_model" and _valid_bbox(panel.bbox)
        ]
        if not real_panels:
            return None
        containing = [panel for panel in real_panels if (
            panel.bbox.x1 <= character_bbox.x1
            and panel.bbox.y1 <= character_bbox.y1
            and panel.bbox.x2 >= character_bbox.x2
            and panel.bbox.y2 >= character_bbox.y2
        )]
        if containing:
            return min(containing, key=lambda panel: _area(panel.bbox)).id

        character_area = _area(character_bbox)
        overlaps = [
            (_intersection_area(panel.bbox, character_bbox) / character_area, panel)
            for panel in real_panels
            if character_area > 0 and _intersection_area(panel.bbox, character_bbox) > 0
        ]
        overlaps.sort(key=lambda item: (-item[0], item[1].id))
        if overlaps and overlaps[0][0] >= 0.8 and (
            len(overlaps) == 1 or overlaps[0][0] - overlaps[1][0] >= 0.2
        ):
            return overlaps[0][1].id
        return None

    @staticmethod
    def _add_crop_diagnostics(
        failures: Mapping[str, Mapping[str, str]],
        diagnostics: list[CharacterPerceptionDiagnostic],
    ) -> None:
        for character_id, component_errors in failures.items():
            for component, message in component_errors.items():
                diagnostics.append(
                    CharacterPerceptionDiagnostic(
                        component=f"character_crop_{component}",
                        code="crop_failure",
                        message=message,
                        character_instance_id=character_id,
                    )
                )
