"""Traceable geometry- and vision-based balloon-to-character grounding."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import quote

import cv2
from pydantic import BaseModel, Field

from app.schemas.page import (
    Balloon,
    BoundingBox,
    CharacterInstance,
    PageRepresentation,
    Panel,
    Point2D,
)


class BalloonTailGeometry(BaseModel):
    endpoint: Point2D
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    method: Literal["mask_polygon", "injected"]
    evidence: dict[str, Any] = Field(default_factory=dict)


class TailGeometryProvider(Protocol):
    def detect_tail(self, balloon: Balloon) -> BalloonTailGeometry | None: ...


class SpeakerCandidateEvidence(BaseModel):
    same_panel: bool | None = None
    center_distance_px: float | None = None
    normalized_center_distance: float | None = None
    balloon_character_overlap: float | None = None
    normalized_bbox_distance: float | None = None
    normalized_face_distance: float | None = None
    tail_endpoint_distance_px: float | None = None
    normalized_tail_endpoint_distance: float | None = None
    tail_endpoint_inside_character: bool | None = None
    tail_alignment: float | None = None
    face_available: bool = False
    body_available: bool = False
    visual_support: bool | None = None


class SpeakerCandidate(BaseModel):
    balloon_id: str
    character_instance_id: str
    evidence: SpeakerCandidateEvidence


class SpeakerDiagnostic(BaseModel):
    code: str
    message: str
    balloon_id: str


class VisualSpeakerContext(BaseModel):
    image_paths: list[str] = Field(min_length=1)
    character_labels: dict[str, str]


class VisualSpeakerGrounding(BaseModel):
    character_instance_id: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str | None = None
    diagnostics: list[SpeakerDiagnostic] = Field(default_factory=list)


class SpeakerDecision(BaseModel):
    balloon_id: str
    selected_character_instance_id: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    method: Literal["geometry", "visual", "hybrid", "unknown"]
    candidates: list[SpeakerCandidate] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    diagnostics: list[SpeakerDiagnostic] = Field(default_factory=list)


class PageSpeakerGroundingResult(BaseModel):
    page: PageRepresentation
    decisions: list[SpeakerDecision]
    diagnostics: list[SpeakerDiagnostic] = Field(default_factory=list)


class VisualSpeakerGrounder(Protocol):
    def resolve(
        self,
        balloon_id: str,
        candidates: Sequence[SpeakerCandidate],
        context: VisualSpeakerContext,
    ) -> VisualSpeakerGrounding: ...


def _valid_bbox(box: BoundingBox) -> bool:
    values = (box.x1, box.y1, box.x2, box.y2)
    return (
        all(math.isfinite(value) for value in values)
        and box.x2 > box.x1
        and box.y2 > box.y1
    )


def _area(box: BoundingBox) -> float:
    return max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1)


def _intersection_area(first: BoundingBox, second: BoundingBox) -> float:
    width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    return width * height


def _point_bbox_distance(point: Point2D, bbox: BoundingBox) -> float:
    dx = max(bbox.x1 - point.x, 0.0, point.x - bbox.x2)
    dy = max(bbox.y1 - point.y, 0.0, point.y - bbox.y2)
    return math.hypot(dx, dy)


def _bbox_distance(first: BoundingBox, second: BoundingBox) -> float:
    """Minimum Euclidean distance between two axis-aligned boxes."""
    dx = max(first.x1 - second.x2, second.x1 - first.x2, 0.0)
    dy = max(first.y1 - second.y2, second.y1 - first.y2, 0.0)
    return math.hypot(dx, dy)


def _point_inside(point: Point2D, bbox: BoundingBox) -> bool:
    return bbox.x1 <= point.x <= bbox.x2 and bbox.y1 <= point.y <= bbox.y2


def _cosine(a: tuple[float, float], b: tuple[float, float]) -> float | None:
    norm_a = math.hypot(*a)
    norm_b = math.hypot(*b)
    if norm_a == 0 or norm_b == 0:
        return None
    return max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (norm_a * norm_b)))


class MaskPolygonTailGeometryProvider:
    """Extract an optional pointed tail tip from a preserved balloon mask polygon."""

    def __init__(self, max_tip_angle_degrees: float = 60.0, min_extension_ratio: float = 0.68):
        self.max_tip_angle_degrees = max_tip_angle_degrees
        self.min_extension_ratio = min_extension_ratio

    def detect_tail(self, balloon: Balloon) -> BalloonTailGeometry | None:
        polygon = balloon.mask_polygon
        if polygon is None or len(polygon) < 4:
            return None
        center_x = sum(point.x for point in polygon) / len(polygon)
        center_y = sum(point.y for point in polygon) / len(polygon)
        radii = [math.hypot(point.x - center_x, point.y - center_y) for point in polygon]
        max_radius = max(radii, default=0.0)
        if max_radius <= 0:
            return None

        possible_tips: list[tuple[float, int, float, float]] = []
        for index, point in enumerate(polygon):
            previous = polygon[index - 1]
            following = polygon[(index + 1) % len(polygon)]
            left = (previous.x - point.x, previous.y - point.y)
            right = (following.x - point.x, following.y - point.y)
            left_length = math.hypot(*left)
            right_length = math.hypot(*right)
            if left_length == 0 or right_length == 0:
                continue
            cosine = max(
                -1.0,
                min(1.0, (left[0] * right[0] + left[1] * right[1]) / (left_length * right_length)),
            )
            angle = math.degrees(math.acos(cosine))
            extension = radii[index] / max_radius
            if angle <= self.max_tip_angle_degrees and extension >= self.min_extension_ratio:
                strength = ((self.max_tip_angle_degrees - angle) / self.max_tip_angle_degrees) * extension
                possible_tips.append((strength, index, angle, extension))
        if not possible_tips:
            return None
        possible_tips.sort(reverse=True)
        if len(possible_tips) > 1 and possible_tips[0][0] - possible_tips[1][0] < 0.12:
            return None
        strength, index, angle, extension = possible_tips[0]
        return BalloonTailGeometry(
            endpoint=polygon[index],
            confidence=None,
            method="mask_polygon",
            evidence={
                "tip_supported": True,
                "tip_strength": strength,
                "vertex_angle_degrees": angle,
                "radial_extension_ratio": extension,
            },
        )


def generate_speaker_candidates(
    balloon: Balloon,
    characters: Sequence[CharacterInstance],
    panels: Sequence[Panel],
    *,
    page_size: tuple[int, int] | None = None,
    tail_geometry: BalloonTailGeometry | None = None,
) -> list[SpeakerCandidate]:
    """Generate geometric speaker evidence without looking at transcription text."""
    if not _valid_bbox(balloon.bbox):
        return []
    if not characters:
        return []

    real_panel_ids = {panel.id for panel in panels if panel.source == "layout_model"}
    panel_context_exists = balloon.panel_id in real_panel_ids
    panel_characters = [
        character for character in characters
        if _valid_bbox(character.bbox)
        and panel_context_exists
        and character.panel_id == balloon.panel_id
    ]
    selected_characters = panel_characters if panel_characters else [
        character for character in characters if _valid_bbox(character.bbox)
    ]
    if not selected_characters:
        return []

    width, height = page_size or (0, 0)
    page_diagonal = math.hypot(width, height) if width > 0 and height > 0 else None
    balloon_center = (
        (balloon.bbox.x1 + balloon.bbox.x2) / 2,
        (balloon.bbox.y1 + balloon.bbox.y2) / 2,
    )
    tail_vector = None
    if tail_geometry is not None:
        tail_vector = (
            tail_geometry.endpoint.x - balloon_center[0],
            tail_geometry.endpoint.y - balloon_center[1],
        )

    candidates: list[SpeakerCandidate] = []
    for character in selected_characters:
        character_center = (
            (character.bbox.x1 + character.bbox.x2) / 2,
            (character.bbox.y1 + character.bbox.y2) / 2,
        )
        center_distance = math.hypot(
            character_center[0] - balloon_center[0],
            character_center[1] - balloon_center[1],
        )
        intersection = _intersection_area(balloon.bbox, character.bbox)
        smaller_area = min(_area(balloon.bbox), _area(character.bbox))
        overlap = intersection / smaller_area if smaller_area > 0 else 0.0
        bbox_distance = _bbox_distance(balloon.bbox, character.bbox)
        face_distance = (
            _point_bbox_distance(
                Point2D(x=balloon_center[0], y=balloon_center[1]),
                character.face_bbox,
            )
            if character.face_bbox is not None and _valid_bbox(character.face_bbox)
            else None
        )
        tail_distance = None
        tail_inside = None
        alignment = None
        if tail_geometry is not None:
            tail_distance = _point_bbox_distance(tail_geometry.endpoint, character.bbox)
            tail_inside = _point_inside(tail_geometry.endpoint, character.bbox)
            if tail_vector is not None:
                alignment = _cosine(
                    tail_vector,
                    (
                        character_center[0] - balloon_center[0],
                        character_center[1] - balloon_center[1],
                    ),
                )
        candidates.append(
            SpeakerCandidate(
                balloon_id=balloon.id,
                character_instance_id=character.id,
                evidence=SpeakerCandidateEvidence(
                    same_panel=(
                        character.panel_id == balloon.panel_id
                        if panel_context_exists
                        else None
                    ),
                    center_distance_px=center_distance,
                    normalized_center_distance=(
                        center_distance / page_diagonal if page_diagonal else None
                    ),
                    balloon_character_overlap=overlap,
                    normalized_bbox_distance=(
                        bbox_distance / page_diagonal if page_diagonal else None
                    ),
                    normalized_face_distance=(
                        face_distance / page_diagonal
                        if face_distance is not None and page_diagonal
                        else None
                    ),
                    tail_endpoint_distance_px=tail_distance,
                    normalized_tail_endpoint_distance=(
                        tail_distance / page_diagonal
                        if tail_distance is not None and page_diagonal
                        else None
                    ),
                    tail_endpoint_inside_character=tail_inside,
                    tail_alignment=alignment,
                    face_available=character.face_bbox is not None,
                    body_available=character.body_bbox is not None,
                ),
            )
        )
    return candidates


class QwenVisualSpeakerGrounder:
    """Optional local Qwen ID selector for ambiguous, explicitly labeled candidates."""

    def __init__(self, runner: Any) -> None:
        if not callable(getattr(runner, "generate_json", None)):
            raise TypeError("runner must expose generate_json(image_paths, prompt)")
        self.runner = runner

    def resolve(
        self,
        balloon_id: str,
        candidates: Sequence[SpeakerCandidate],
        context: VisualSpeakerContext,
    ) -> VisualSpeakerGrounding:
        allowed_ids = {candidate.character_instance_id for candidate in candidates}
        labels = {
            label: character_id
            for label, character_id in context.character_labels.items()
            if character_id in allowed_ids
        }
        prompt = (
            "Select which visible character most likely produced the marked balloon. "
            "Use only the balloon and labeled panel images. Do not use text content to infer "
            "the speaker. Inspect balloon-tail direction and visible character position. "
            "Return strict JSON with character_instance_id (one allowed ID or null), "
            "confidence (optional, uncalibrated), and reason. Never invent IDs.\n"
            f"Balloon ID: {balloon_id}\n"
            f"Allowed character IDs: {json.dumps(sorted(allowed_ids))}\n"
            f"Image labels: {json.dumps(labels, sort_keys=True)}\n"
            f"Candidate evidence: {json.dumps([candidate.model_dump(mode='json') for candidate in candidates])}"
        )
        try:
            response = self.runner.generate_json(
                [Path(image_path) for image_path in context.image_paths],
                prompt,
            )
        except Exception as exc:  # noqa: BLE001 - optional visual evidence cannot break geometry
            return VisualSpeakerGrounding(
                diagnostics=[
                    SpeakerDiagnostic(
                        balloon_id=balloon_id,
                        code="visual_grounder_failure",
                        message=f"{type(exc).__name__}: {exc}",
                    )
                ]
            )

        if isinstance(response, str):
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            try:
                response = json.loads(cleaned)
            except json.JSONDecodeError:
                return VisualSpeakerGrounding(
                    diagnostics=[
                        SpeakerDiagnostic(
                            balloon_id=balloon_id,
                            code="visual_grounder_malformed_json",
                            message="Qwen visual grounding response was not valid JSON.",
                        )
                    ]
                )
        if not isinstance(response, Mapping) or response.get("_json_parse_error"):
            return VisualSpeakerGrounding(
                diagnostics=[
                    SpeakerDiagnostic(
                        balloon_id=balloon_id,
                        code="visual_grounder_malformed_json",
                        message="Qwen visual grounding did not return a JSON object.",
                    )
                ]
            )

        selected_id = response.get("character_instance_id")
        diagnostics: list[SpeakerDiagnostic] = []
        if selected_id is not None and selected_id not in allowed_ids:
            diagnostics.append(
                SpeakerDiagnostic(
                    balloon_id=balloon_id,
                    code="invalid_visual_character_id",
                    message="Qwen selected a character ID not present in the supplied candidates.",
                )
            )
            selected_id = None
        confidence_value = response.get("confidence")
        confidence = None
        if confidence_value is not None:
            if (
                isinstance(confidence_value, bool)
                or not isinstance(confidence_value, (int, float))
                or not math.isfinite(float(confidence_value))
                or not 0.0 <= float(confidence_value) <= 1.0
            ):
                diagnostics.append(
                    SpeakerDiagnostic(
                        balloon_id=balloon_id,
                        code="invalid_visual_confidence",
                        message="Qwen visual confidence was malformed or out of range.",
                    )
                )
            else:
                confidence = float(confidence_value)
        reason = response.get("reason")
        return VisualSpeakerGrounding(
            character_instance_id=selected_id,
            confidence=confidence,
            reason=str(reason) if reason is not None else None,
            diagnostics=diagnostics,
        )


class SpeakerResolver:
    """Resolve speakers from corroborated geometry, then optional Qwen grounding."""

    def __init__(
        self,
        visual_grounder: VisualSpeakerGrounder | None = None,
        *,
        visual_acceptance_threshold: float = 0.65,
        tail_geometry_provider: TailGeometryProvider | None = None,
    ) -> None:
        if not 0.0 <= visual_acceptance_threshold <= 1.0:
            raise ValueError("visual_acceptance_threshold must be between 0 and 1")
        self.visual_grounder = visual_grounder
        self.visual_acceptance_threshold = visual_acceptance_threshold
        self.tail_geometry_provider = tail_geometry_provider or MaskPolygonTailGeometryProvider()

    @staticmethod
    def _no_tail_score(candidate: SpeakerCandidate) -> tuple[float, dict[str, float]]:
        """Score independent geometric relationships when no tail is usable.

        Overlap and bbox proximity are deliberately bounded contributions: a
        nearest-center candidate without overlap cannot pass the acceptance
        gate below.  The returned feature map is persisted in diagnostics so
        decisions remain inspectable.
        """
        evidence = candidate.evidence
        features: dict[str, float] = {}
        if evidence.balloon_character_overlap is not None:
            features["overlap"] = min(1.0, evidence.balloon_character_overlap / 0.75)
        if evidence.normalized_bbox_distance is not None:
            features["bbox_proximity"] = max(
                0.0, 1.0 - evidence.normalized_bbox_distance / 0.15
            )
        if evidence.normalized_face_distance is not None:
            features["face_proximity"] = max(
                0.0, 1.0 - evidence.normalized_face_distance / 0.25
            )
        if evidence.same_panel is True:
            features["same_panel"] = 1.0

        weights = {
            "overlap": 0.55,
            "bbox_proximity": 0.20,
            "face_proximity": 0.15,
            "same_panel": 0.10,
        }
        score = sum(weights[name] * value for name, value in features.items())
        return score, features

    @classmethod
    def _resolve_no_tail_geometry(
        cls,
        candidate_list: list[SpeakerCandidate],
        base_evidence: dict[str, Any],
    ) -> SpeakerDecision | None:
        ranked = sorted(
            ((cls._no_tail_score(candidate), candidate) for candidate in candidate_list),
            key=lambda item: (-item[0][0], item[1].character_instance_id),
        )
        if not ranked:
            return None
        (best_score, best_features), best_candidate = ranked[0]
        runner_up_score = ranked[1][0][0] if len(ranked) > 1 else 0.0
        margin = best_score - runner_up_score
        normalized_overlap = best_features.get("overlap", 0.0)
        has_strong_overlap = normalized_overlap >= 0.65
        accepted = best_score >= 0.60 and margin >= 0.15 and (
            has_strong_overlap or best_candidate.evidence.same_panel is True
        )
        ranking = [
            {
                "character_id": candidate.character_instance_id,
                "score": score,
                "features": features,
            }
            for (score, features), candidate in ranked
        ]
        evidence = {
            **base_evidence,
            "rule": "multi_feature_no_tail",
            "candidate_scores": ranking,
            "selected_score": best_score,
            "runner_up_score": runner_up_score,
            "score_margin": margin,
            "acceptance_threshold": 0.60,
            "margin_threshold": 0.15,
        }
        if not accepted:
            return SpeakerDecision(
                balloon_id=best_candidate.balloon_id,
                method="unknown",
                candidates=candidate_list,
                evidence={**evidence, "rejection_reason": "insufficient_or_ambiguous_no_tail_evidence"},
            )
        confidence = min(0.99, max(0.0, 0.5 + 0.5 * min(1.0, best_score + margin)))
        return SpeakerDecision(
            balloon_id=best_candidate.balloon_id,
            selected_character_instance_id=best_candidate.character_instance_id,
            confidence=confidence,
            method="geometry",
            candidates=candidate_list,
            evidence=evidence,
        )

    def resolve(
        self,
        balloon: Balloon,
        candidates: Sequence[SpeakerCandidate],
        *,
        tail_geometry: BalloonTailGeometry | None = None,
        visual_context: VisualSpeakerContext | None = None,
    ) -> SpeakerDecision:
        candidate_list = list(candidates)
        diagnostics: list[SpeakerDiagnostic] = []
        base_evidence: dict[str, Any] = {
            "tail_geometry": tail_geometry.model_dump(mode="json") if tail_geometry else None,
            "selection_rule": "distance alone never selects a speaker",
        }
        if not candidate_list:
            return SpeakerDecision(
                balloon_id=balloon.id,
                method="unknown",
                candidates=[],
                evidence=base_evidence,
            )

        tail_hits = [
            candidate for candidate in candidate_list
            if candidate.evidence.tail_endpoint_inside_character is True
        ]
        if tail_geometry is not None and len(tail_hits) == 1:
            selected = tail_hits[0]
            return SpeakerDecision(
                balloon_id=balloon.id,
                selected_character_instance_id=selected.character_instance_id,
                confidence=tail_geometry.confidence,
                method="geometry",
                candidates=candidate_list,
                evidence={**base_evidence, "rule": "unique_tail_endpoint_inside_character"},
            )

        if tail_geometry is not None:
            tail_distances = [
                candidate for candidate in candidate_list
                if candidate.evidence.normalized_tail_endpoint_distance is not None
            ]
            tail_distances.sort(
                key=lambda candidate: candidate.evidence.normalized_tail_endpoint_distance
            )
            if len(tail_distances) > 1:
                best_distance = tail_distances[0].evidence.normalized_tail_endpoint_distance
                next_distance = tail_distances[1].evidence.normalized_tail_endpoint_distance
                if (
                    best_distance is not None
                    and next_distance is not None
                    and best_distance <= 0.025
                    and next_distance - best_distance >= 0.03
                ):
                    selected = tail_distances[0]
                    return SpeakerDecision(
                        balloon_id=balloon.id,
                        selected_character_instance_id=selected.character_instance_id,
                        confidence=tail_geometry.confidence,
                        method="geometry",
                        candidates=candidate_list,
                        evidence={**base_evidence, "rule": "unique_near_tail_endpoint"},
                    )

        panel_candidates = [
            candidate for candidate in candidate_list
            if candidate.evidence.same_panel is True
        ]
        if len(candidate_list) == 1 and len(panel_candidates) == 1:
            candidate = panel_candidates[0]
            return SpeakerDecision(
                balloon_id=balloon.id,
                selected_character_instance_id=candidate.character_instance_id,
                confidence=0.58,
                method="geometry",
                candidates=candidate_list,
                evidence={
                    **base_evidence,
                    "rule": "only_visible_character_instance_in_panel",
                    "confidence_is_heuristic": True,
                },
            )

        if tail_geometry is None:
            no_tail_decision = self._resolve_no_tail_geometry(candidate_list, base_evidence)
            if no_tail_decision is not None and no_tail_decision.selected_character_instance_id is not None:
                return no_tail_decision
            no_tail_evidence = (
                no_tail_decision.evidence
                if no_tail_decision is not None
                else {**base_evidence, "rejection_reason": "no_candidates"}
            )
        else:
            no_tail_evidence = base_evidence

        if self.visual_grounder is not None and visual_context is not None:
            visual = self.visual_grounder.resolve(balloon.id, candidate_list, visual_context)
            diagnostics.extend(visual.diagnostics)
            allowed_ids = {candidate.character_instance_id for candidate in candidate_list}
            if (
                visual.character_instance_id in allowed_ids
                and visual.confidence is not None
                and visual.confidence >= self.visual_acceptance_threshold
            ):
                method: Literal["visual", "hybrid"] = (
                    "hybrid"
                    if tail_geometry is not None or panel_candidates
                    else "visual"
                )
                return SpeakerDecision(
                    balloon_id=balloon.id,
                    selected_character_instance_id=visual.character_instance_id,
                    confidence=visual.confidence,
                    method=method,
                    candidates=candidate_list,
                    evidence={
                        **base_evidence,
                        "visual_reason": visual.reason,
                        "visual_confidence_is_uncalibrated": True,
                    },
                    diagnostics=diagnostics,
                )
            if visual.character_instance_id is not None:
                diagnostics.append(
                    SpeakerDiagnostic(
                        balloon_id=balloon.id,
                        code="visual_decision_below_threshold",
                        message="Qwen selection was not accepted; speaker remains unresolved.",
                    )
                )

        return SpeakerDecision(
            balloon_id=balloon.id,
            method="unknown",
            candidates=candidate_list,
            evidence=no_tail_evidence,
            diagnostics=diagnostics,
        )

    def resolve_page(
        self,
        page: PageRepresentation,
        *,
        page_size: tuple[int, int] | None = None,
        visual_contexts: Mapping[str, VisualSpeakerContext] | None = None,
    ) -> PageSpeakerGroundingResult:
        decisions: list[SpeakerDecision] = []
        diagnostics: list[SpeakerDiagnostic] = []
        context_map = dict(visual_contexts or {})
        balloon_updates: dict[str, Balloon] = {}
        for balloon in page.balloons:
            try:
                tail = self.tail_geometry_provider.detect_tail(balloon)
            except Exception as exc:  # noqa: BLE001 - tail geometry is optional evidence
                tail = None
                diagnostics.append(
                    SpeakerDiagnostic(
                        balloon_id=balloon.id,
                        code="tail_geometry_failure",
                        message=f"{type(exc).__name__}: {exc}",
                    )
                )
            candidates = generate_speaker_candidates(
                balloon,
                page.characters,
                page.panels,
                page_size=page_size,
                tail_geometry=tail,
            )
            decision = self.resolve(
                balloon,
                candidates,
                tail_geometry=tail,
                visual_context=context_map.get(balloon.id),
            )
            decisions.append(decision)
            diagnostics.extend(decision.diagnostics)
            balloon_updates[balloon.id] = balloon.model_copy(update={
                "candidate_character_ids": [candidate.character_instance_id for candidate in candidates]
            })

        updated_page = page.model_copy(update={
            "balloons": [balloon_updates[balloon.id] for balloon in page.balloons]
        })
        return PageSpeakerGroundingResult(
            page=updated_page,
            decisions=decisions,
            diagnostics=diagnostics,
        )


class SpeakerContextImageStore:
    """Persist a labeled panel/page view for Qwen only when visual grounding is needed."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def create(
        self,
        page_image_path: str | Path,
        sequence_id: str,
        page_index: int,
        balloon: Balloon,
        characters: Sequence[CharacterInstance],
        *,
        panel: Panel | None = None,
    ) -> VisualSpeakerContext:
        page_image_path = Path(page_image_path)
        image = cv2.imread(str(page_image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read speaker context image: {page_image_path}")
        image_height, image_width = image.shape[:2]
        crop_bbox = (
            panel.bbox
            if panel is not None and panel.source == "layout_model" and _valid_bbox(panel.bbox)
            else BoundingBox(x1=0, y1=0, x2=image_width, y2=image_height)
        )
        x1, y1 = max(0, math.floor(crop_bbox.x1)), max(0, math.floor(crop_bbox.y1))
        x2, y2 = min(image_width, math.ceil(crop_bbox.x2)), min(image_height, math.ceil(crop_bbox.y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("panel/page context bbox does not intersect the page")
        context = image[y1:y2, x1:x2].copy()

        balloon_box = self._offset_bbox(balloon.bbox, x1, y1)
        cv2.rectangle(
            context,
            (int(balloon_box.x1), int(balloon_box.y1)),
            (int(balloon_box.x2), int(balloon_box.y2)),
            (0, 0, 255),
            3,
        )
        cv2.putText(
            context,
            "BALLOON",
            (int(balloon_box.x1), max(18, int(balloon_box.y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )

        labels: dict[str, str] = {}
        for index, character in enumerate(characters, start=1):
            label = f"C{index}"
            labels[label] = character.id
            char_box = self._offset_bbox(character.bbox, x1, y1)
            cv2.rectangle(
                context,
                (int(char_box.x1), int(char_box.y1)),
                (int(char_box.x2), int(char_box.y2)),
                (0, 220, 0),
                2,
            )
            cv2.putText(
                context,
                label,
                (int(char_box.x1), max(18, int(char_box.y1) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 0),
                2,
            )

        sequence_component = quote(sequence_id, safe="-_.") or "_"
        balloon_component = quote(balloon.id, safe="-_.") or "_"
        output_path = (
            self.root
            / sequence_component
            / f"page_{page_index:02d}"
            / balloon_component
            / "speaker_context.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), context):
            raise OSError(f"Could not write speaker context image: {output_path}")
        return VisualSpeakerContext(
            image_paths=[str(output_path)],
            character_labels=labels,
        )

    @staticmethod
    def _offset_bbox(bbox: BoundingBox, x_offset: int, y_offset: int) -> BoundingBox:
        return BoundingBox(
            x1=max(0.0, bbox.x1 - x_offset),
            y1=max(0.0, bbox.y1 - y_offset),
            x2=max(0.0, bbox.x2 - x_offset),
            y2=max(0.0, bbox.y2 - y_offset),
        )
