"""Build CTD-owned candidate groups and structured input for Laya.

This module stops at transcription evidence and policy decisions. It does not
filter story text, establish reading order, or resolve speakers or characters.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.schemas.candidates import (
    CandidateBank,
    CandidateEvidence,
    CandidateGroup,
    CandidateSource,
    TranscriptionCandidate,
)
from app.schemas.page import BoundingBox, TextRegion

CANDIDATE_SOURCES: tuple[CandidateSource, ...] = (
    "ocr_base",
    "ocr_upscale",
    "ocr_contrast",
    "ocr_threshold",
    "ocr_rotate_cw",
    "ocr_rotate_ccw",
    "nemotron",
    "qwen3_vl",
    "fused",
)

PREPROCESSING_SOURCES: dict[str, CandidateSource] = {
    "base": "ocr_base",
    "none": "ocr_base",
    "identity": "ocr_base",
    "upscale": "ocr_upscale",
    "contrast": "ocr_contrast",
    "threshold": "ocr_threshold",
    "rotate_cw": "ocr_rotate_cw",
    "rotate_ccw": "ocr_rotate_ccw",
}

SEMANTIC_TYPES = {
    "unknown",
    "dialogue",
    "thought",
    "narration",
    "vocalisation",
    "sound_effect",
    "sign",
    "title",
    "metadata",
    "document_text",
}


def _as_mapping(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        import json

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"transcription": raw}
        return dict(parsed) if isinstance(parsed, Mapping) else {}

    json_attr = getattr(raw, "json", None)
    if callable(json_attr):
        return _as_mapping(json_attr())
    if json_attr is not None:
        return _as_mapping(json_attr)
    data_attr = getattr(raw, "data", None)
    if isinstance(data_attr, Mapping):
        return dict(data_attr)
    return {}


def _optional_confidence(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None
    return confidence


def _semantic_type(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"
    normalized = value.strip().lower().replace(" ", "_")
    if normalized == "other":
        return "unknown"
    if normalized == "vocalization":
        return "vocalisation"
    return normalized if normalized in SEMANTIC_TYPES else "unknown"


def _nemotron_bbox(
    raw: Mapping[str, Any],
    region_bbox: BoundingBox,
) -> BoundingBox | None:
    values = raw.get("bbox")
    if isinstance(values, (list, tuple)) and len(values) == 4:
        left, top, right, bottom = values
    elif all(key in raw for key in ("left", "upper", "right", "lower")):
        left, top, right, bottom = (
            raw["left"],
            raw["upper"],
            raw["right"],
            raw["lower"],
        )
    else:
        return None

    try:
        left, top, right, bottom = map(float, (left, top, right, bottom))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (left, top, right, bottom)):
        return None

    region_width = max(0.0, region_bbox.x2 - region_bbox.x1)
    region_height = max(0.0, region_bbox.y2 - region_bbox.y1)
    if max(abs(left), abs(top), abs(right), abs(bottom)) <= 1.5:
        left, right = left * region_width, right * region_width
        top, bottom = top * region_height, bottom * region_height

    x1, x2 = sorted((region_bbox.x1 + left, region_bbox.x1 + right))
    y1, y2 = sorted((region_bbox.y1 + top, region_bbox.y1 + bottom))
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def normalize_candidate(
    region: TextRegion,
    source: CandidateSource,
    raw: Any,
    *,
    candidate_id: str | None = None,
    preprocessing: str | None = None,
) -> TranscriptionCandidate | None:
    """Normalize one producer result; blank transcriptions yield no candidate."""
    payload = _as_mapping(raw)
    if isinstance(payload.get("res"), Mapping):
        payload = dict(payload["res"])

    text_keys = {
        "ocr_base": ("rec_text", "text", "transcription"),
        "ocr_upscale": ("rec_text", "text", "transcription"),
        "ocr_contrast": ("rec_text", "text", "transcription"),
        "ocr_threshold": ("rec_text", "text", "transcription"),
        "ocr_rotate_cw": ("rec_text", "text", "transcription"),
        "ocr_rotate_ccw": ("rec_text", "text", "transcription"),
        "nemotron": ("text", "transcription", "rec_text"),
        "qwen3_vl": ("transcription", "text", "rec_text"),
        "fused": ("text", "transcription", "rec_text"),
    }
    text = next(
        (payload.get(key) for key in text_keys[source] if payload.get(key) is not None),
        "",
    )
    text = str(text).strip()
    if not text:
        return None

    if source.startswith("ocr_"):
        confidence_value = payload.get("rec_score", payload.get("score"))
        ocr_confidence = _optional_confidence(confidence_value)
        visual_confidence = None
    elif source == "nemotron":
        confidence_value = payload.get("confidence", payload.get("rec_score"))
        ocr_confidence = _optional_confidence(confidence_value)
        visual_confidence = None
    else:
        ocr_confidence = _optional_confidence(
            payload.get("ocr_confidence", payload.get("rec_score"))
        )
        visual_confidence = _optional_confidence(payload.get("visual_confidence"))

    supported = payload.get("candidate_supported")
    if not isinstance(supported, bool):
        supported = None

    semantic = _semantic_type(payload.get("text_type", payload.get("semantic_type")))
    notes = payload.get("notes")
    notes = str(notes) if notes is not None else None
    bbox = _nemotron_bbox(payload, region.bbox) if source == "nemotron" else None
    evidence = CandidateEvidence(
        ocr_confidence=ocr_confidence,
        visual_confidence=visual_confidence,
        candidate_supported=supported,
        semantic_type=semantic,
        preprocessing=preprocessing,
        notes=notes,
        source_metadata=dict(payload),
    )
    return TranscriptionCandidate(
        candidate_id=candidate_id or f"{region.id}:{source}",
        region_id=region.id,
        source=source,
        text=text,
        bbox=bbox,
        evidence=evidence,
    )


def _candidate_text_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.upper()).strip()


def _bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    intersection_width = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    intersection_height = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def build_candidate_bank(
    regions: Iterable[TextRegion],
    candidates: Iterable[TranscriptionCandidate],
) -> CandidateBank:
    """Group results by CTD region ID, retaining distinct source evidence."""
    region_by_id: dict[str, TextRegion] = {}
    for region in regions:
        if region.id in region_by_id:
            raise ValueError(f"duplicate CTD region id: {region.id}")
        region_by_id[region.id] = region

    grouped: dict[str, list[TranscriptionCandidate]] = {}
    for candidate in candidates:
        if candidate.region_id not in region_by_id:
            raise ValueError(
                f"candidate {candidate.candidate_id!r} refers to unknown CTD region "
                f"{candidate.region_id!r}"
            )
        grouped.setdefault(candidate.region_id, []).append(candidate)

    groups: list[CandidateGroup] = []
    for region_id, region in region_by_id.items():
        seen_records: set[str] = set()
        retained: list[TranscriptionCandidate] = []
        for candidate in grouped.get(region_id, []):
            record_key = candidate.model_dump_json()
            if record_key not in seen_records:
                retained.append(candidate)
                seen_records.add(record_key)
        if retained:
            groups.append(
                CandidateGroup(
                    region_id=region_id,
                    bbox=region.bbox,
                    candidates=retained,
                )
            )
    return CandidateBank(groups=groups)


class CandidateProducer(Protocol):
    def candidates(
        self,
        region: TextRegion,
        crop_path: Path,
    ) -> Sequence[TranscriptionCandidate]: ...


def produce_candidate_bank(
    regions: Iterable[TextRegion],
    crop_paths: Mapping[str, str | Path | Mapping[str, str | Path]],
    producers: Sequence[CandidateProducer],
) -> CandidateBank:
    """Run injected producers on pre-cropped CTD regions and build the bank.

    A region may map to one base crop or to preprocessing-name/crop pairs.
    Variant-aware producers require their exact crop; other producers use the
    ``base`` crop when given a variant mapping.
    """
    region_list = list(regions)
    candidates: list[TranscriptionCandidate] = []
    for region in region_list:
        if region.id not in crop_paths:
            raise KeyError(f"missing localized crop for CTD region {region.id!r}")
        region_crops = crop_paths[region.id]
        for producer in producers:
            if isinstance(region_crops, Mapping):
                crop_variant = getattr(producer, "preprocessing", "base")
                if crop_variant not in region_crops:
                    raise KeyError(
                        f"missing {crop_variant!r} crop for CTD region {region.id!r}"
                    )
                crop_path = Path(region_crops[crop_variant])
            else:
                crop_path = Path(region_crops)
            candidates.extend(producer.candidates(region, crop_path))
    return build_candidate_bank(region_list, candidates)


class RegionGeometry(BaseModel):
    width: float
    height: float
    area: float
    aspect_ratio: float | None = None
    normalized_width: float | None = None
    normalized_height: float | None = None


class CandidateSpatialFeatures(BaseModel):
    region_iou: float | None = None
    center_offset_x: float | None = None
    center_offset_y: float | None = None


class LayaCandidateInput(BaseModel):
    candidate_id: str
    source: CandidateSource
    text: str
    ocr_confidence: float | None = None
    visual_confidence: float | None = None
    candidate_supported: bool | None = None
    text_type: str
    preprocessing: str | None = None
    notes: str | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    text_length: int
    character_count: int
    word_count: int
    bbox: BoundingBox | None = None
    normalized_text_similarity: dict[str, float] = Field(default_factory=dict)
    spatial_consistency: CandidateSpatialFeatures = Field(
        default_factory=CandidateSpatialFeatures
    )


class LayaInput(BaseModel):
    """Image-free, per-region evidence contract sent to the Laya policy."""

    region_id: str
    region_bbox: BoundingBox | None = None
    region_geometry: RegionGeometry | None = None
    candidate_count: int
    source_presence: dict[str, bool]
    candidates: list[LayaCandidateInput] = Field(min_length=1)


def to_laya_input(
    group: CandidateGroup,
    *,
    page_size: tuple[float, float] | None = None,
) -> LayaInput:
    """Compute deterministic comparison and geometry features for one group."""
    box = group.bbox
    geometry = None
    if box is not None:
        width = max(0.0, box.x2 - box.x1)
        height = max(0.0, box.y2 - box.y1)
        page_width, page_height = page_size or (None, None)
        geometry = RegionGeometry(
            width=width,
            height=height,
            area=width * height,
            aspect_ratio=width / height if height else None,
            normalized_width=width / page_width if page_width else None,
            normalized_height=height / page_height if page_height else None,
        )

    source_presence = {
        source: any(candidate.source == source for candidate in group.candidates)
        for source in CANDIDATE_SOURCES
    }
    features: list[LayaCandidateInput] = []
    for candidate in group.candidates:
        similarities = {
            other.candidate_id: SequenceMatcher(
                None,
                _candidate_text_key(candidate.text),
                _candidate_text_key(other.text),
                autojunk=False,
            ).ratio()
            for other in group.candidates
            if other.candidate_id != candidate.candidate_id
        }
        spatial = CandidateSpatialFeatures()
        if box is not None and candidate.bbox is not None:
            region_width = max(0.0, box.x2 - box.x1)
            region_height = max(0.0, box.y2 - box.y1)
            candidate_center_x = (candidate.bbox.x1 + candidate.bbox.x2) / 2
            candidate_center_y = (candidate.bbox.y1 + candidate.bbox.y2) / 2
            region_center_x = (box.x1 + box.x2) / 2
            region_center_y = (box.y1 + box.y2) / 2
            spatial = CandidateSpatialFeatures(
                region_iou=_bbox_iou(candidate.bbox, box),
                center_offset_x=(candidate_center_x - region_center_x) / region_width
                if region_width
                else None,
                center_offset_y=(candidate_center_y - region_center_y) / region_height
                if region_height
                else None,
            )
        features.append(
            LayaCandidateInput(
                candidate_id=candidate.candidate_id,
                source=candidate.source,
                text=candidate.text,
                ocr_confidence=candidate.evidence.ocr_confidence,
                visual_confidence=candidate.evidence.visual_confidence,
                candidate_supported=candidate.evidence.candidate_supported,
                text_type=candidate.evidence.semantic_type,
                preprocessing=candidate.evidence.preprocessing,
                notes=candidate.evidence.notes,
                source_metadata=candidate.evidence.source_metadata,
                text_length=len(candidate.text),
                character_count=sum(not character.isspace() for character in candidate.text),
                word_count=len(candidate.text.split()),
                bbox=candidate.bbox,
                normalized_text_similarity=similarities,
                spatial_consistency=spatial,
            )
        )
    return LayaInput(
        region_id=group.region_id,
        region_bbox=box,
        region_geometry=geometry,
        candidate_count=len(features),
        source_presence=source_presence,
        candidates=features,
    )


class LayaDecision(BaseModel):
    region_id: str
    selected_candidate_id: str
    decision_confidence: float | None = None
    reason: str | None = None
    features_used: dict[str, Any] = Field(default_factory=dict)
    raw_decision: dict[str, Any] = Field(default_factory=dict)


def _default_laya_response(raw: Any) -> Mapping[str, Any] | str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, Mapping):
        return raw
    model_dump = getattr(raw, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    raise TypeError("Laya response must be a candidate ID or a mapping")


class LayaPolicyAdapter:
    """Bridge image-free region evidence to Laya's predict(state, questions) API."""

    def __init__(
        self,
        policy: Any,
        response_parser: Callable[[Any], Mapping[str, Any] | str] | None = None,
    ) -> None:
        self.policy = policy
        self.response_parser = response_parser or _default_laya_response

    def decide(self, laya_input: LayaInput) -> LayaDecision:
        state = laya_input.model_dump(mode="json")
        questions = {
            "select_transcription": {
                "type": "choice",
                "instructions": (
                    "Select the candidate transcription best supported by the "
                    "structured evidence. Return its candidate_id exactly. "
                    "Do not invent or rewrite a transcription."
                ),
                "criteria": {
                    candidate.candidate_id: (
                        f"source={candidate.source}; text={candidate.text}"
                    )
                    for candidate in laya_input.candidates
                },
            }
        }
        parsed = self.response_parser(self.policy.predict(state, questions))
        if isinstance(parsed, str):
            selected_id = parsed.strip()
            response: Mapping[str, Any] = {}
        else:
            response = parsed
            selected_id = response.get(
                "selected_candidate_id",
                response.get("candidate_id", response.get("select_transcription", "")),
            )
            if isinstance(selected_id, Mapping):
                selected_id = selected_id.get("candidate_id", selected_id.get("choice", ""))
            selected_id = str(selected_id).strip() if selected_id is not None else ""

        candidate_ids = {candidate.candidate_id for candidate in laya_input.candidates}
        if selected_id not in candidate_ids:
            raise ValueError(
                "Laya response must explicitly select one of the supplied candidate IDs"
            )

        confidence = _optional_confidence(
            response.get("decision_confidence", response.get("confidence"))
        )
        reason = response.get("reason", response.get("notes"))
        features_used = response.get("features_used", response.get("features", {}))
        return LayaDecision(
            region_id=laya_input.region_id,
            selected_candidate_id=selected_id,
            decision_confidence=confidence,
            reason=str(reason) if reason is not None else None,
            features_used=dict(features_used) if isinstance(features_used, Mapping) else {},
            raw_decision=dict(response),
        )