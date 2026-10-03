"""Build CTD-owned candidate groups and structured evidence for adjudication.

This module stops at transcription evidence. It does not
filter story text, establish reading order, or resolve speakers or characters.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

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

_SKIP_METADATA = object()


def _metadata_value(value: Any) -> Any:
    """Keep useful scalar metadata without retaining images or opaque objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {
            str(key): safe_value
            for key, item in value.items()
            if (safe_value := _metadata_value(item)) is not _SKIP_METADATA
        }
    if isinstance(value, (list, tuple)):
        return [
            safe_value
            for item in value
            if (safe_value := _metadata_value(item)) is not _SKIP_METADATA
        ]

    scalar_item = getattr(value, "item", None)
    if callable(scalar_item):
        try:
            scalar = scalar_item()
        except (TypeError, ValueError):
            return _SKIP_METADATA
        if scalar is not value:
            return _metadata_value(scalar)
    return _SKIP_METADATA


def _safe_source_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): safe_value
        for key, value in payload.items()
        if (safe_value := _metadata_value(value)) is not _SKIP_METADATA
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
        source_metadata=_safe_source_metadata(payload),
    )
    return TranscriptionCandidate(
        candidate_id=candidate_id or f"{region.id}:{source}",
        region_id=region.id,
        source=source,
        text=text,
        bbox=bbox,
        evidence=evidence,
    )


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
