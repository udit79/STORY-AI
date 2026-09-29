from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .page import BoundingBox

CandidateSource = Literal[
    "ocr_base",
    "ocr_upscale",
    "ocr_contrast",
    "ocr_threshold",
    "ocr_rotate_cw",
    "ocr_rotate_ccw",
    "nemotron",
    "qwen3_vl",
    "fused",
]


class CandidateEvidence(BaseModel):
    """
    Structured evidence attached to one transcription candidate.

    These values are evidence for Laya; they are not assumed to be
    calibrated probabilities.
    """

    ocr_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    visual_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    candidate_supported: bool | None = None

    semantic_type: Literal[
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
    ] = "unknown"

    preprocessing: str | None = None

    notes: str | None = None

    source_metadata: dict[str, Any] = Field(default_factory=dict)


class TranscriptionCandidate(BaseModel):
    """
    One possible transcription for one localized text block.

    A candidate is evidence, not the final answer.
    """

    candidate_id: str
    region_id: str

    source: CandidateSource

    text: str

    bbox: BoundingBox | None = None

    evidence: CandidateEvidence = Field(
        default_factory=CandidateEvidence
    )

    @field_validator("candidate_id", "region_id", "text")
    @classmethod
    def require_non_empty_value(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class CandidateGroup(BaseModel):
    """
    All competing transcription candidates for one localized region.
    """

    region_id: str

    bbox: BoundingBox | None = None

    candidates: list[TranscriptionCandidate] = Field(
        default_factory=list,
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_candidate_ownership(self) -> "CandidateGroup":
        if any(candidate.region_id != self.region_id for candidate in self.candidates):
            raise ValueError("all candidates must belong to the group's region_id")
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate_id values must be unique within a group")
        return self


class CandidateBank(BaseModel):
    """
    Candidate collection passed downstream to the learned decision layer.
    """

    groups: list[CandidateGroup] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_unique_regions(self) -> "CandidateBank":
        region_ids = [group.region_id for group in self.groups]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("region_id values must be unique within a bank")
        return self
