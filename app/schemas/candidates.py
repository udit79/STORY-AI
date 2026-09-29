from typing import Literal

from pydantic import BaseModel, Field

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


class CandidateGroup(BaseModel):
    """
    All competing transcription candidates for one localized region.
    """

    region_id: str

    candidates: list[TranscriptionCandidate] = Field(
        default_factory=list,
        min_length=1,
    )


class CandidateBank(BaseModel):
    """
    Candidate collection passed downstream to the learned decision layer.
    """

    groups: list[CandidateGroup] = Field(
        default_factory=list
    )
