"""Typed contracts for balloon-level multimodal adjudication."""

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .candidates import CandidateSource
from .page import BoundingBox, TextRegionCategory


class BalloonRegionReference(BaseModel):
    region_id: str
    bbox: BoundingBox
    category: TextRegionCategory = "unknown"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("region_id")
    @classmethod
    def require_region_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("region_id must not be empty")
        return value


class BalloonCandidateEvidence(BaseModel):
    candidate_id: str
    region_id: str
    source: CandidateSource
    text: str
    ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    visual_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    candidate_supported: bool | None = None
    semantic_type: TextRegionCategory = "unknown"
    preprocessing: str | None = None
    bbox: BoundingBox | None = None
    similarity_to_candidates: dict[str, float] = Field(default_factory=dict)
    notes: str | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("candidate_id", "region_id")
    @classmethod
    def require_non_empty_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("candidate and region IDs must not be empty")
        return value


class BalloonAdjudicationInput(BaseModel):
    """Image-free metadata plus separately referenced local visual crops."""

    balloon_id: str
    balloon_bbox: BoundingBox
    balloon_image_path: str
    panel_id: str | None = None
    panel_bbox: BoundingBox | None = None
    panel_image_path: str | None = None
    region_references: list[BalloonRegionReference] = Field(min_length=1)
    candidates: list[BalloonCandidateEvidence] = Field(default_factory=list)

    @field_validator("balloon_id", "balloon_image_path")
    @classmethod
    def require_non_empty_path_or_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("balloon_id and balloon_image_path must not be empty")
        return value

    @field_validator("panel_image_path")
    @classmethod
    def validate_panel_image_path(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("panel_image_path must be non-empty when provided")
        return value

    @model_validator(mode="after")
    def validate_candidate_references(self) -> "BalloonAdjudicationInput":
        region_ids = [region.region_id for region in self.region_references]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("region references must be unique within a balloon")
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique within a balloon")
        if any(candidate.region_id not in set(region_ids) for candidate in self.candidates):
            raise ValueError("every candidate must reference a region in this balloon")
        return self


class BalloonCorrection(BaseModel):
    candidate_id: str
    from_text: str
    to_text: str


class AdjudicationDiagnostic(BaseModel):
    code: str
    message: str


class BalloonAdjudicationResult(BaseModel):
    balloon_id: str
    final_text: str = ""
    text_type: TextRegionCategory = "unknown"
    include_in_story: bool = False
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    corrections: list[BalloonCorrection] = Field(default_factory=list)
    notes: str | None = None
    diagnostics: list[AdjudicationDiagnostic] = Field(default_factory=list)

    @field_validator("balloon_id")
    @classmethod
    def require_balloon_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("balloon_id must not be empty")
        return value

    @model_validator(mode="after")
    def validate_included_text(self) -> "BalloonAdjudicationResult":
        if self.include_in_story and not self.final_text.strip():
            raise ValueError("final_text must be non-empty when include_in_story is true")
        return self