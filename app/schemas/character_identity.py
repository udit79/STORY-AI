"""Typed contracts for sequence-level cross-page character identity."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CharacterPairEvidence(BaseModel):
    """Visual and contextual evidence for two CharacterInstances being the same person."""

    first_character_id: str
    second_character_id: str

    # --- visual similarity scores [0, 1] ---
    full_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    face_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    body_similarity: float | None = Field(default=None, ge=0.0, le=1.0)

    # --- combined evidence ---
    combined_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    # --- context ---
    same_page: bool = False
    first_page_index: int
    second_page_index: int
    method: Literal[
        "visual_embedding",
        "face_embedding",
        "body_embedding",
        "combined_embedding",
        "fallback_geometry",
        "injected",
    ] = "combined_embedding"
    evidence: dict[str, Any] = Field(default_factory=dict)
    diagnostics: list[str] = Field(default_factory=list)


class IdentityMember(BaseModel):
    character_id: str
    page_index: int
    panel_id: str | None = None


class CharacterIdentityCluster(BaseModel):
    """One resolved cross-page identity: a set of page-scoped character instances."""

    identity_id: str          # internal cluster identifier, stable within sequence
    label: str                # anonymous label: "A", "B", "C", …
    members: list[IdentityMember]
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    state: Literal["matched", "unmatched", "ambiguous"] = "matched"
    evidence_summary: dict[str, Any] = Field(default_factory=dict)


class IdentityDiagnostic(BaseModel):
    component: str
    code: str
    message: str
    character_ids: list[str] = Field(default_factory=list)


class SequenceCharacterIdentity(BaseModel):
    """Sequence-level cross-page character identity result."""

    sequence_id: str

    clusters: list[CharacterIdentityCluster]

    # flat map: character_instance_id → identity_id  (or None if unresolved)
    character_to_identity: dict[str, str | None]

    # flat map: character_instance_id → anonymous label  (or None)
    character_to_label: dict[str, str | None]

    # pairwise evidence that drove the graph
    pair_evidence: list[CharacterPairEvidence] = Field(default_factory=list)

    diagnostics: list[IdentityDiagnostic] = Field(default_factory=list)
