"""Typed contracts for the Sequence Consistency Resolver.

The resolver is the global reconciliation layer that combines:
    - balloon adjudication results
    - reading order
    - page-scoped speaker CharacterInstance
    - cross-page identity cluster
    - anonymous sequence character label

into one internally consistent SequenceResolution.

Internal representation is deliberately richer than the competition output;
the competition serializer is a separate layer.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class ResolverDiagnostic(BaseModel):
    """One structured problem detected during reconciliation."""

    code: str
    message: str

    # Optional provenance fields
    balloon_id: str | None = None
    page_index: int | None = None
    panel_id: str | None = None
    character_instance_id: str | None = None
    identity_id: str | None = None

    severity: Literal["info", "warning", "error"] = "warning"


# ---------------------------------------------------------------------------
# Per-balloon resolved entry
# ---------------------------------------------------------------------------

class ResolvedBalloon(BaseModel):
    """One balloon's fully resolved entry in the sequence narrative.

    Provenance is preserved so every field is traceable back to the source.
    """

    # --- position in the global sequence ---
    sequence_position: int   # 0-based, from SequenceReadingOrder

    # --- origin ---
    page_index: int
    panel_id: str | None = None
    balloon_id: str          # original Balloon.id from PageRepresentation

    # --- adjudication results (verbatim from BalloonAdjudicationResult) ---
    text: str                # final_text from adjudicator
    text_type: str           # TextRegionCategory from adjudicator
    include_in_story: bool   # from adjudicator

    # --- speaker provenance ---
    # These three fields form a traceability chain:
    #   speaker_character_instance_id → speaker_identity_id → speaker_label
    speaker_character_instance_id: str | None = None  # SpeakerDecision result
    speaker_identity_id: str | None = None             # from character_to_identity map
    speaker_label: str | None = None                   # anonymous A/B/C or "NARRATION"

    # --- identity resolution state ---
    identity_state: Literal["matched", "unmatched", "ambiguous", "null"] = "null"

    # --- diagnostics for this specific balloon ---
    diagnostics: list[ResolverDiagnostic] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-identity summary
# ---------------------------------------------------------------------------

class ResolvedIdentity(BaseModel):
    """One identity cluster as seen by the resolver."""

    identity_id: str
    label: str              # anonymous label (Character A, Character B, …) or "NARRATION"
    state: Literal["matched", "unmatched", "ambiguous"]
    member_character_ids: list[str] = Field(default_factory=list)
    page_indices: list[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level sequence resolution
# ---------------------------------------------------------------------------

class SequenceResolution(BaseModel):
    """Fully reconciled internal representation of one 3-page sequence.

    This is the rich internal object.  It is NOT directly serialized to the
    competition format.  The competition serializer consumes this object.
    """

    sequence_id: str

    # Ordered story balloons (include_in_story=True), in reading-order position.
    # Non-story balloons (SFX, signs, metadata) are excluded here.
    ordered_balloons: list[ResolvedBalloon] = Field(default_factory=list)

    # Identity clusters present in this sequence
    identities: list[ResolvedIdentity] = Field(default_factory=list)

    # Balloons that were adjudicated but skipped from final output
    # (include_in_story=False), preserved for traceability.
    excluded_balloons: list[ResolvedBalloon] = Field(default_factory=list)

    # Items that could not be fully resolved
    unresolved_items: list[dict[str, Any]] = Field(default_factory=list)

    # Sequence-level diagnostics
    diagnostics: list[ResolverDiagnostic] = Field(default_factory=list)
