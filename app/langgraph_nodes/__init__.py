"""LangGraph node boundary for sequence consistency resolution.

This module defines the LangGraph-compatible node that invokes the
SequenceConsistencyResolver.

Architecture:
    LangGraph state
        ↓
    sequence_consistency_node(state)
        ↓
    SequenceConsistencyResolver.resolve(...)
        ↓
    SequenceResolution (stored back into state)

The node is intentionally thin.  Complex reconciliation logic lives in
SequenceConsistencyResolver, not here.

Usage:
    from app.langgraph_nodes.sequence_consistency import (
        SequenceConsistencyState,
        sequence_consistency_node,
    )

    # Build a LangGraph StateGraph, add the node:
    graph.add_node("sequence_consistency", sequence_consistency_node)
"""

from __future__ import annotations

from typing import Any, TypedDict

from app.schemas.adjudication import BalloonAdjudicationResult
from app.schemas.character_identity import SequenceCharacterIdentity
from app.schemas.page import PageRepresentation
from app.schemas.reading_order import SequenceReadingOrder
from app.schemas.resolution import SequenceResolution
from app.sequence_resolver import SequenceConsistencyResolver
from app.speaker_grounding import PageSpeakerGroundingResult


# ---------------------------------------------------------------------------
# State contract
# ---------------------------------------------------------------------------

class SequenceConsistencyState(TypedDict, total=False):
    """LangGraph state keys consumed and produced by the consistency node.

    All fields are optional (``total=False``) so the state can be built
    incrementally by preceding nodes.

    Consumed:
        sequence_id            — str
        pages                  — list[PageRepresentation]
        adjudication_results   — list[BalloonAdjudicationResult]
        reading_order          — SequenceReadingOrder
        speaker_results        — list[PageSpeakerGroundingResult]
        identity               — SequenceCharacterIdentity | None

    Produced / updated:
        sequence_resolution    — SequenceResolution
    """

    sequence_id: str
    pages: list[PageRepresentation]
    adjudication_results: list[BalloonAdjudicationResult]
    reading_order: SequenceReadingOrder
    speaker_results: list[PageSpeakerGroundingResult]
    identity: SequenceCharacterIdentity | None

    # Output
    sequence_resolution: SequenceResolution


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def sequence_consistency_node(
    state: dict[str, Any],
    *,
    resolver: SequenceConsistencyResolver | None = None,
) -> dict[str, Any]:
    """LangGraph node that resolves sequence consistency.

    Args:
        state:    LangGraph state dict.  Must contain the keys listed in
                  SequenceConsistencyState (except ``sequence_resolution``).
        resolver: Optional pre-built SequenceConsistencyResolver.
                  Defaults to a new instance with default settings.

    Returns:
        Partial state update containing only ``sequence_resolution``.
        LangGraph merges this into the existing state.

    Raises:
        KeyError: if a mandatory state key is missing.
        TypeError: if a state value has the wrong type.
    """
    # --- validate mandatory keys ---
    missing = [
        key for key in ("sequence_id", "pages", "adjudication_results",
                        "reading_order", "speaker_results")
        if key not in state
    ]
    if missing:
        raise KeyError(
            f"sequence_consistency_node: missing required state keys: {missing}"
        )

    sequence_id: str = state["sequence_id"]
    pages: list[PageRepresentation] = state["pages"]
    adjudication_results: list[BalloonAdjudicationResult] = state["adjudication_results"]
    reading_order: SequenceReadingOrder = state["reading_order"]
    speaker_results: list[PageSpeakerGroundingResult] = state["speaker_results"]
    identity: SequenceCharacterIdentity | None = state.get("identity")

    if resolver is None:
        resolver = SequenceConsistencyResolver()

    resolution = resolver.resolve(
        sequence_id=sequence_id,
        pages=pages,
        adjudication_results=adjudication_results,
        reading_order=reading_order,
        speaker_results=speaker_results,
        identity=identity,
    )

    return {"sequence_resolution": resolution}
