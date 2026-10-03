"""Sequence Consistency Resolver.

This is the GLOBAL RECONCILIATION layer.  It does NOT run OCR, re-order
balloons, re-detect characters, or re-match identities.  It consumes the
already-computed outputs from all upstream components and produces one
internally consistent SequenceResolution.

Precedence for decisions (descending):
    1. Hard schema/ownership constraints (page ordering, ID existence)
    2. Explicit adjudicator result (text, include_in_story, text_type)
    3. Explicit reading-order result (sequence_position)
    4. Explicit speaker result (selected_character_instance_id)
    5. Identity cluster mapping (character_to_identity, character_to_label)
    6. Ambiguity diagnostics

No component may overwrite another's source result without recording why.

Competition output contract (from score.py / sample_submission.jsonl):
    - JSONL line: {"sequence_id": str, "pages": [[...], [...], [...]]}
    - page item: {"speaker": str, "text": str}  — EXACTLY these two keys
    - speaker:   non-empty trimmed string
    - NARRATION: narration/thought/sound_effect bubbles use "NARRATION"
    - unresolved speakers: use "UNKNOWN" (competition strips it as a valid
      reference field; they won't be scored against ground-truth speakers)
"""

from __future__ import annotations

import re
import string
from typing import Any

from app.schemas.adjudication import BalloonAdjudicationResult
from app.schemas.character_identity import (
    CharacterIdentityCluster,
    IdentityMember,
    SequenceCharacterIdentity,
)
from app.schemas.page import CharacterInstance, PageRepresentation
from app.schemas.reading_order import SequenceReadingOrder
from app.schemas.resolution import (
    ResolvedBalloon,
    ResolvedIdentity,
    ResolverDiagnostic,
    SequenceResolution,
)
from app.speaker_grounding import PageSpeakerGroundingResult, SpeakerDecision


# ---------------------------------------------------------------------------
# Narration-type text categories
# ---------------------------------------------------------------------------

# These text_type values map to the fixed "NARRATION" speaker label in the
# competition output.  The adjudicator has already determined text_type.
_NARRATION_TYPES: frozenset[str] = frozenset(
    [
        "narration",
        "thought",
        "sound_effect",
        "sign",
        "title",
        "metadata",
        "document_text",
        "vocalisation",
    ]
)

# Speaker-attributed types: the speaker must come from identity resolution.
_DIALOGUE_TYPES: frozenset[str] = frozenset(["dialogue", "unknown"])


def _anonymous_label(label: str | None) -> str | None:
    """Normalize legacy cluster labels without changing real character names."""
    if label is not None and re.fullmatch(r"[A-Z]+", label):
        return f"Character {label}"
    return label


def _alphabetic_label(index: int) -> str:
    """0→A, 1→B, …, 25→Z, 26→AA."""
    result = ""
    index += 1
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        result = string.ascii_uppercase[remainder] + result
    return f"Character {result}"


def _ensure_grounded_singletons(
    identity: SequenceCharacterIdentity | None,
    pages: list[PageRepresentation],
) -> SequenceCharacterIdentity | None:
    """Materialize deterministic singleton identities for grounded characters.

    The normal identity resolver already creates these clusters.  This small
    reconciliation guard covers partial/legacy identity results where a
    detected character is present in the flat map with ``None`` (or omitted),
    so a successfully grounded anonymous speaker is not converted to UNKNOWN.
    Existing cluster IDs and labels are preserved; new labels are appended in
    sequence appearance order.
    """
    if identity is None:
        return None

    character_index = {
        character.id: (page.page_index, character)
        for page in pages
        for character in page.characters
    }
    missing_ids = [
        character_id
        for character_id in character_index
        if identity.character_to_identity.get(character_id) is None
    ]
    if not missing_ids:
        return identity

    missing_ids.sort(
        key=lambda character_id: (
            character_index[character_id][0],
            character_index[character_id][1].bbox.y1,
            character_index[character_id][1].bbox.x1,
            character_id,
        )
    )
    clusters = list(identity.clusters)
    character_to_identity = dict(identity.character_to_identity)
    character_to_label = dict(identity.character_to_label)
    used_identity_ids = {cluster.identity_id for cluster in clusters}
    used_labels = {cluster.label for cluster in clusters}
    next_identity_number = max(
        [
            int(match.group(1))
            for match in (
                re.fullmatch(r"identity-(\d+)", identity_id)
                for identity_id in used_identity_ids
            )
            if match is not None
        ],
        default=0,
    ) + 1
    next_label_index = 0
    while _alphabetic_label(next_label_index) in used_labels:
        next_label_index += 1

    for character_id in missing_ids:
        page_index, character = character_index[character_id]
        while f"identity-{next_identity_number:03d}" in used_identity_ids:
            next_identity_number += 1
        identity_id = f"identity-{next_identity_number:03d}"
        label = _alphabetic_label(next_label_index)
        while label in used_labels:
            next_label_index += 1
            label = _alphabetic_label(next_label_index)
        clusters.append(CharacterIdentityCluster(
            identity_id=identity_id,
            label=label,
            members=[IdentityMember(
                character_id=character_id,
                page_index=page_index,
                panel_id=character.panel_id,
            )],
            confidence=1.0,
            state="unmatched",
        ))
        character_to_identity[character_id] = identity_id
        character_to_label[character_id] = label
        used_identity_ids.add(identity_id)
        used_labels.add(label)
        next_identity_number += 1
        next_label_index += 1

    return identity.model_copy(update={
        "clusters": clusters,
        "character_to_identity": character_to_identity,
        "character_to_label": character_to_label,
    })


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_adjudication_index(
    adjudication_results: list[BalloonAdjudicationResult],
) -> dict[str, BalloonAdjudicationResult]:
    index: dict[str, BalloonAdjudicationResult] = {}
    for result in adjudication_results:
        if result.balloon_id in index:
            # Duplicate balloon_id: keep the first (earlier is more authoritative)
            pass
        else:
            index[result.balloon_id] = result
    return index


def _build_speaker_index(
    speaker_results: list[PageSpeakerGroundingResult],
) -> dict[str, SpeakerDecision]:
    """Flatten per-page speaker decisions into a balloon_id → SpeakerDecision map."""
    index: dict[str, SpeakerDecision] = {}
    for page_result in speaker_results:
        for decision in page_result.decisions:
            if decision.balloon_id not in index:
                index[decision.balloon_id] = decision
    return index


def _build_page_character_index(
    pages: list[PageRepresentation],
) -> dict[str, CharacterInstance]:
    """Map character_instance_id → CharacterInstance across all pages."""
    index: dict[str, CharacterInstance] = {}
    for page in pages:
        for char in page.characters:
            if char.id not in index:
                index[char.id] = char
    return index


def _build_balloon_page_index(
    pages: list[PageRepresentation],
) -> dict[str, int]:
    """Map balloon_id → page_index."""
    index: dict[str, int] = {}
    for page in pages:
        for balloon in page.balloons:
            if balloon.id not in index:
                index[balloon.id] = page.page_index
    return index


def _build_balloon_panel_index(
    pages: list[PageRepresentation],
) -> dict[str, str | None]:
    """Map balloon_id → panel_id (or None)."""
    index: dict[str, str | None] = {}
    for page in pages:
        for balloon in page.balloons:
            if balloon.id not in index:
                index[balloon.id] = balloon.panel_id
    return index


# ---------------------------------------------------------------------------
# Speaker label resolution
# ---------------------------------------------------------------------------

def _resolve_speaker_label(
    balloon_id: str,
    text_type: str,
    speaker_char_id: str | None,
    identity: SequenceCharacterIdentity | None,
    char_index: dict[str, CharacterInstance],
    diagnostics: list[ResolverDiagnostic],
) -> tuple[str | None, str | None, str]:
    """Return (identity_id, label, identity_state) for a balloon.

    identity_state is one of: "matched", "unmatched", "ambiguous", "null"
    """
    # Narration-type: label is always NARRATION, no character instance needed
    if text_type in _NARRATION_TYPES:
        return None, "NARRATION", "null"

    # No speaker resolved by grounding
    if speaker_char_id is None:
        diagnostics.append(ResolverDiagnostic(
            code="no_speaker_resolved",
            message=f"Balloon {balloon_id!r} has no grounded speaker; speaker_label=None.",
            balloon_id=balloon_id,
            severity="warning",
        ))
        return None, None, "null"

    # Speaker references a character that does not exist
    if speaker_char_id not in char_index:
        diagnostics.append(ResolverDiagnostic(
            code="invalid_speaker_character_reference",
            message=(
                f"Balloon {balloon_id!r} speaker_character_instance_id={speaker_char_id!r} "
                "does not reference a known CharacterInstance."
            ),
            balloon_id=balloon_id,
            character_instance_id=speaker_char_id,
            severity="error",
        ))
        return None, None, "null"

    if identity is None:
        # No identity resolution available at all
        diagnostics.append(ResolverDiagnostic(
            code="no_identity_result",
            message=(
                f"Balloon {balloon_id!r} speaker={speaker_char_id!r}: "
                "no SequenceCharacterIdentity provided; label unresolved."
            ),
            balloon_id=balloon_id,
            character_instance_id=speaker_char_id,
            severity="warning",
        ))
        return None, None, "unmatched"

    identity_id = identity.character_to_identity.get(speaker_char_id)
    label = _anonymous_label(identity.character_to_label.get(speaker_char_id))

    if identity_id is None:
        # Character exists but was not assigned to any cluster
        diagnostics.append(ResolverDiagnostic(
            code="speaker_identity_unresolved",
            message=(
                f"Balloon {balloon_id!r} speaker={speaker_char_id!r} has no identity cluster."
            ),
            balloon_id=balloon_id,
            character_instance_id=speaker_char_id,
            severity="warning",
        ))
        return None, None, "unmatched"

    if label is None:
        # Has an identity_id but no label — abnormal
        diagnostics.append(ResolverDiagnostic(
            code="speaker_identity_label_missing",
            message=(
                f"Balloon {balloon_id!r} speaker={speaker_char_id!r} "
                f"has identity_id={identity_id!r} but no label."
            ),
            balloon_id=balloon_id,
            identity_id=identity_id,
            severity="error",
        ))
        return identity_id, None, "ambiguous"

    # Look up the cluster state
    cluster_state: str = "unmatched"
    for cluster in identity.clusters:
        if cluster.identity_id == identity_id:
            cluster_state = cluster.state
            break

    if cluster_state == "ambiguous":
        diagnostics.append(ResolverDiagnostic(
            code="speaker_identity_ambiguous",
            message=(
                f"Balloon {balloon_id!r} speaker={speaker_char_id!r} "
                f"maps to identity_id={identity_id!r} (label={label!r}) which is ambiguous. "
                "Label retained for traceability; do not treat as confident."
            ),
            balloon_id=balloon_id,
            character_instance_id=speaker_char_id,
            identity_id=identity_id,
            severity="warning",
        ))

    return identity_id, label, cluster_state


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------

def _check_identity_conflicts(
    resolved_balloons: list[ResolvedBalloon],
    identity: SequenceCharacterIdentity | None,
    diagnostics: list[ResolverDiagnostic],
) -> None:
    """Check: same identity cluster → same anonymous label (global consistency)."""
    if identity is None:
        return

    identity_to_labels: dict[str, set[str]] = {}
    for balloon in resolved_balloons:
        if balloon.speaker_identity_id is not None and balloon.speaker_label is not None:
            identity_to_labels.setdefault(balloon.speaker_identity_id, set()).add(
                balloon.speaker_label
            )

    for identity_id, labels in identity_to_labels.items():
        if len(labels) > 1:
            diagnostics.append(ResolverDiagnostic(
                code="identity_label_conflict",
                message=(
                    f"identity_id={identity_id!r} has inconsistent labels: {sorted(labels)}. "
                    "This indicates a bug in the identity resolver."
                ),
                identity_id=identity_id,
                severity="error",
            ))


def _check_same_page_identity_collapse(
    pages: list[PageRepresentation],
    identity: SequenceCharacterIdentity | None,
    diagnostics: list[ResolverDiagnostic],
) -> None:
    """Check: two same-page CharacterInstances should never share one identity."""
    if identity is None:
        return

    for page in pages:
        page_char_ids = {char.id for char in page.characters}
        for cluster in identity.clusters:
            member_ids = {m.character_id for m in cluster.members}
            overlap = page_char_ids & member_ids
            if len(overlap) > 1:
                diagnostics.append(ResolverDiagnostic(
                    code="same_page_identity_collapse",
                    message=(
                        f"Page {page.page_index}: identity_id={cluster.identity_id!r} "
                        f"has {len(overlap)} members from the same page: {sorted(overlap)}. "
                        "Same-page constraint violated."
                    ),
                    page_index=page.page_index,
                    identity_id=cluster.identity_id,
                    severity="error",
                ))


def _check_adjudication_coverage(
    reading_order: SequenceReadingOrder,
    adjudication_index: dict[str, BalloonAdjudicationResult],
    diagnostics: list[ResolverDiagnostic],
) -> None:
    """Check: each reading-order balloon must have an adjudication result."""
    for item in reading_order.ordered_items:
        if item.balloon_id not in adjudication_index:
            diagnostics.append(ResolverDiagnostic(
                code="missing_adjudication_result",
                message=(
                    f"Balloon {item.balloon_id!r} appears in reading order "
                    f"(page {item.page_index}, pos {item.sequence_position}) "
                    "but has no adjudication result."
                ),
                balloon_id=item.balloon_id,
                page_index=item.page_index,
                severity="error",
            ))


def _check_excluded_in_output(
    resolved_balloons: list[ResolvedBalloon],
    diagnostics: list[ResolverDiagnostic],
) -> None:
    """Sanity: no excluded balloon may appear in the ordered list."""
    for balloon in resolved_balloons:
        if not balloon.include_in_story:
            diagnostics.append(ResolverDiagnostic(
                code="excluded_balloon_in_story",
                message=(
                    f"Balloon {balloon.balloon_id!r} (include_in_story=False) "
                    "appeared in the ordered story output. This is a resolver bug."
                ),
                balloon_id=balloon.balloon_id,
                severity="error",
            ))


def _check_page_order_monotonic(
    ordered_balloons: list[ResolvedBalloon],
    diagnostics: list[ResolverDiagnostic],
) -> None:
    """Check: page indices must be non-decreasing in global order."""
    prev_page = -1
    for balloon in ordered_balloons:
        if balloon.page_index < prev_page:
            diagnostics.append(ResolverDiagnostic(
                code="page_order_violation",
                message=(
                    f"Balloon {balloon.balloon_id!r} at position {balloon.sequence_position} "
                    f"has page_index={balloon.page_index} which is less than "
                    f"previous page_index={prev_page}. Page ordering violated."
                ),
                balloon_id=balloon.balloon_id,
                page_index=balloon.page_index,
                severity="error",
            ))
        prev_page = balloon.page_index


def _check_duplicate_balloons(
    ordered_balloons: list[ResolvedBalloon],
    diagnostics: list[ResolverDiagnostic],
) -> None:
    """Check: no balloon_id appears more than once."""
    seen: set[str] = set()
    for balloon in ordered_balloons:
        if balloon.balloon_id in seen:
            diagnostics.append(ResolverDiagnostic(
                code="duplicate_balloon_in_sequence",
                message=f"Balloon {balloon.balloon_id!r} appears more than once in ordered output.",
                balloon_id=balloon.balloon_id,
                severity="error",
            ))
        seen.add(balloon.balloon_id)


def _check_identity_reference_validity(
    identity: SequenceCharacterIdentity | None,
    char_index: dict[str, CharacterInstance],
    diagnostics: list[ResolverDiagnostic],
) -> None:
    """Check: identity mapping references must point to existing CharacterInstances."""
    if identity is None:
        return
    for char_id in identity.character_to_identity:
        if char_id not in char_index:
            diagnostics.append(ResolverDiagnostic(
                code="identity_references_unknown_character",
                message=(
                    f"SequenceCharacterIdentity.character_to_identity contains "
                    f"{char_id!r} which is not a known CharacterInstance."
                ),
                character_instance_id=char_id,
                severity="error",
            ))


# ---------------------------------------------------------------------------
# Build resolved identities list
# ---------------------------------------------------------------------------

def _build_resolved_identities(
    identity: SequenceCharacterIdentity | None,
) -> list[ResolvedIdentity]:
    if identity is None:
        return []
    result: list[ResolvedIdentity] = []
    for cluster in identity.clusters:
        page_indices = sorted({m.page_index for m in cluster.members})
        result.append(ResolvedIdentity(
            identity_id=cluster.identity_id,
            label=_anonymous_label(cluster.label) or cluster.label,
            state=cluster.state,
            member_character_ids=[m.character_id for m in cluster.members],
            page_indices=page_indices,
        ))
    return result


# ---------------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------------

class SequenceConsistencyResolver:
    """Deterministic reconciliation of all upstream component outputs.

    Inputs:
        pages                  — 3 PageRepresentation (already processed)
        adjudication_results   — per-balloon BalloonAdjudicationResult list
        reading_order          — SequenceReadingOrder (already computed)
        speaker_results        — per-page PageSpeakerGroundingResult list
        identity               — SequenceCharacterIdentity (may be None)

    Output:
        SequenceResolution     — rich internal object

    The resolver records ALL contradictions as diagnostics and never silently
    discards or invents evidence.
    """

    def resolve(
        self,
        sequence_id: str,
        pages: list[PageRepresentation],
        adjudication_results: list[BalloonAdjudicationResult],
        reading_order: SequenceReadingOrder,
        speaker_results: list[PageSpeakerGroundingResult],
        identity: SequenceCharacterIdentity | None = None,
    ) -> SequenceResolution:
        """Produce a SequenceResolution from all upstream outputs.

        This method does NOT run any model, OCR, re-ordering, or inference.
        It only combines, validates, and reconciles the existing results.
        """
        diagnostics: list[ResolverDiagnostic] = []

        # --- build lookup indices ---
        adj_index = _build_adjudication_index(adjudication_results)
        speaker_index = _build_speaker_index(speaker_results)
        char_index = _build_page_character_index(pages)
        balloon_page_index = _build_balloon_page_index(pages)
        balloon_panel_index = _build_balloon_panel_index(pages)
        identity = _ensure_grounded_singletons(identity, pages)

        # --- pre-resolution consistency checks ---
        _check_adjudication_coverage(reading_order, adj_index, diagnostics)
        _check_identity_reference_validity(identity, char_index, diagnostics)
        _check_same_page_identity_collapse(pages, identity, diagnostics)

        # --- resolve each balloon in reading order ---
        ordered_balloons: list[ResolvedBalloon] = []
        excluded_balloons: list[ResolvedBalloon] = []
        unresolved_items: list[dict[str, Any]] = []

        for item in reading_order.ordered_items:
            balloon_id = item.balloon_id
            balloon_diags: list[ResolverDiagnostic] = []

            # --- must come from a known page ---
            page_index = balloon_page_index.get(balloon_id)
            if page_index is None:
                diagnostics.append(ResolverDiagnostic(
                    code="balloon_not_in_any_page",
                    message=(
                        f"Reading order references balloon {balloon_id!r} "
                        "which was not found in any PageRepresentation."
                    ),
                    balloon_id=balloon_id,
                    severity="error",
                ))
                unresolved_items.append({
                    "balloon_id": balloon_id,
                    "reason": "balloon_not_in_any_page",
                })
                continue

            panel_id = balloon_panel_index.get(balloon_id)

            # --- adjudication result (mandatory) ---
            adj = adj_index.get(balloon_id)
            if adj is None:
                # Already diagnosed above; create minimal unresolved entry
                unresolved_items.append({
                    "balloon_id": balloon_id,
                    "page_index": page_index,
                    "reason": "missing_adjudication_result",
                })
                continue

            # Validate page_index consistency: reading order item vs page lookup
            if item.page_index != page_index:
                balloon_diags.append(ResolverDiagnostic(
                    code="page_index_mismatch",
                    message=(
                        f"Balloon {balloon_id!r}: reading order says page={item.page_index} "
                        f"but balloon found on page={page_index}."
                    ),
                    balloon_id=balloon_id,
                    page_index=page_index,
                    severity="warning",
                ))

            # --- speaker grounding ---
            speaker_decision = speaker_index.get(balloon_id)
            speaker_char_id: str | None = None
            if speaker_decision is not None:
                speaker_char_id = speaker_decision.selected_character_instance_id

            # --- speaker → identity → label ---
            identity_id, label, identity_state = _resolve_speaker_label(
                balloon_id=balloon_id,
                text_type=adj.text_type,
                speaker_char_id=speaker_char_id,
                identity=identity,
                char_index=char_index,
                diagnostics=balloon_diags,
            )

            resolved = ResolvedBalloon(
                sequence_position=item.sequence_position,
                page_index=page_index,
                panel_id=panel_id,
                balloon_id=balloon_id,
                text=adj.final_text,
                text_type=adj.text_type,
                include_in_story=adj.include_in_story,
                speaker_character_instance_id=speaker_char_id,
                speaker_identity_id=identity_id,
                speaker_label=label,
                identity_state=identity_state,
                diagnostics=balloon_diags,
            )

            if adj.include_in_story:
                ordered_balloons.append(resolved)
            else:
                excluded_balloons.append(resolved)

        # --- post-resolution consistency checks ---
        _check_duplicate_balloons(ordered_balloons, diagnostics)
        _check_page_order_monotonic(ordered_balloons, diagnostics)
        _check_excluded_in_output(ordered_balloons, diagnostics)
        _check_identity_conflicts(ordered_balloons, identity, diagnostics)

        # --- build identity summaries ---
        resolved_identities = _build_resolved_identities(identity)

        return SequenceResolution(
            sequence_id=sequence_id,
            ordered_balloons=ordered_balloons,
            excluded_balloons=excluded_balloons,
            identities=resolved_identities,
            unresolved_items=unresolved_items,
            diagnostics=diagnostics,
        )
