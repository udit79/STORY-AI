"""Validator for SequenceResolution.

Checks the fully resolved sequence against the hard constraints listed in the
task specification.  All validation errors are collected and returned; nothing
is silently repaired.

Call validate_sequence_resolution(resolution) and inspect the returned list of
ValidationError objects.  An empty list means the resolution passed all checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.schemas.resolution import SequenceResolution


# ---------------------------------------------------------------------------
# Validation error
# ---------------------------------------------------------------------------

@dataclass
class ValidationError:
    rule: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Individual rule implementations
# ---------------------------------------------------------------------------

def _rule_exactly_three_pages(resolution: SequenceResolution) -> list[ValidationError]:
    """All story balloons must belong to exactly {0, 1, 2} pages."""
    errors: list[ValidationError] = []
    page_indices = {b.page_index for b in resolution.ordered_balloons}
    unknown_pages = page_indices - {0, 1, 2}
    if unknown_pages:
        errors.append(ValidationError(
            rule="exactly_three_pages",
            message=f"Story balloons reference page indices outside {{0,1,2}}: {sorted(unknown_pages)}",
            details={"invalid_page_indices": sorted(unknown_pages)},
        ))
    return errors


def _rule_all_page_indices_valid(resolution: SequenceResolution) -> list[ValidationError]:
    """Every balloon must have a page_index in [0, 2]."""
    errors: list[ValidationError] = []
    for balloon in resolution.ordered_balloons + resolution.excluded_balloons:
        if balloon.page_index not in (0, 1, 2):
            errors.append(ValidationError(
                rule="all_page_indices_valid",
                message=(
                    f"Balloon {balloon.balloon_id!r} has invalid page_index={balloon.page_index}."
                ),
                details={"balloon_id": balloon.balloon_id, "page_index": balloon.page_index},
            ))
    return errors


def _rule_no_duplicate_balloon(resolution: SequenceResolution) -> list[ValidationError]:
    """No balloon_id may appear more than once in ordered_balloons."""
    errors: list[ValidationError] = []
    seen: dict[str, int] = {}
    for balloon in resolution.ordered_balloons:
        if balloon.balloon_id in seen:
            errors.append(ValidationError(
                rule="no_duplicate_balloon",
                message=f"Balloon {balloon.balloon_id!r} appears more than once in ordered output.",
                details={
                    "balloon_id": balloon.balloon_id,
                    "first_position": seen[balloon.balloon_id],
                    "duplicate_position": balloon.sequence_position,
                },
            ))
        else:
            seen[balloon.balloon_id] = balloon.sequence_position
    return errors


def _rule_page_order_monotonic(resolution: SequenceResolution) -> list[ValidationError]:
    """Page indices must be non-decreasing in sequence_position order."""
    errors: list[ValidationError] = []
    prev_page = -1
    for balloon in sorted(resolution.ordered_balloons, key=lambda b: b.sequence_position):
        if balloon.page_index < prev_page:
            errors.append(ValidationError(
                rule="page_order_monotonic",
                message=(
                    f"Balloon {balloon.balloon_id!r} at position {balloon.sequence_position} "
                    f"has page_index={balloon.page_index} which precedes "
                    f"previous page_index={prev_page}."
                ),
                details={
                    "balloon_id": balloon.balloon_id,
                    "sequence_position": balloon.sequence_position,
                    "page_index": balloon.page_index,
                    "previous_page_index": prev_page,
                },
            ))
        prev_page = balloon.page_index
    return errors


def _rule_story_excluded_balloons_absent(resolution: SequenceResolution) -> list[ValidationError]:
    """Story-excluded balloons (include_in_story=False) must not appear in ordered_balloons."""
    errors: list[ValidationError] = []
    for balloon in resolution.ordered_balloons:
        if not balloon.include_in_story:
            errors.append(ValidationError(
                rule="story_excluded_balloons_absent",
                message=(
                    f"Balloon {balloon.balloon_id!r} has include_in_story=False "
                    "but appears in ordered_balloons."
                ),
                details={"balloon_id": balloon.balloon_id, "page_index": balloon.page_index},
            ))
    return errors


def _rule_every_balloon_has_source_id(resolution: SequenceResolution) -> list[ValidationError]:
    """Every balloon must have a non-empty balloon_id."""
    errors: list[ValidationError] = []
    for balloon in resolution.ordered_balloons:
        if not balloon.balloon_id or not balloon.balloon_id.strip():
            errors.append(ValidationError(
                rule="every_balloon_has_source_id",
                message=f"Balloon at position {balloon.sequence_position} has empty balloon_id.",
                details={"sequence_position": balloon.sequence_position},
            ))
    return errors


def _rule_text_exists_for_story_balloons(resolution: SequenceResolution) -> list[ValidationError]:
    """Every story balloon must have non-empty text."""
    errors: list[ValidationError] = []
    for balloon in resolution.ordered_balloons:
        if not balloon.text or not balloon.text.strip():
            errors.append(ValidationError(
                rule="text_exists_for_story_balloons",
                message=(
                    f"Balloon {balloon.balloon_id!r} (page {balloon.page_index}) "
                    "is a story balloon but has empty text."
                ),
                details={"balloon_id": balloon.balloon_id, "page_index": balloon.page_index},
            ))
    return errors


def _rule_speaker_references_valid_character(
    resolution: SequenceResolution,
    known_character_ids: set[str],
) -> list[ValidationError]:
    """speaker_character_instance_id must refer to a known CharacterInstance (when set)."""
    errors: list[ValidationError] = []
    for balloon in resolution.ordered_balloons:
        cid = balloon.speaker_character_instance_id
        if cid is not None and cid not in known_character_ids:
            errors.append(ValidationError(
                rule="speaker_references_valid_character",
                message=(
                    f"Balloon {balloon.balloon_id!r} speaker_character_instance_id={cid!r} "
                    "does not exist in any page's character list."
                ),
                details={"balloon_id": balloon.balloon_id, "character_instance_id": cid},
            ))
    return errors


def _rule_speaker_label_maps_to_valid_identity(
    resolution: SequenceResolution,
) -> list[ValidationError]:
    """speaker_label (when set and not NARRATION) must map to a valid identity in identities."""
    errors: list[ValidationError] = []
    valid_labels = {ident.label for ident in resolution.identities}
    valid_labels.add("NARRATION")
    for balloon in resolution.ordered_balloons:
        label = balloon.speaker_label
        if label is not None and label not in valid_labels:
            errors.append(ValidationError(
                rule="speaker_label_maps_to_valid_identity",
                message=(
                    f"Balloon {balloon.balloon_id!r} speaker_label={label!r} "
                    "does not correspond to any identity label or NARRATION."
                ),
                details={"balloon_id": balloon.balloon_id, "speaker_label": label},
            ))
    return errors


def _rule_no_identity_label_conflicts(resolution: SequenceResolution) -> list[ValidationError]:
    """The same identity_id must not appear with different labels across balloons."""
    errors: list[ValidationError] = []
    identity_to_labels: dict[str, set[str]] = {}
    for balloon in resolution.ordered_balloons:
        iid = balloon.speaker_identity_id
        lbl = balloon.speaker_label
        if iid is not None and lbl is not None:
            identity_to_labels.setdefault(iid, set()).add(lbl)
    for iid, labels in identity_to_labels.items():
        if len(labels) > 1:
            errors.append(ValidationError(
                rule="no_identity_label_conflicts",
                message=(
                    f"identity_id={iid!r} has multiple labels: {sorted(labels)}. "
                    "Same identity must always map to the same label."
                ),
                details={"identity_id": iid, "labels": sorted(labels)},
            ))
    return errors


def _rule_sequence_labels_deterministic(resolution: SequenceResolution) -> list[ValidationError]:
    """Check that identity labels appear in the identities list with unique assignments."""
    errors: list[ValidationError] = []
    label_to_ids: dict[str, list[str]] = {}
    for ident in resolution.identities:
        label_to_ids.setdefault(ident.label, []).append(ident.identity_id)
    for label, ids in label_to_ids.items():
        if label != "NARRATION" and len(ids) > 1:
            errors.append(ValidationError(
                rule="sequence_labels_deterministic",
                message=(
                    f"Label {label!r} is assigned to multiple identity IDs: {sorted(ids)}. "
                    "Each label must correspond to exactly one identity."
                ),
                details={"label": label, "identity_ids": sorted(ids)},
            ))
    return errors


def _rule_no_output_references_different_sequence(
    resolution: SequenceResolution,
    sequence_id: str,
) -> list[ValidationError]:
    """Resolution sequence_id must match the provided sequence_id."""
    errors: list[ValidationError] = []
    if resolution.sequence_id != sequence_id:
        errors.append(ValidationError(
            rule="no_output_references_different_sequence",
            message=(
                f"SequenceResolution.sequence_id={resolution.sequence_id!r} "
                f"does not match expected sequence_id={sequence_id!r}."
            ),
            details={
                "resolution_sequence_id": resolution.sequence_id,
                "expected_sequence_id": sequence_id,
            },
        ))
    return errors


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_sequence_resolution(
    resolution: SequenceResolution,
    *,
    known_character_ids: set[str] | None = None,
    expected_sequence_id: str | None = None,
) -> list[ValidationError]:
    """Run all validation rules over a SequenceResolution.

    Args:
        resolution:            The SequenceResolution to validate.
        known_character_ids:   Set of valid CharacterInstance IDs from the pages.
                               When None, the speaker-reference check is skipped.
        expected_sequence_id:  When provided, validates that resolution.sequence_id
                               matches.  When None, cross-sequence check is skipped.

    Returns:
        List of ValidationError objects.  Empty list = all checks passed.
    """
    errors: list[ValidationError] = []

    errors.extend(_rule_exactly_three_pages(resolution))
    errors.extend(_rule_all_page_indices_valid(resolution))
    errors.extend(_rule_no_duplicate_balloon(resolution))
    errors.extend(_rule_page_order_monotonic(resolution))
    errors.extend(_rule_story_excluded_balloons_absent(resolution))
    errors.extend(_rule_every_balloon_has_source_id(resolution))
    errors.extend(_rule_text_exists_for_story_balloons(resolution))
    errors.extend(_rule_no_identity_label_conflicts(resolution))
    errors.extend(_rule_sequence_labels_deterministic(resolution))
    errors.extend(_rule_speaker_label_maps_to_valid_identity(resolution))

    if known_character_ids is not None:
        errors.extend(_rule_speaker_references_valid_character(resolution, known_character_ids))

    if expected_sequence_id is not None:
        errors.extend(_rule_no_output_references_different_sequence(resolution, expected_sequence_id))

    return errors
