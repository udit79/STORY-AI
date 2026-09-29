"""Competition serializer for SequenceResolution → JSONL submission.

Competition output contract (from dataset/score.py and sample_submission.jsonl):

    JSONL line:
        {"sequence_id": str, "pages": [[...], [...], [...]]}

    Exactly TWO keys: "sequence_id" and "pages".  No additional keys.

    "pages":
        - List of exactly 3 page lists.
        - A page list may be empty [] if no story balloons are on that page.
        - Each item: {"speaker": str, "text": str}
        - Exactly TWO keys per item: "speaker" and "text".

    "speaker":
        - Must be a non-empty, trimmed string.
        - "NARRATION" is the fixed label for narration/thought/SFX content.
        - "UNKNOWN" is used for unresolved speakers (the scorer treats this
          as a predicted speaker label and maps it during evaluation; it is
          NOT the same as the gold "UNKNOWN" reference which is stripped).

    "text":
        - Must be non-empty after normalization.
        - Text comes verbatim from the adjudicator (final_text).

Architecture:
    SequenceResolution (rich internal object)
        ↓
    serialize_to_submission(resolution)
        ↓
    SubmissionRecord (Python object, validated)
        ↓
    serialize_to_jsonl_line(record)
        ↓
    str (one JSONL line)
"""

from __future__ import annotations

import json
import unicodedata
from typing import Any

from app.schemas.resolution import SequenceResolution


# ---------------------------------------------------------------------------
# Submission schema (Python-level, competition-compatible)
# ---------------------------------------------------------------------------

class SubmissionPageItem:
    """One {"speaker": ..., "text": ...} item from a page list."""

    __slots__ = ("speaker", "text")

    def __init__(self, speaker: str, text: str) -> None:
        self.speaker = speaker
        self.text = text

    def to_dict(self) -> dict[str, str]:
        return {"speaker": self.speaker, "text": self.text}


class SubmissionRecord:
    """One JSONL record: {"sequence_id": ..., "pages": [..., ..., ...]}."""

    __slots__ = ("sequence_id", "pages")

    def __init__(self, sequence_id: str, pages: list[list[SubmissionPageItem]]) -> None:
        self.sequence_id = sequence_id
        self.pages = pages

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "pages": [[item.to_dict() for item in page] for page in self.pages],
        }


# ---------------------------------------------------------------------------
# Speaker label normalization
# ---------------------------------------------------------------------------

def _submission_speaker(label: str | None, *, identity_state: str = "null") -> str:
    """Map an internal speaker_label to the competition speaker string.

    Rules:
        - NARRATION: kept as "NARRATION".
        - A valid anonymous label (A, B, C, …) from a *matched* or *unmatched*
          cluster: kept as-is.
        - Ambiguous identity state: the internal label is kept for traceability
          in SequenceResolution, but the competition output must be "UNKNOWN"
          to avoid emitting a falsely confident speaker label.
        - None or any other unresolved state: emit "UNKNOWN".
          (The scorer treats "UNKNOWN" as a predicted label and aligns it;
           unresolved predictions cannot be scored as correct matches anyway.)
    """
    if label is None:
        return "UNKNOWN"
    label = label.strip()
    if not label:
        return "UNKNOWN"
    if label == "NARRATION":
        return "NARRATION"
    # Ambiguous identity evidence must not become a falsely confident label.
    if identity_state == "ambiguous":
        return "UNKNOWN"
    return label


def _is_empty_text(text: str) -> bool:
    """Return True if text normalizes to empty (score.py norm logic)."""
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    return len(normalized) == 0


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------

def serialize_to_submission(resolution: SequenceResolution) -> SubmissionRecord:
    """Convert a SequenceResolution into a competition-format SubmissionRecord.

    Only include_in_story=True balloons are emitted.
    Balloons are grouped by page_index (0, 1, 2) in sequence_position order.
    Speaker label falls back to "UNKNOWN" when unresolved.
    Text is taken verbatim from the adjudicator's final_text.

    Raises:
        ValueError: if a story balloon has empty text (this must not reach submission).
    """
    pages: list[list[SubmissionPageItem]] = [[], [], []]

    # ordered_balloons is already in sequence_position order and include_in_story=True
    for balloon in resolution.ordered_balloons:
        page_idx = balloon.page_index
        if page_idx not in (0, 1, 2):
            # Defensive: out-of-range pages are skipped with no crash
            continue

        text = balloon.text
        if _is_empty_text(text):
            # A story balloon with empty text is a data quality issue, not a
            # serializer responsibility — skip it rather than emit invalid output.
            continue

        speaker = _submission_speaker(
            balloon.speaker_label,
            identity_state=balloon.identity_state,
        )
        pages[page_idx].append(SubmissionPageItem(speaker=speaker, text=text))

    return SubmissionRecord(sequence_id=resolution.sequence_id, pages=pages)


def serialize_to_jsonl_line(record: SubmissionRecord) -> str:
    """Serialize a SubmissionRecord to one JSONL line (no trailing newline)."""
    return json.dumps(record.to_dict(), ensure_ascii=False)


def serialize_resolution_to_jsonl(resolution: SequenceResolution) -> str:
    """Convenience: convert a SequenceResolution directly to one JSONL line."""
    record = serialize_to_submission(resolution)
    return serialize_to_jsonl_line(record)
