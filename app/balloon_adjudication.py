"""Image-primary Qwen3-VL adjudication for grouped semantic balloons."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol

from app.models.local_runners import Qwen3VLLocalRunner
from app.schemas.adjudication import (
    AdjudicationDiagnostic,
    BalloonAdjudicationInput,
    BalloonAdjudicationResult,
    BalloonCandidateEvidence,
    BalloonCorrection,
    BalloonRegionReference,
)
from app.schemas.candidates import CandidateBank
from app.schemas.page import Balloon, Panel, TextRegion

ADJUDICATION_SYSTEM_PROMPT = """You are a careful manga balloon transcription adjudicator.
The first supplied image is the actual balloon crop and is the primary evidence.
If a second image is supplied, it is auxiliary panel context only. Never inspect
or infer from any other part of the page.

Follow this order:
1. Read the balloon image independently before considering any OCR hypotheses.
2. Identify every visible textual line that belongs to this one balloon.
3. Transcribe exact visible wording and punctuation; do not paraphrase, normalize,
   rewrite spelling, or make unusual wording more natural.
4. Preserve punctuation exactly, including ?, !, !!, ellipses, and hyphens.
5. Compare each separately supplied candidate transcription and resolve conflicts
   using the balloon image. Candidates are hypotheses, not ground truth. Do not
   copy a candidate merely because multiple sources agree.
6. Produce the final transcription, text type, and story-inclusion decision.

Do not invent words, fill in implied dialogue, use outside knowledge, or merge
text from another balloon. Read every visible line in this balloon. A punctuation-
only utterance such as "!", "...", or "?!" can be valid story text. Sound effects,
signs, titles, metadata, and document text should be classified accordingly and
normally excluded from story text, but do not discard them before classification.

If the image is genuinely unreadable, do not guess: return final_text as an empty
string, text_type "unknown", include_in_story false, and explain the uncertainty.
Return strict JSON only, with keys final_text, text_type, include_in_story,
confidence, corrections, and notes. Confidence is an uncalibrated model estimate.
Do not include chain-of-thought; notes should be brief evidence or uncertainty.
"""


class BalloonAdjudicator(Protocol):
    def adjudicate(
        self,
        input: BalloonAdjudicationInput,
    ) -> BalloonAdjudicationResult: ...


def _similarity_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def build_balloon_adjudication_input(
    balloon: Balloon,
    candidate_bank: CandidateBank,
    text_regions: Sequence[TextRegion],
    balloon_image_path: str | Path,
    *,
    panel: Panel | None = None,
    panel_image_path: str | Path | None = None,
) -> BalloonAdjudicationInput:
    """Join all independent candidate records referenced by one grouped balloon."""
    if not balloon.text_region_ids:
        raise ValueError("a balloon must reference at least one CTD text region")
    region_by_id: dict[str, TextRegion] = {}
    for region in text_regions:
        if region.id in region_by_id:
            raise ValueError(f"duplicate CTD region ID: {region.id}")
        region_by_id[region.id] = region

    missing_region_ids = [
        region_id for region_id in balloon.text_region_ids if region_id not in region_by_id
    ]
    if missing_region_ids:
        raise ValueError(f"balloon references missing CTD regions: {missing_region_ids}")

    candidate_groups = {group.region_id: group for group in candidate_bank.groups}
    region_references: list[BalloonRegionReference] = []
    candidate_records: list[BalloonCandidateEvidence] = []
    for region_id in balloon.text_region_ids:
        region = region_by_id[region_id]
        group = candidate_groups.get(region_id)
        region_candidates = group.candidates if group is not None else []
        region_references.append(
            BalloonRegionReference(
                region_id=region.id,
                bbox=region.bbox,
                category=region.category,
                confidence=region.confidence,
            )
        )
        candidate_records.extend(
            BalloonCandidateEvidence(
                candidate_id=candidate.candidate_id,
                region_id=candidate.region_id,
                source=candidate.source,
                text=candidate.text,
                ocr_confidence=candidate.evidence.ocr_confidence,
                visual_confidence=candidate.evidence.visual_confidence,
                candidate_supported=candidate.evidence.candidate_supported,
                semantic_type=candidate.evidence.semantic_type,
                preprocessing=candidate.evidence.preprocessing,
                bbox=candidate.bbox,
                notes=candidate.evidence.notes,
                source_metadata=candidate.evidence.source_metadata,
            )
            for candidate in region_candidates
        )

    all_candidate_text = {
        candidate.candidate_id: _similarity_key(candidate.text)
        for candidate in candidate_records
    }
    candidates = [
        candidate.model_copy(update={
            "similarity_to_candidates": {
                other_id: SequenceMatcher(
                    None,
                    all_candidate_text[candidate.candidate_id],
                    other_text,
                    autojunk=False,
                ).ratio()
                for other_id, other_text in all_candidate_text.items()
                if other_id != candidate.candidate_id
            }
        })
        for candidate in candidate_records
    ]

    return BalloonAdjudicationInput(
        balloon_id=balloon.id,
        balloon_bbox=balloon.bbox,
        balloon_image_path=str(Path(balloon_image_path)),
        panel_id=panel.id if panel is not None else balloon.panel_id,
        panel_bbox=panel.bbox if panel is not None else None,
        panel_image_path=(str(Path(panel_image_path)) if panel_image_path is not None else None),
        region_references=region_references,
        candidates=candidates,
    )


def build_adjudication_prompt(input: BalloonAdjudicationInput) -> str:
    """Serialize each proposal independently; never fuse or select hypotheses here."""
    candidates = [candidate.model_dump(mode="json") for candidate in input.candidates]
    region_refs = [region.model_dump(mode="json") for region in input.region_references]
    panel_context = (
        "A second image follows as auxiliary panel context. The first image remains primary."
        if input.panel_image_path
        else "No panel context image is available."
    )
    return (
        f"{ADJUDICATION_SYSTEM_PROMPT}\n\n"
        f"Balloon ID: {input.balloon_id}\n"
        f"CTD region references (provenance only): {json.dumps(region_refs, ensure_ascii=True)}\n"
        f"Panel bbox (optional geometry only): "
        f"{json.dumps(input.panel_bbox.model_dump(mode='json') if input.panel_bbox else None)}\n"
        f"{panel_context}\n"
        "Independent candidate hypotheses follow as separate records. Compare them, "
        "but do not vote or collapse them:\n"
        f"{json.dumps(candidates, ensure_ascii=True, separators=(',', ':'))}\n\n"
        "Return one JSON object. Use a text_type from dialogue, thought, narration, "
        "vocalisation, sound_effect, sign, title, metadata, document_text, or unknown. "
        "corrections must be a list of objects with candidate_id, from_text, and to_text. "
        "Only cite candidate IDs that were supplied."
    )


def _strip_json_fence(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _uncertain_result(
    balloon_id: str,
    code: str,
    message: str,
) -> BalloonAdjudicationResult:
    return BalloonAdjudicationResult(
        balloon_id=balloon_id,
        final_text="",
        text_type="unknown",
        include_in_story=False,
        confidence=None,
        notes=message,
        diagnostics=[AdjudicationDiagnostic(code=code, message=message)],
    )


class Qwen3VLBalloonAdjudicator:
    """Use one existing local Qwen runner to adjudicate multiple balloon crops."""

    def __init__(self, runner: Qwen3VLLocalRunner | Any) -> None:
        if not callable(getattr(runner, "generate_json", None)):
            raise TypeError("runner must expose generate_json(image_paths, prompt)")
        self.runner = runner

    def adjudicate(
        self,
        input: BalloonAdjudicationInput,
    ) -> BalloonAdjudicationResult:
        image_paths = [Path(input.balloon_image_path)]
        if input.panel_image_path is not None:
            image_paths.append(Path(input.panel_image_path))
        prompt = build_adjudication_prompt(input)
        try:
            response = self.runner.generate_json(image_paths, prompt)
        except Exception as exc:  # noqa: BLE001 - return explicit uncertainty on backend failure
            return _uncertain_result(
                input.balloon_id,
                "backend_failure",
                f"Qwen adjudication failed: {type(exc).__name__}: {exc}",
            )

        if isinstance(response, str):
            try:
                response = json.loads(_strip_json_fence(response))
            except json.JSONDecodeError:
                return _uncertain_result(
                    input.balloon_id,
                    "malformed_json",
                    "Qwen output was not valid JSON.",
                )
        if not isinstance(response, Mapping):
            return _uncertain_result(
                input.balloon_id,
                "invalid_response_type",
                "Qwen did not return a JSON object.",
            )
        if response.get("_json_parse_error"):
            return _uncertain_result(
                input.balloon_id,
                "malformed_json",
                str(response["_json_parse_error"]),
            )
        if isinstance(response, Mapping) and "_raw_output" in response:
            response = dict(response)
            response.pop("_raw_output", None)

        output = dict(response)
        if isinstance(output.get("final_text"), str):
            output["final_text"] = output["final_text"]
        elif "final_text" in output:
            return _uncertain_result(
                input.balloon_id,
                "invalid_final_text",
                "Qwen final_text was not a string.",
            )
        else:
            output["final_text"] = ""

        diagnostics: list[AdjudicationDiagnostic] = []
        confidence_value = output.get("confidence", output.get("decision_confidence"))
        confidence: float | None = None
        if confidence_value is not None:
            if (
                isinstance(confidence_value, bool)
                or not isinstance(confidence_value, (int, float))
                or not math.isfinite(float(confidence_value))
                or not 0.0 <= float(confidence_value) <= 1.0
            ):
                diagnostics.append(
                    AdjudicationDiagnostic(
                        code="invalid_confidence",
                        message="Qwen confidence was out of range or malformed; omitted.",
                    )
                )
            else:
                confidence = float(confidence_value)

        include_in_story = output.get("include_in_story", False)
        if not isinstance(include_in_story, bool):
            diagnostics.append(
                AdjudicationDiagnostic(
                    code="invalid_story_decision",
                    message="Qwen include_in_story was not boolean; defaulted to false.",
                )
            )
            include_in_story = False
        elif "include_in_story" not in output:
            diagnostics.append(
                AdjudicationDiagnostic(
                    code="missing_story_decision",
                    message="Qwen omitted include_in_story; defaulted to false.",
                )
            )

        text_type_value = output.get("text_type", "unknown")
        allowed_types = {
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
        if text_type_value not in allowed_types:
            diagnostics.append(
                AdjudicationDiagnostic(
                    code="invalid_text_type",
                    message="Qwen text_type was not in the semantic taxonomy; set to unknown.",
                )
            )
            text_type_value = "unknown"

        corrections: list[BalloonCorrection] = []
        raw_corrections = output.get("corrections", [])
        if not isinstance(raw_corrections, list):
            diagnostics.append(
                AdjudicationDiagnostic(
                    code="invalid_corrections",
                    message="Qwen corrections was not a list; corrections were omitted.",
                )
            )
            raw_corrections = []
        candidate_ids = {candidate.candidate_id for candidate in input.candidates}
        for raw_correction in raw_corrections:
            try:
                correction = BalloonCorrection.model_validate(raw_correction)
            except Exception as exc:  # noqa: BLE001 - reject malformed correction records individually
                diagnostics.append(
                    AdjudicationDiagnostic(
                        code="invalid_correction",
                        message=f"Correction record was invalid: {exc}",
                    )
                )
                continue
            if correction.candidate_id not in candidate_ids:
                diagnostics.append(
                    AdjudicationDiagnostic(
                        code="unknown_correction_candidate",
                        message=(
                            "A correction referenced a candidate ID not present in this balloon; "
                            "that correction was omitted."
                        ),
                    )
                )
                continue
            corrections.append(correction)

        notes = output.get("notes")
        if notes is not None and not isinstance(notes, str):
            notes = str(notes)
            diagnostics.append(
                AdjudicationDiagnostic(
                    code="coerced_notes",
                    message="Non-string notes were converted to text.",
                )
            )
        try:
            return BalloonAdjudicationResult(
                balloon_id=input.balloon_id,
                final_text=output["final_text"],
                text_type=text_type_value,
                include_in_story=include_in_story,
                confidence=confidence,
                corrections=corrections,
                notes=notes,
                diagnostics=diagnostics,
            )
        except Exception as exc:  # noqa: BLE001 - malformed decisions become explicit uncertainty
            return _uncertain_result(
                input.balloon_id,
                "invalid_adjudication_result",
                f"Qwen result failed validation: {type(exc).__name__}: {exc}",
            )

def adjudicate_balloon(
    adjudicator: BalloonAdjudicator,
    balloon: Balloon,
    candidate_bank: CandidateBank,
    text_regions: Sequence[TextRegion],
    balloon_crop: str | Path,
    *,
    panel: Panel | None = None,
    panel_crop: str | Path | None = None,
) -> BalloonAdjudicationResult:
    """Build structured balloon input and adjudicate without passing the page image."""
    input = build_balloon_adjudication_input(
        balloon,
        candidate_bank,
        text_regions,
        balloon_crop,
        panel=panel,
        panel_image_path=panel_crop,
    )
    return adjudicator.adjudicate(input)
