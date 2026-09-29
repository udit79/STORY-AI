"""Lazy, dependency-injected adapters for localized OCR/VLM candidates.

Each adapter receives a crop already localized by CTD. Model packages and
checkpoints are supplied by the caller, so importing this module does not load
GPU libraries, download weights, or cross the Nemotron environment boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from app.candidate_bank import PREPROCESSING_SOURCES, normalize_candidate
from app.schemas.candidates import CandidateSource, TranscriptionCandidate
from app.schemas.page import TextRegion


def _as_results(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


class PaddleOCRCandidateAdapter:
    """Call an injected PaddleOCR TextRecognition-compatible instance."""

    def __init__(self, recognizer: Any, preprocessing: str = "base") -> None:
        normalized = preprocessing.strip().lower()
        if normalized not in PREPROCESSING_SOURCES:
            raise ValueError(f"unsupported preprocessing identity: {preprocessing!r}")
        self.recognizer = recognizer
        self.preprocessing = normalized
        self.source: CandidateSource = PREPROCESSING_SOURCES[normalized]

    def candidates(
        self,
        region: TextRegion,
        crop_path: Path,
    ) -> list[TranscriptionCandidate]:
        results = self.recognizer.predict(input=str(crop_path), batch_size=1)
        candidates = []
        for index, result in enumerate(_as_results(results)):
            candidate = normalize_candidate(
                region,
                self.source,
                result,
                candidate_id=f"{region.id}:{self.source}:{index}",
                preprocessing=self.preprocessing,
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates


class NemotronCandidateAdapter:
    """Run a local Nemotron-compatible pipeline on one CTD crop only."""

    def __init__(self, pipeline: Callable[..., Any], merge_level: str = "paragraph") -> None:
        if merge_level not in {"word", "sentence", "paragraph"}:
            raise ValueError(f"unsupported Nemotron merge level: {merge_level!r}")
        self.pipeline = pipeline
        self.merge_level = merge_level

    def candidates(
        self,
        region: TextRegion,
        crop_path: Path,
    ) -> list[TranscriptionCandidate]:
        predictions = self.pipeline(str(crop_path), merge_level=self.merge_level)
        candidates = []
        for index, prediction in enumerate(_as_results(predictions)):
            candidate = normalize_candidate(
                region,
                "nemotron",
                prediction,
                candidate_id=f"{region.id}:nemotron:{index}",
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates


class Qwen3VLCandidateAdapter:
    """Normalize output from an injected local, localized-block Qwen runner."""

    def __init__(self, runner: Callable[[Path], Any]) -> None:
        self.runner = runner

    def candidates(
        self,
        region: TextRegion,
        crop_path: Path,
    ) -> list[TranscriptionCandidate]:
        output = self.runner(crop_path)
        candidates = []
        for index, prediction in enumerate(_as_results(output)):
            candidate = normalize_candidate(
                region,
                "qwen3_vl",
                prediction,
                candidate_id=f"{region.id}:qwen3_vl:{index}",
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates


def candidate_producers(
    *,
    paddle: PaddleOCRCandidateAdapter | None = None,
    nemotron: NemotronCandidateAdapter | None = None,
    qwen3_vl: Qwen3VLCandidateAdapter | None = None,
) -> Sequence[
    PaddleOCRCandidateAdapter | NemotronCandidateAdapter | Qwen3VLCandidateAdapter
]:
    """Return configured producers in stable source order; omitted models are skipped."""
    return tuple(
        producer
        for producer in (paddle, nemotron, qwen3_vl)
        if producer is not None
    )