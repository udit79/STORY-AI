"""Sequence-safe orchestration from one page image to page evidence and candidates."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from app.candidate_bank import CandidateProducer, build_candidate_bank
from app.models.crops import CropTransform, CTDCropStore
from app.schemas.candidates import CandidateBank, TranscriptionCandidate
from app.schemas.page import PageRepresentation


class PerceptionDiagnostic(BaseModel):
    stage: Literal["localization", "crop", "producer"]
    component: str
    status: Literal["failed", "unavailable"]
    region_id: str | None = None
    error_type: str
    message: str


class PagePerceptionResult(BaseModel):
    page: PageRepresentation
    candidate_bank: CandidateBank
    diagnostics: list[PerceptionDiagnostic]


class PagePerceptionService:
    """Coordinate CTD, crops, and injected candidate producers for one page."""

    def __init__(
        self,
        localizer: Any,
        crop_store: CTDCropStore,
        producers: Mapping[str, CandidateProducer | None],
        *,
        crop_variants: Mapping[str, CropTransform] | None = None,
        backend_errors: Mapping[str, str] | None = None,
    ) -> None:
        self.localizer = localizer
        self.crop_store = crop_store
        self.producers = dict(producers)
        self.crop_variants = dict(crop_variants or {})
        self.backend_errors = dict(backend_errors or {})

    def process_page(
        self,
        image_path: str | Path,
        sequence_id: str,
        page_index: int,
    ) -> PagePerceptionResult:
        image_path = Path(image_path)
        diagnostics: list[PerceptionDiagnostic] = []
        if page_index < 0:
            raise ValueError("page_index must be zero-based and non-negative")
        try:
            regions = self.localizer.localize(image_path)
        except Exception as exc:  # noqa: BLE001 - localization failure is returned explicitly
            diagnostics.append(
                PerceptionDiagnostic(
                    stage="localization",
                    component="ctd",
                    status="failed",
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            regions = []

        page = PageRepresentation(
            page_index=page_index,
            image_path=str(image_path),
            text_regions=regions,
        )
        if not regions:
            return PagePerceptionResult(
                page=page,
                candidate_bank=CandidateBank(),
                diagnostics=diagnostics,
            )

        try:
            crop_result = self.crop_store.generate(
                image_path,
                sequence_id,
                page_index,
                regions,
                variants=self.crop_variants,
            )
        except Exception as exc:  # noqa: BLE001 - crop-store failure is reported, not hidden
            diagnostics.append(
                PerceptionDiagnostic(
                    stage="crop",
                    component="crop_store",
                    status="failed",
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            return PagePerceptionResult(
                page=page,
                candidate_bank=CandidateBank(),
                diagnostics=diagnostics,
            )

        for region_id, message in crop_result.failures.items():
            diagnostics.append(
                PerceptionDiagnostic(
                    stage="crop",
                    component="crop_store",
                    status="failed",
                    region_id=region_id,
                    error_type="CropGenerationError",
                    message=message,
                )
            )

        candidates: list[TranscriptionCandidate] = []
        for producer_name, producer in self.producers.items():
            if producer is None:
                diagnostics.append(
                    PerceptionDiagnostic(
                        stage="producer",
                        component=producer_name,
                        status="unavailable",
                        error_type="BackendUnavailable",
                        message=self.backend_errors.get(
                            producer_name, "No backend was configured."
                        ),
                    )
                )
                continue

            for region in regions:
                region_crops = crop_result.crop_paths.get(region.id)
                if region_crops is None:
                    continue
                variant = getattr(producer, "preprocessing", "base")
                if variant in {"none", "identity"}:
                    variant = "base"
                crop_path = region_crops.get(variant)
                if crop_path is None:
                    diagnostics.append(
                        PerceptionDiagnostic(
                            stage="producer",
                            component=producer_name,
                            status="failed",
                            region_id=region.id,
                            error_type="CropVariantUnavailable",
                            message=f"No {variant!r} crop was generated for this region.",
                        )
                    )
                    continue
                try:
                    produced = producer.candidates(region, crop_path)
                    if any(candidate.region_id != region.id for candidate in produced):
                        raise ValueError("producer returned a candidate for another region")
                    candidates.extend(produced)
                except Exception as exc:  # noqa: BLE001 - isolate model failures per producer/region
                    diagnostics.append(
                        PerceptionDiagnostic(
                            stage="producer",
                            component=producer_name,
                            status="failed",
                            region_id=region.id,
                            error_type=type(exc).__name__,
                            message=str(exc),
                        )
                    )

        bank = build_candidate_bank(regions, candidates)
        return PagePerceptionResult(
            page=page,
            candidate_bank=bank,
            diagnostics=diagnostics,
        )