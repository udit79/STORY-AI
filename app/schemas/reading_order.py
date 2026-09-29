"""Typed contracts for sequence-level reading order."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ReadingOrderItem(BaseModel):
    """One balloon's position within the reading sequence."""

    page_index: int
    panel_id: str | None
    balloon_id: str
    sequence_position: int  # 0-based global index in the final ordered list


class ReadingOrderDecision(BaseModel):
    """Structured pairwise ordering decision: does *first* precede *second*?"""

    first_balloon_id: str
    second_balloon_id: str
    first_precedes_second: bool
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: dict[str, Any] = Field(default_factory=dict)
    method: Literal[
        "panel_geometry",
        "balloon_geometry",
        "pairwise_model",
        "fallback",
    ] = "balloon_geometry"


class ReadingOrderDiagnostic(BaseModel):
    component: str
    code: str
    message: str
    page_index: int | None = None
    panel_id: str | None = None
    balloon_ids: list[str] = Field(default_factory=list)


class PageReadingOrder(BaseModel):
    """Reading order for a single page: panels then balloons within each panel."""

    page_index: int
    panel_ids_in_order: list[str | None]
    balloon_ids_in_order: list[str]  # all story balloons for this page, in order
    decisions: list[ReadingOrderDecision] = Field(default_factory=list)
    diagnostics: list[ReadingOrderDiagnostic] = Field(default_factory=list)


class SequenceReadingOrder(BaseModel):
    """Reading order across all pages in the 3-page sequence."""

    ordered_items: list[ReadingOrderItem]
    ordered_balloon_ids: list[str]  # flat final list
    page_orders: list[PageReadingOrder]
    diagnostics: list[ReadingOrderDiagnostic] = Field(default_factory=list)
