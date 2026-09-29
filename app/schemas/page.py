from typing import Any, Literal

from pydantic import BaseModel, Field

TextRegionCategory = Literal[
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
]


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Point2D(BaseModel):
    x: float
    y: float


class TextRegion(BaseModel):
    id: str
    bbox: BoundingBox
    raw_text: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    category: TextRegionCategory = "unknown"


class CharacterInstance(BaseModel):
    id: str
    bbox: BoundingBox
    description: str | None = None
    panel_id: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    face_bbox: BoundingBox | None = None
    face_bboxes: list[BoundingBox] = Field(default_factory=list)
    face_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    body_bbox: BoundingBox | None = None
    visual_embedding: list[float] | None = None
    visual_metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def character_instance_id(self) -> str:
        return self.id


class Panel(BaseModel):
    id: str
    bbox: BoundingBox
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source: Literal["layout_model", "page_fallback"] = "layout_model"


class Balloon(BaseModel):
    id: str
    bbox: BoundingBox
    panel_id: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    grouping_method: Literal[
        "layout_model",
        "geometry",
        "containment",
        "hybrid",
        "visual_fallback",
    ] = "geometry"
    grouping_evidence: dict[str, Any] = Field(default_factory=dict)
    mask_polygon: list[Point2D] | None = None

    kind: Literal[
        "speech",
        "thought",
        "narration",
        "unknown",
    ] = "unknown"

    text_region_ids: list[str] = Field(default_factory=list)
    candidate_character_ids: list[str] = Field(default_factory=list)


class PageRepresentation(BaseModel):
    page_index: int
    image_path: str

    text_regions: list[TextRegion] = Field(default_factory=list)
    panels: list[Panel] = Field(default_factory=list)
    balloons: list[Balloon] = Field(default_factory=list)
    characters: list[CharacterInstance] = Field(default_factory=list)

    reading_order: list[str] = Field(default_factory=list)