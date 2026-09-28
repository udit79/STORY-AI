from typing import Literal

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class TextRegion(BaseModel):
    id: str
    bbox: BoundingBox
    raw_text: str
    confidence: float = Field(ge=0.0, le=1.0)

    category: Literal[
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
    ] = "unknown"


class CharacterInstance(BaseModel):
    id: str
    bbox: BoundingBox
    description: str | None = None


class Balloon(BaseModel):
    id: str
    bbox: BoundingBox

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
    balloons: list[Balloon] = Field(default_factory=list)
    characters: list[CharacterInstance] = Field(default_factory=list)

    reading_order: list[str] = Field(default_factory=list)