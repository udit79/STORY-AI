from typing import Literal

from pydantic import BaseModel, Field

from .page import PageRepresentation


DatasetSplit = Literal["development", "test"]


class SequenceRecord(BaseModel):
    sequence_id: str
    split: DatasetSplit
    page_paths: list[str] = Field(min_length=3, max_length=3)


class SequenceRepresentation(BaseModel):
    sequence_id: str

    pages: list[PageRepresentation] = Field(
        min_length=3,
        max_length=3,
    )

    character_groups: dict[str, list[str]] = Field(
        default_factory=dict
    )

    ordered_story_lines: list[dict] = Field(
        default_factory=list
    )