from __future__ import annotations

from pydantic import BaseModel, Field


class ParseResult(BaseModel):
    text: str
    file_name: str
    file_type: str
    char_count: int = Field(ge=0)
    page_count: int | None = None
    warnings: list[str] = Field(default_factory=list)
