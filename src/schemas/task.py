from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Deadline(BaseModel):
    raw_text: str | None = Field(default=None, description="Original deadline text from the notice.")
    normalized_date: str | None = Field(default=None, description="ISO date if confidently inferable.")
    timezone: str | None = Field(default=None, description="Timezone if mentioned or inferable.")
    status: Literal["found", "missing", "ambiguous"] = "found"
    evidence: str | None = Field(default=None, description="Exact evidence copied from the source text.")

    @model_validator(mode="after")
    def validate_missing_deadline(self) -> "Deadline":
        if self.status == "missing":
            if self.raw_text is not None or self.normalized_date is not None:
                raise ValueError("Missing deadlines must have null raw_text and null normalized_date.")
            if self.evidence is not None:
                raise ValueError("Missing deadlines cannot include evidence.")
        return self


class Prerequisite(BaseModel):
    name: str
    description: str | None = None
    required: bool = True
    evidence: str | None = Field(default=None, description="Exact evidence copied from the source text.")


class Material(BaseModel):
    name: str
    description: str | None = None
    required: bool = True
    evidence: str | None = Field(default=None, description="Exact evidence copied from the source text.")


class Requirement(BaseModel):
    description: str
    priority: Literal["must", "should", "optional", "unknown"] = "unknown"
    evidence: str | None = Field(default=None, description="Exact evidence copied from the source text.")


class Risk(BaseModel):
    description: str
    severity: Literal["low", "medium", "high", "unknown"] = "unknown"
    mitigation: str | None = None


class ConfirmationQuestion(BaseModel):
    question: str
    reason: str | None = None


class ExtractedTask(BaseModel):
    title: str
    summary: str
    deadlines: list[Deadline] = Field(default_factory=list)
    prerequisites: list[Prerequisite] = Field(default_factory=list)
    materials: list[Material] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    confirmation_questions: list[ConfirmationQuestion] = Field(default_factory=list)


class TaskExtractionResult(BaseModel):
    tasks: list[ExtractedTask] = Field(default_factory=list)
    source_language: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
