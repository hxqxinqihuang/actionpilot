from __future__ import annotations

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    title: str
    description: str
    due_hint: str | None = None


class ActionPlan(BaseModel):
    objective: str
    steps: list[PlanStep] = Field(default_factory=list)
