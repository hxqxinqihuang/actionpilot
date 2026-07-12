from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


PlanPriority = Literal["high", "medium", "low"]
PlanType = Literal["daily", "weekly", "phase"]


class PlanPhase(BaseModel):
    name: str
    start_date: date
    end_date: date
    objective: str
    key_actions: list[str] = Field(default_factory=list)
    deliverable: str | None = None


class DailyTask(BaseModel):
    date: date
    title: str
    description: str
    estimated_hours: float = Field(gt=0)
    priority: PlanPriority = "medium"
    deliverable: str
    related_materials: list[str] = Field(default_factory=list)


class Milestone(BaseModel):
    date: date
    name: str
    success_criteria: str


class ActionPlan(BaseModel):
    title: str
    goal: str
    start_date: date
    target_date: date
    total_days: int = Field(ge=1)
    available_hours_per_day: float = Field(gt=0, le=16)
    plan_type: PlanType = "daily"
    current_focus: str | None = None
    next_actions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    phases: list[PlanPhase] = Field(default_factory=list)
    daily_tasks: list[DailyTask] = Field(default_factory=list)
    milestones: list[Milestone] = Field(default_factory=list)
    final_checklist: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
