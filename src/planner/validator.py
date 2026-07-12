from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from src.schemas.action_plan import ActionPlan


class PlanValidationError(ValueError):
    """Raised when a generated plan violates local planning constraints."""


@dataclass(frozen=True)
class PlanRequestValidation:
    allowed: bool
    errors: list[str]
    warnings: list[str]


def validate_plan_request(
    *,
    start_date: date,
    target_date: date,
    available_hours_per_day: float,
    available_days_per_week: int,
    buffer_days: int,
) -> PlanRequestValidation:
    errors: list[str] = []
    warnings: list[str] = []
    total_days = (target_date - start_date).days + 1

    if target_date < start_date:
        errors.append("Target date must not be earlier than the start date.")
    if available_hours_per_day <= 0 or available_hours_per_day > 16:
        errors.append("Daily available hours must be greater than 0 and no more than 16.")
    if available_days_per_week < 1 or available_days_per_week > 7:
        errors.append("Available days per week must be between 1 and 7.")
    if total_days > 0 and buffer_days >= total_days:
        errors.append("Final check buffer must be smaller than the total available days.")
    if total_days > 0 and total_days <= 2:
        warnings.append("The available time is very short; the generated plan may be tight.")

    return PlanRequestValidation(allowed=not errors, errors=errors, warnings=warnings)


def validate_generated_plan(
    plan: ActionPlan,
    *,
    start_date: date,
    target_date: date,
    available_hours_per_day: float,
    completed_materials: list[str],
) -> None:
    if plan.start_date != start_date or plan.target_date != target_date:
        raise PlanValidationError("Plan dates do not match the requested date range.")

    daily_hours: dict[date, float] = defaultdict(float)
    for task in plan.daily_tasks:
        if task.date < start_date or task.date > target_date:
            raise PlanValidationError("A daily task is outside the requested date range.")
        daily_hours[task.date] += task.estimated_hours
        _validate_completed_material_not_recreated(task.title, task.description, task.deliverable, completed_materials)

    for task_date, hours in daily_hours.items():
        if hours > available_hours_per_day + 0.01:
            raise PlanValidationError(f"Daily tasks on {task_date.isoformat()} exceed available hours.")

    for phase in plan.phases:
        if phase.start_date < start_date or phase.end_date > target_date or phase.end_date < phase.start_date:
            raise PlanValidationError("A plan phase is outside the requested date range.")

    for milestone in plan.milestones:
        if milestone.date < start_date or milestone.date > target_date:
            raise PlanValidationError("A milestone is outside the requested date range.")


def _validate_completed_material_not_recreated(
    title: str,
    description: str,
    deliverable: str,
    completed_materials: list[str],
) -> None:
    text = f"{title} {description} {deliverable}".lower()
    allowed_review_words = ("check", "review", "polish", "integrate", "verify", "检查", "复核", "润色", "整合", "确认")
    if any(word in text for word in allowed_review_words):
        return

    for material in completed_materials:
        if material and material.lower() in text:
            raise PlanValidationError("Completed materials must not be scheduled as main creation tasks.")
