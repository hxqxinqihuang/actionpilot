from __future__ import annotations

from src.schemas.plan import ActionPlan
from src.schemas.task import TaskExtractionResult


def create_empty_plan(result: TaskExtractionResult) -> ActionPlan:
    objective = result.tasks[0].title if result.tasks else "No task extracted"
    return ActionPlan(objective=objective)
