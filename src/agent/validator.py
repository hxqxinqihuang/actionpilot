from __future__ import annotations

from src.schemas.task import TaskExtractionResult


def has_tasks(result: TaskExtractionResult) -> bool:
    return len(result.tasks) > 0
