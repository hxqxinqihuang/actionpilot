from __future__ import annotations

from src.schemas.task import TaskExtractionResult


def export_result_as_json(result: TaskExtractionResult) -> str:
    return result.model_dump_json(indent=2)
