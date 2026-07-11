from __future__ import annotations

from src.schemas.task import ExtractedTask, TaskExtractionResult
from src.tools.export_tool import export_result_as_json


def test_export_result_as_json() -> None:
    result = TaskExtractionResult(tasks=[ExtractedTask(title="Submit report", summary="Submit report.")])

    exported = export_result_as_json(result)

    assert "Submit report" in exported
