from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas.task import Deadline, ExtractedTask, TaskExtractionResult


def test_task_extraction_result_defaults() -> None:
    result = TaskExtractionResult(tasks=[ExtractedTask(title="Task", summary="Summary")])

    assert result.tasks[0].deadlines == []
    assert result.tasks[0].prerequisites == []
    assert result.confidence == 0.0


def test_missing_deadline_requires_null_raw_text_and_normalized_date() -> None:
    deadline = Deadline(status="missing", raw_text=None, normalized_date=None, timezone=None)

    assert deadline.raw_text is None
    assert deadline.normalized_date is None


def test_missing_deadline_rejects_generated_raw_text() -> None:
    with pytest.raises(ValidationError):
        Deadline(status="missing", raw_text="原文没有给出具体提交日期", normalized_date=None)
