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


def test_found_deadline_with_normalized_date_requires_raw_text_and_has_type() -> None:
    deadline = Deadline(
        status="found",
        raw_text="作品提交截止2026年9月5日",
        normalized_date="2026-09-05",
        type="submission",
    )

    assert deadline.raw_text == "作品提交截止2026年9月5日"
    assert deadline.type == "submission"

    with pytest.raises(ValidationError):
        Deadline(status="found", raw_text=None, normalized_date="2026-09-05")
