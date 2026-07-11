from __future__ import annotations

import pytest

from src.agent.extractor import ExtractionError, TaskExtractor
from src.providers.base import LLMProvider


class FakeProvider(LLMProvider):
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> str:
        return self._payload


def test_extractor_accepts_missing_deadline_with_null_source_fields() -> None:
    payload = """
    {
      "tasks": [
        {
          "title": "课程项目",
          "summary": "提交 README。",
          "deadlines": [{"raw_text": null, "normalized_date": null, "timezone": null, "status": "missing", "evidence": null}],
          "prerequisites": [{"name": "DeepSeek API", "description": null, "required": true, "evidence": "DeepSeek API"}],
          "materials": [{"name": "README.md", "description": null, "required": true, "evidence": "README.md"}],
          "requirements": [{"description": "人数要求 1 人或 1-2 人均可", "priority": "must", "evidence": "人数要求 1 人或 1-2 人均可"}],
          "risks": [],
          "confirmation_questions": [{"question": "具体提交日期是什么？", "reason": "原文未给出提交日期。"}]
        }
      ],
      "source_language": "中文",
      "confidence": 0.8
    }
    """
    extractor = TaskExtractor(FakeProvider(payload))
    source = "需要使用 DeepSeek API。提交 README.md。人数要求 1 人或 1-2 人均可。"

    result = extractor.extract(source)

    task = result.tasks[0]
    assert task.deadlines[0].status == "missing"
    assert task.deadlines[0].raw_text is None
    assert task.materials[0].name == "README.md"
    assert task.prerequisites[0].name == "DeepSeek API"
    assert task.requirements[0].priority == "must"


def test_extractor_rejects_evidence_not_present_in_source() -> None:
    payload = """
    {
      "tasks": [
        {
          "title": "课程项目",
          "summary": "提交 README。",
          "deadlines": [{"raw_text": null, "normalized_date": null, "timezone": null, "status": "missing", "evidence": null}],
          "prerequisites": [],
          "materials": [{"name": "README PDF", "description": null, "required": true, "evidence": "README PDF"}],
          "requirements": [],
          "risks": [],
          "confirmation_questions": []
        }
      ],
      "source_language": "中文",
      "confidence": 0.8
    }
    """
    extractor = TaskExtractor(FakeProvider(payload))

    with pytest.raises(ExtractionError):
        extractor.extract("提交 README.md。")
