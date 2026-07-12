from __future__ import annotations

import pytest

from src.agent.extractor import ExtractionError, TaskExtractor
from src.providers.base import LLMProvider


class FakeProvider(LLMProvider):
    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.calls = 0

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return self._payload


def _payload_with_items(items_json: str, deadlines_json: str | None = None) -> str:
    deadlines = deadlines_json or '[{"raw_text": null, "normalized_date": null, "timezone": null, "status": "missing", "evidence": null}]'
    return f"""
    {{
      "tasks": [
        {{
          "title": "Task",
          "summary": "Summary",
          "deadlines": {deadlines},
          "prerequisites": {items_json},
          "materials": [],
          "requirements": [],
          "risks": [],
          "confirmation_questions": []
        }}
      ],
      "source_language": "English",
      "confidence": 0.8
    }}
    """


def test_evidence_matches_after_unicode_and_whitespace_normalization() -> None:
    payload = _payload_with_items(
        '[{"name": "API", "description": null, "required": true, "evidence": "Use DeepSeek API"}]'
    )
    extractor = TaskExtractor(FakeProvider(payload))

    result = extractor.extract("Use\u3000DeepSeek\nAPI to build the project.")

    item = result.tasks[0].prerequisites[0]
    assert item.evidence_status == "verified"
    assert item.evidence == "Use DeepSeek API"
    assert result.warnings == []


def test_one_ordinary_unverified_evidence_does_not_fail_whole_result() -> None:
    payload = _payload_with_items(
        '[{"name": "API", "description": null, "required": true, "evidence": "not in source"}]'
    )
    extractor = TaskExtractor(FakeProvider(payload))

    result = extractor.extract("Use DeepSeek API.")

    item = result.tasks[0].prerequisites[0]
    assert item.evidence is None
    assert item.evidence_status == "unverified"
    assert result.warnings


def test_deadline_unverified_evidence_is_downgraded() -> None:
    deadlines = """
    [{
      "raw_text": "Submit before July 1",
      "normalized_date": "2026-07-01",
      "timezone": null,
      "status": "found",
      "evidence": "Submit before July 1"
    }]
    """
    payload = _payload_with_items("[]", deadlines)
    extractor = TaskExtractor(FakeProvider(payload))

    result = extractor.extract("Submit when the teacher announces the date.")

    deadline = result.tasks[0].deadlines[0]
    assert deadline.status == "missing"
    assert deadline.raw_text is None
    assert deadline.normalized_date is None
    assert deadline.evidence is None
    assert deadline.evidence_status == "unverified"
    assert result.warnings


def test_many_unverified_evidence_items_raise_extraction_error() -> None:
    items = ", ".join(
        f'{{"name": "Item {index}", "description": null, "required": true, "evidence": "missing evidence {index}"}}'
        for index in range(5)
    )
    payload = _payload_with_items(f"[{items}]")
    extractor = TaskExtractor(FakeProvider(payload))

    with pytest.raises(ExtractionError):
        extractor.extract("A source with no matching evidence.")


def test_evidence_validation_failure_does_not_retry_provider() -> None:
    items = ", ".join(
        f'{{"name": "Item {index}", "description": null, "required": true, "evidence": "missing evidence {index}"}}'
        for index in range(5)
    )
    provider = FakeProvider(_payload_with_items(f"[{items}]"))
    extractor = TaskExtractor(provider)

    with pytest.raises(ExtractionError):
        extractor.extract("A source with no matching evidence.")

    assert provider.calls == 1
