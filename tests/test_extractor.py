from __future__ import annotations

from src.agent.extractor import TaskExtractor
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
          "title": "Course project",
          "summary": "Submit README.",
          "deadlines": [{"raw_text": null, "normalized_date": null, "timezone": null, "status": "missing", "evidence": null}],
          "prerequisites": [{"name": "DeepSeek API", "description": null, "required": true, "evidence": "DeepSeek API"}],
          "materials": [{"name": "README.md", "description": null, "required": true, "evidence": "README.md"}],
          "requirements": [{"description": "Team size can be 1 or 2 people", "priority": "must", "evidence": "1 or 2 people"}],
          "risks": [],
          "confirmation_questions": [{"question": "What is the deadline?", "reason": "The source has no date."}]
        }
      ],
      "source_language": "English",
      "confidence": 0.8
    }
    """
    extractor = TaskExtractor(FakeProvider(payload))
    source = "Need DeepSeek API. Submit README.md. Team size can be 1 or 2 people."

    result = extractor.extract(source)

    task = result.tasks[0]
    assert task.deadlines[0].status == "missing"
    assert task.deadlines[0].raw_text is None
    assert task.materials[0].name == "README.md"
    assert task.prerequisites[0].name == "DeepSeek API"
    assert task.requirements[0].priority == "must"


def test_extractor_downgrades_ordinary_evidence_not_present_in_source() -> None:
    payload = """
    {
      "tasks": [
        {
          "title": "Course project",
          "summary": "Submit README.",
          "deadlines": [{"raw_text": null, "normalized_date": null, "timezone": null, "status": "missing", "evidence": null}],
          "prerequisites": [],
          "materials": [{"name": "README PDF", "description": null, "required": true, "evidence": "README PDF"}],
          "requirements": [],
          "risks": [],
          "confirmation_questions": []
        }
      ],
      "source_language": "English",
      "confidence": 0.8
    }
    """
    extractor = TaskExtractor(FakeProvider(payload))

    result = extractor.extract("Submit README.md.")

    material = result.tasks[0].materials[0]
    assert material.evidence is None
    assert material.evidence_status == "unverified"
    assert result.warnings


def test_extractor_keeps_fallback_normalized_deadlines_without_evidence() -> None:
    payload = """
    {
      "tasks": [
        {
          "title": "Competition notice",
          "summary": "Competition notice",
          "deadlines": [
            {"raw_text": "2026-06-30", "normalized_date": "2026-06-30", "timezone": null, "status": "found", "type": "registration", "evidence": null},
            {"raw_text": "2026-09-15", "normalized_date": "2026-09-15", "timezone": null, "status": "found", "type": "submission", "evidence": null}
          ],
          "prerequisites": [],
          "materials": [{"name": "Submission file", "description": null, "required": true, "evidence": null}],
          "requirements": [{"description": "Complete registration", "priority": "must", "evidence": null}],
          "risks": [],
          "confirmation_questions": []
        }
      ],
      "source_language": "English",
      "confidence": 0.0,
      "warnings": []
    }
    """
    extractor = TaskExtractor(FakeProvider(payload))

    result = extractor.extract("A long competition notice. 2026-06-30 registration. 2026-09-15 submission.")

    deadlines = result.tasks[0].deadlines
    assert [deadline.normalized_date for deadline in deadlines] == ["2026-06-30", "2026-09-15"]
    assert [deadline.status for deadline in deadlines] == ["found", "found"]
    assert [deadline.raw_text for deadline in deadlines] == ["2026-06-30", "2026-09-15"]
    assert [deadline.type for deadline in deadlines] == ["registration", "submission"]
    assert result.tasks[0].materials[0].name == "Submission file"
    assert result.tasks[0].requirements[0].description == "Complete registration"
