from __future__ import annotations

import pytest

from src.agent.extractor import ExtractionError, TaskExtractor
from src.agent.prompts import COMPACT_EXTRA_INSTRUCTIONS, CORE_SYSTEM_PROMPT
from src.analysis_policy import (
    COMPACT_MODE_WARNING,
    COMPACT_RECOMMENDED_MAX_CHARS,
    MAX_INPUT_CHARS,
    STANDARD_MODE_MAX_CHARS,
    check_input_length,
)
from src.providers.base import LLMProvider


class PromptRecordingProvider(LLMProvider):
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.system_prompts: list[str] = []

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> str:
        self.system_prompts.append(system_prompt)
        return self.payload


def _empty_result_payload() -> str:
    return '{"tasks": [], "source_language": "English", "confidence": 0.0, "warnings": []}'


def test_standard_mode_success_does_not_use_compact_prompt() -> None:
    provider = PromptRecordingProvider(_empty_result_payload())
    extractor = TaskExtractor(provider)

    extractor.extract("short task notice")

    assert len(provider.system_prompts) == 1
    assert "COMPACT EXTRACTION MODE" not in provider.system_prompts[0]


def test_under_8000_chars_still_allows_provider_length_to_trigger_compact_retry() -> None:
    check = check_input_length("a" * STANDARD_MODE_MAX_CHARS)

    assert check.allowed is True
    assert check.mode == "standard"


def test_8001_to_15000_chars_use_compact_mode() -> None:
    check = check_input_length("a" * (STANDARD_MODE_MAX_CHARS + 1))

    assert check.allowed is True
    assert check.mode == "compact"
    assert check.warning is not None


def test_15001_to_30000_chars_use_compact_mode_with_stronger_warning() -> None:
    check = check_input_length("a" * (COMPACT_RECOMMENDED_MAX_CHARS + 1))

    assert check.allowed is True
    assert check.mode == "compact"
    assert check.warning is not None
    assert "removing appendices" in check.warning


def test_over_30000_chars_is_rejected_before_provider_call() -> None:
    check = check_input_length("a" * (MAX_INPUT_CHARS + 1))

    assert check.allowed is False
    assert check.error is not None


def test_long_input_uses_compact_prompt_and_adds_warning() -> None:
    provider = PromptRecordingProvider(_empty_result_payload())
    extractor = TaskExtractor(provider)

    result = extractor.extract("a" * (STANDARD_MODE_MAX_CHARS + 1))

    assert "COMPACT EXTRACTION MODE" in provider.system_prompts[0]
    assert COMPACT_MODE_WARNING in result.warnings


def test_compact_prompt_contains_all_category_limits() -> None:
    assert "deadlines 5" in COMPACT_EXTRA_INSTRUCTIONS
    assert "prerequisites 8" in COMPACT_EXTRA_INSTRUCTIONS
    assert "materials 10" in COMPACT_EXTRA_INSTRUCTIONS
    assert "requirements 15" in COMPACT_EXTRA_INSTRUCTIONS
    assert "risks 5" in COMPACT_EXTRA_INSTRUCTIONS
    assert "confirmation_questions 5" in COMPACT_EXTRA_INSTRUCTIONS


def test_core_prompt_contains_limits_and_excludes_non_core_output() -> None:
    assert "deadlines max 6" in CORE_SYSTEM_PROMPT
    assert "materials max 8" in CORE_SYSTEM_PROMPT
    assert "requirements max 8" in CORE_SYSTEM_PROMPT
    assert "confirmation_questions max 3" in CORE_SYSTEM_PROMPT
    assert "summary max 100" in CORE_SYSTEM_PROMPT
    assert "each evidence max 60" in CORE_SYSTEM_PROMPT
    assert "mitigation" not in CORE_SYSTEM_PROMPT
    assert "full scoring rubrics" in CORE_SYSTEM_PROMPT
    assert '"risks": []' in CORE_SYSTEM_PROMPT


def test_evidence_validation_failure_does_not_trigger_compact_retry() -> None:
    items = ", ".join(
        f'{{"name": "Item {index}", "description": null, "required": true, "evidence": "missing evidence {index}"}}'
        for index in range(5)
    )
    payload = f"""
    {{
      "tasks": [
        {{
          "title": "Task",
          "summary": "Summary",
          "deadlines": [{{"raw_text": null, "normalized_date": null, "timezone": null, "status": "missing", "evidence": null}}],
          "prerequisites": [{items}],
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
    provider = PromptRecordingProvider(payload)
    extractor = TaskExtractor(provider)

    with pytest.raises(ExtractionError):
        extractor.extract("A source with no matching evidence.")

    assert len(provider.system_prompts) == 1
