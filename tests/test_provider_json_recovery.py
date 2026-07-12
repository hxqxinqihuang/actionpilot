from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from src.providers.openai_compatible import (
    LLMEmptyResponseError,
    LLMFilteredResponseError,
    LLMInsufficientResourceError,
    LLMNoChoicesError,
    LLMTruncatedResponseError,
    OpenAICompatibleProvider,
)
from src.analysis_policy import CORE_ACTION_MODE_WARNING, EMERGENCY_EXTRACTION_WARNING
from src.schemas.task import TaskExtractionResult


def _provider() -> OpenAICompatibleProvider:
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model = "deepseek-test"
    return provider


def _response(
    *,
    content: str | None,
    finish_reason: str = "stop",
    reasoning_content: str | None = None,
) -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    if reasoning_content is not None:
        message.reasoning_content = reasoning_content
    choice = SimpleNamespace(finish_reason=finish_reason, message=message)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    return SimpleNamespace(model="deepseek-test", choices=[choice], usage=usage)


def test_json_mode_empty_content_falls_back_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    calls: list[bool] = []

    def fake_create(
        *, system_prompt: str, user_prompt: str, use_json_mode: bool, attempt: int, analysis_mode: str
    ) -> SimpleNamespace:
        calls.append(use_json_mode)
        if len(calls) == 1:
            return _response(content="", finish_reason="stop")
        return _response(content='{"tasks": [], "source_language": "English", "confidence": 0.0, "warnings": []}')

    monkeypatch.setattr(provider, "_create_completion", fake_create)

    result = provider.generate_json(system_prompt="Return JSON", user_prompt="hello")

    assert calls == [True, False]
    assert '"tasks"' in result


def test_normal_success_does_not_trigger_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    calls: list[str] = []

    def fake_create(
        *, system_prompt: str, user_prompt: str, use_json_mode: bool, attempt: int, analysis_mode: str
    ) -> SimpleNamespace:
        calls.append(analysis_mode)
        return _response(content='{"tasks": [], "source_language": "English", "confidence": 0.0, "warnings": []}')

    monkeypatch.setattr(provider, "_create_completion", fake_create)

    result = provider.generate_json(system_prompt="Return JSON", user_prompt="hello")

    assert calls == ["normal"]
    assert '"tasks"' in result


def test_json_mode_empty_content_stops_after_fallback_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    calls = 0

    def fake_create(
        *, system_prompt: str, user_prompt: str, use_json_mode: bool, attempt: int, analysis_mode: str
    ) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return _response(content="", finish_reason="stop")

    monkeypatch.setattr(provider, "_create_completion", fake_create)

    with pytest.raises(LLMEmptyResponseError):
        provider.generate_json(system_prompt="Return JSON", user_prompt="hello")

    assert calls == 2


def test_no_choices_raises_clear_error() -> None:
    provider = _provider()
    response = SimpleNamespace(model="deepseek-test", choices=[], usage=None)

    with pytest.raises(LLMNoChoicesError):
        provider._extract_json_content(response, attempt=1, use_json_mode=True)


def test_finish_reason_length_raises_truncated_error() -> None:
    provider = _provider()

    with pytest.raises(LLMTruncatedResponseError):
        provider._extract_json_content(_response(content="{}", finish_reason="length"), attempt=1, use_json_mode=True)


def test_normal_length_retries_once_with_core_prompt_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    calls: list[str] = []

    def fake_create(
        *, system_prompt: str, user_prompt: str, use_json_mode: bool, attempt: int, analysis_mode: str
    ) -> SimpleNamespace:
        calls.append(system_prompt)
        if len(calls) == 1:
            return _response(content="", finish_reason="length")
        return _response(content='{"tasks": [], "source_language": "English", "confidence": 0.0, "warnings": []}')

    monkeypatch.setattr(provider, "_create_completion", fake_create)

    result = provider.generate_json(system_prompt="Return JSON", user_prompt="hello")

    assert len(calls) == 2
    assert "CORE ACTION MODE" not in calls[0]
    assert "CORE ACTION MODE" in calls[1]
    assert CORE_ACTION_MODE_WARNING in result


def test_core_length_uses_emergency_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    calls: list[str] = []

    def fake_create(
        *, system_prompt: str, user_prompt: str, use_json_mode: bool, attempt: int, analysis_mode: str
    ) -> SimpleNamespace:
        calls.append(analysis_mode)
        if analysis_mode == "core":
            return _response(content="", finish_reason="length")
        return _response(
            content='{"title": "Competition", "deadline": "9月15日前提交作品", "must_do": ["完成报名"], "materials": ["作品"], "questions": ["确认提交环节"], "warnings": []}'
        )

    monkeypatch.setattr(provider, "_create_completion", fake_create)

    result = provider.generate_json(system_prompt="Return JSON\n\nCORE ACTION MODE:", user_prompt="hello")

    assert calls == ["core", "emergency"]
    assert EMERGENCY_EXTRACTION_WARNING in result
    assert '"materials"' in result


def test_emergency_deadline_materials_and_requirements_map_to_task_result() -> None:
    provider = _provider()

    raw = provider._convert_emergency_json_to_task_result(
        '{"title": "比赛通知", "deadline": "2026-06-30 和 2026-09-15", '
        '"must_do": ["完成报名", "提交作品"], "materials": ["报名表", "作品文件"], '
        '"questions": ["确认提交入口"], "warnings": []}'
    )
    result = TaskExtractionResult.model_validate_json(raw)
    task = result.tasks[0]

    assert [deadline.normalized_date for deadline in task.deadlines] == ["2026-06-30", "2026-09-15"]
    assert all(deadline.status == "found" for deadline in task.deadlines)
    assert [deadline.raw_text for deadline in task.deadlines] == ["2026-06-30", "2026-09-15"]
    assert [material.name for material in task.materials] == ["报名表", "作品文件"]
    assert [requirement.description for requirement in task.requirements] == ["完成报名", "提交作品"]


def test_emergency_fallback_extracts_deadline_from_must_do_text() -> None:
    provider = _provider()

    raw = provider._convert_emergency_json_to_task_result(
        '{"title": "比赛通知", "deadlines": [], '
        '"must_do": ["报名时间为2026年5月30日至6月30日"], '
        '"materials": ["报名表"], "warnings": []}'
    )
    result = TaskExtractionResult.model_validate_json(raw)

    deadline = result.tasks[0].deadlines[0]
    assert deadline.status == "found"
    assert deadline.normalized_date == "2026-06-30"
    assert deadline.raw_text == "报名时间为2026年5月30日至6月30日"
    assert deadline.type == "registration"
    assert result.tasks[0].materials[0].name == "报名表"


def test_emergency_without_date_keeps_missing_deadline() -> None:
    provider = _provider()

    raw = provider._convert_emergency_json_to_task_result(
        '{"title": "比赛通知", "deadlines": [], '
        '"must_do": ["完成报名"], "materials": ["报名表"], "warnings": []}'
    )
    result = TaskExtractionResult.model_validate_json(raw)

    deadline = result.tasks[0].deadlines[0]
    assert deadline.status == "missing"
    assert deadline.normalized_date is None


def test_emergency_fallback_extracts_registration_and_submission_deadline_types() -> None:
    provider = _provider()

    raw = provider._convert_emergency_json_to_task_result(
        '{"title": "比赛通知", "deadlines": [], '
        '"must_do": ["报名时间2026年5月30日-6月30日", "作品提交截止2026年9月5日"], '
        '"materials": ["报名表", "作品文件"], "warnings": []}'
    )
    result = TaskExtractionResult.model_validate_json(raw)
    deadlines = result.tasks[0].deadlines

    assert [deadline.normalized_date for deadline in deadlines] == ["2026-06-30", "2026-09-05"]
    assert all(deadline.raw_text for deadline in deadlines)
    assert [deadline.type for deadline in deadlines] == ["registration", "submission"]


def test_compact_length_retries_once_with_core_prompt_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    calls: list[str] = []

    def fake_create(
        *, system_prompt: str, user_prompt: str, use_json_mode: bool, attempt: int, analysis_mode: str
    ) -> SimpleNamespace:
        calls.append(system_prompt)
        if len(calls) == 1:
            return _response(content="", finish_reason="length")
        return _response(
            content='{"tasks": [{"title": "Competition", "summary": "Core facts", "deadlines": [], "prerequisites": [], "materials": [{"name": "作品", "required": true, "evidence": "提交作品"}], "requirements": [{"description": "符合报名资格", "priority": "must", "evidence": "报名资格"}], "risks": [], "confirmation_questions": []}], "source_language": "中文", "confidence": 0.8, "warnings": []}'
        )

    monkeypatch.setattr(provider, "_create_completion", fake_create)

    result = provider.generate_json(system_prompt="Return JSON\n\nCOMPACT EXTRACTION MODE:", user_prompt="hello")

    assert len(calls) == 2
    assert "COMPACT EXTRACTION MODE" in calls[0]
    assert "CORE ACTION MODE" in calls[1]
    assert "COMPACT EXTRACTION MODE" not in calls[1]
    assert CORE_ACTION_MODE_WARNING in result


def test_core_success_does_not_trigger_emergency(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    calls: list[str] = []

    def fake_create(
        *, system_prompt: str, user_prompt: str, use_json_mode: bool, attempt: int, analysis_mode: str
    ) -> SimpleNamespace:
        calls.append(analysis_mode)
        if len(calls) == 1:
            return _response(content="", finish_reason="length")
        return _response(content='{"tasks": [], "source_language": "English", "confidence": 0.0, "warnings": []}')

    monkeypatch.setattr(provider, "_create_completion", fake_create)

    provider.generate_json(system_prompt="Return JSON", user_prompt="hello")

    assert calls == ["normal", "core"]


def test_core_length_then_emergency_success_uses_at_most_three_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    calls: list[str] = []

    def fake_create(
        *, system_prompt: str, user_prompt: str, use_json_mode: bool, attempt: int, analysis_mode: str
    ) -> SimpleNamespace:
        calls.append(analysis_mode)
        if analysis_mode in {"normal", "core"}:
            return _response(content="", finish_reason="length")
        return _response(
            content='{"title": "比赛通知", "deadline": null, "must_do": ["报名"], "materials": ["作品"], "questions": [], "warnings": []}'
        )

    monkeypatch.setattr(provider, "_create_completion", fake_create)

    result = provider.generate_json(system_prompt="Return JSON", user_prompt="hello")

    assert calls == ["normal", "core", "emergency"]
    assert len(calls) <= 3
    assert EMERGENCY_EXTRACTION_WARNING in result


def test_emergency_length_stops_with_truncated_error(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    calls: list[str] = []

    def fake_create(
        *, system_prompt: str, user_prompt: str, use_json_mode: bool, attempt: int, analysis_mode: str
    ) -> SimpleNamespace:
        calls.append(analysis_mode)
        return _response(content="", finish_reason="length")

    monkeypatch.setattr(provider, "_create_completion", fake_create)

    with pytest.raises(LLMTruncatedResponseError):
        provider.generate_json(system_prompt="Return JSON", user_prompt="hello")

    assert calls == ["normal", "core", "emergency"]
    assert len(calls) == 3


def test_finish_reason_content_filter_raises_filtered_error() -> None:
    provider = _provider()

    with pytest.raises(LLMFilteredResponseError):
        provider._extract_json_content(
            _response(content="{}", finish_reason="content_filter"),
            attempt=1,
            use_json_mode=True,
        )


def test_insufficient_system_resource_retries_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    calls = 0

    def fake_create(
        *, system_prompt: str, user_prompt: str, use_json_mode: bool, attempt: int, analysis_mode: str
    ) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _response(content="", finish_reason="insufficient_system_resource")
        return _response(content='{"tasks": [], "source_language": "English", "confidence": 0.0, "warnings": []}')

    monkeypatch.setattr(provider, "_create_completion", fake_create)

    result = provider.generate_json(system_prompt="Return JSON", user_prompt="hello")

    assert calls == 2
    assert '"tasks"' in result


def test_fallback_parses_markdown_json_fence() -> None:
    provider = _provider()

    result = provider._extract_json_from_fallback(
        '```json\n{"tasks": [], "source_language": "English", "confidence": 0.0, "warnings": []}\n```'
    )

    assert result.startswith("{")
    assert result.endswith("}")


def test_fallback_invalid_json_fails() -> None:
    provider = _provider()

    with pytest.raises(LLMEmptyResponseError):
        provider._extract_json_from_fallback("{not valid json}")


def test_non_task_json_is_valid_non_empty_content() -> None:
    provider = _provider()
    response = _response(
        content='{"tasks": [], "source_language": "English", "confidence": 0.0, "warnings": ["No actionable task was identified."]}'
    )

    result = provider._extract_json_content(response, attempt=1, use_json_mode=True)

    assert '"tasks": []' in result


def test_response_metadata_log_does_not_include_api_key_or_full_input(caplog: pytest.LogCaptureFixture) -> None:
    provider = _provider()
    response = _response(content="{}", reasoning_content="private reasoning")

    with caplog.at_level(logging.INFO, logger="src.providers.openai_compatible"):
        provider._extract_json_content(response, attempt=1, use_json_mode=True)

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "sk-secret" not in logs
    assert "full user input" not in logs
    assert "request_attempt=1" in logs
    assert "analysis_mode=unknown" in logs
    assert "response_format_enabled=True" in logs
    assert "reasoning_content_chars=17" in logs
    assert "prompt_tokens=10" in logs
