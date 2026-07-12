from __future__ import annotations

import httpx
import pytest
from openai import APITimeoutError

from src.providers.openai_compatible import LLMProviderTimeoutError, OpenAICompatibleProvider


def _timeout_error() -> APITimeoutError:
    return APITimeoutError(request=httpx.Request("POST", "https://api.example.test"))


def test_timeout_retries_once_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = object.__new__(OpenAICompatibleProvider)
    calls = 0

    def fake_once(*, system_prompt: str, user_prompt: str, attempt: int, use_json_mode: bool, analysis_mode: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _timeout_error()
        return '{"tasks": [], "source_language": "English", "confidence": 0.8}'

    monkeypatch.setattr(provider, "_generate_json_once", fake_once)

    result = provider.generate_json(system_prompt="json", user_prompt="text")

    assert calls == 2
    assert '"tasks"' in result


def test_second_timeout_stops_after_one_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = object.__new__(OpenAICompatibleProvider)
    calls = 0

    def fake_once(*, system_prompt: str, user_prompt: str, attempt: int, use_json_mode: bool, analysis_mode: str) -> str:
        nonlocal calls
        calls += 1
        raise _timeout_error()

    monkeypatch.setattr(provider, "_generate_json_once", fake_once)

    with pytest.raises(LLMProviderTimeoutError):
        provider.generate_json(system_prompt="json", user_prompt="text")

    assert calls == 2
