from __future__ import annotations

from src.parsers import NoExtractableTextError
from src.providers.openai_compatible import LLMProviderError, LLMProviderTimeoutError
from src.ui_state import (
    MAX_INPUT_CHARS,
    STANDARD_MODE_MAX_CHARS,
    build_file_signature,
    can_call_provider,
    check_input_length,
    format_parse_error,
    format_provider_error,
    format_plan_error,
    has_analyzable_text,
    should_clear_analysis,
)


def test_has_analyzable_text_rejects_blank_text() -> None:
    assert has_analyzable_text("  \n") is False
    assert has_analyzable_text("课程项目通知") is True


def test_should_clear_analysis_only_when_text_changes() -> None:
    assert should_clear_analysis("old", "new") is True
    assert should_clear_analysis("same", "same") is False


def test_build_file_signature_is_case_insensitive_for_name_and_sensitive_to_content() -> None:
    first = build_file_signature("NOTICE.PDF", b"abc")
    second = build_file_signature("notice.pdf", b"abc")
    third = build_file_signature("notice.pdf", b"abcd")

    assert first == second
    assert second != third


def test_format_parse_error_mentions_ocr_for_no_extractable_pdf_text() -> None:
    message = format_parse_error(NoExtractableTextError("empty"))

    assert "scanned PDF" in message
    assert "OCR is not supported" in message


def test_input_length_boundaries() -> None:
    standard = check_input_length("a" * STANDARD_MODE_MAX_CHARS)
    long_allowed = check_input_length("a" * (STANDARD_MODE_MAX_CHARS + 1))
    max_allowed = check_input_length("a" * MAX_INPUT_CHARS)
    too_long = check_input_length("a" * (MAX_INPUT_CHARS + 1))

    assert standard.allowed is True
    assert standard.warning is None
    assert long_allowed.allowed is True
    assert long_allowed.warning is not None
    assert max_allowed.allowed is True
    assert too_long.allowed is False
    assert too_long.error is not None


def test_over_max_length_cannot_call_provider() -> None:
    assert can_call_provider("a" * MAX_INPUT_CHARS) is True
    assert can_call_provider("a" * (MAX_INPUT_CHARS + 1)) is False
    assert can_call_provider("   ") is False


def test_provider_error_mapping() -> None:
    assert "timed out" in format_provider_error(LLMProviderTimeoutError("timeout"))
    assert "API Key" in format_provider_error(LLMProviderError("Error code: 403"))
    assert "model service" in format_provider_error(LLMProviderError("server unavailable"))


def test_input_change_clears_old_plan_without_erasing_new_input() -> None:
    from src.ui_state import apply_input_change

    state: dict[str, object] = {
        "current_input_text": "old",
        "input_source": "paste",
        "analysis_result": object(),
        "analysis_error": "old error",
        "action_plan_results": {"old": object()},
        "action_plan_errors": {"old": "error"},
    }

    apply_input_change(state, "new text", "paste")

    assert state["current_input_text"] == "new text"
    assert state["analysis_result"] is None
    assert state["analysis_error"] is None
    assert state["action_plan_results"] == {}
    assert state["action_plan_errors"] == {}


def test_plan_error_mapping_keeps_analysis_result_safe() -> None:
    assert "plan request timed out" in format_plan_error(LLMProviderTimeoutError("timeout"))
