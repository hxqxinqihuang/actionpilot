from __future__ import annotations

import hashlib
from collections.abc import MutableMapping
from json import JSONDecodeError
from typing import Any

from src.analysis_policy import (
    COMPACT_RECOMMENDED_MAX_CHARS,
    MAX_INPUT_CHARS,
    STANDARD_MODE_MAX_CHARS,
    InputLengthCheck,
    check_input_length,
)
from src.parsers import NoExtractableTextError
from src.parsers.exceptions import FileParseError
from src.providers.openai_compatible import (
    LLMEmptyResponseError,
    LLMFilteredResponseError,
    LLMInsufficientResourceError,
    LLMProviderError,
    LLMProviderTimeoutError,
    LLMTruncatedResponseError,
)

def has_analyzable_text(text: str) -> bool:
    return bool(text.strip())


def can_call_provider(text: str) -> bool:
    return has_analyzable_text(text) and check_input_length(text).allowed


def should_clear_analysis(previous_text: str | None, next_text: str) -> bool:
    return previous_text != next_text


def build_file_signature(file_name: str, file_bytes: bytes) -> str:
    digest = hashlib.sha256(file_bytes).hexdigest()
    return f"{file_name.lower()}:{len(file_bytes)}:{digest}"


def format_parse_error(exc: FileParseError) -> str:
    if isinstance(exc, NoExtractableTextError):
        return "PDF has no extractable text. It may be a scanned PDF. OCR is not supported yet."
    return str(exc)


def format_provider_error(exc: LLMProviderError) -> str:
    if isinstance(exc, LLMProviderTimeoutError):
        return "The model request timed out. The document may be long or the service may be busy."
    if isinstance(exc, LLMEmptyResponseError):
        return "The model did not return valid content. Please retry. Your input has not been lost."
    if isinstance(exc, LLMTruncatedResponseError):
        return (
            "The document contains many details and the model output exceeded the limit. "
            "Please remove appendices, reference links, product descriptions, or unrelated sections and try again."
        )
    if isinstance(exc, LLMFilteredResponseError):
        return "The model could not process part of the content. Please keep only task-notice-related sections."
    if isinstance(exc, LLMInsufficientResourceError):
        return "The model service is currently busy. Please retry later."

    message = str(exc).lower()
    if "401" in message or "403" in message or "api key" in message or "authentication" in message:
        return "Model authentication failed. Please check your API Key and provider settings."
    return "The model service returned an error. Please try again later."


def format_plan_error(exc: Exception) -> str:
    if isinstance(exc, JSONDecodeError):
        return "The model returned invalid plan JSON. Please retry."
    if isinstance(exc, LLMProviderTimeoutError):
        return "The plan request timed out. Please retry or reduce optional details."
    if isinstance(exc, LLMEmptyResponseError):
        return "The model did not return a valid plan. Please retry. Your analysis result was kept."
    if isinstance(exc, LLMTruncatedResponseError):
        return "The generated plan was too long. Please simplify the task or shorten the planning period."
    if isinstance(exc, LLMFilteredResponseError):
        return "The model could not process part of the planning input. Please keep only task-related content."
    if isinstance(exc, LLMInsufficientResourceError):
        return "The model service is currently busy. Please retry later."
    if isinstance(exc, LLMProviderError):
        return format_provider_error(exc)
    return str(exc) or "Action plan generation failed."


def input_sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def apply_input_change(state: MutableMapping[str, Any], text: str, source: str) -> None:
    previous_text = state.get("current_input_text")
    previous_source = state.get("input_source")
    if previous_text == text and previous_source == source:
        return

    state["current_input_text"] = text
    state["input_source"] = source
    state["analysis_result"] = None
    state["analysis_error"] = None
    state["action_plan_results"] = {}
    state["action_plan_errors"] = {}
