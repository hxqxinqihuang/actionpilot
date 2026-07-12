from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, OpenAIError

from src.agent.prompts import (
    build_core_action_system_prompt,
    build_core_action_user_prompt,
    build_emergency_system_prompt,
    build_emergency_user_prompt,
)
from src.analysis_policy import CORE_ACTION_MODE_WARNING, EMERGENCY_EXTRACTION_WARNING
from src.config import AppConfig
from src.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider request fails."""


class LLMProviderTimeoutError(LLMProviderError):
    """Raised when an LLM provider request times out after retry."""


class LLMEmptyResponseError(LLMProviderError):
    """Raised when JSON mode returns empty content and recovery fails."""


class LLMTruncatedResponseError(LLMProviderError):
    """Raised when the model output is truncated by token limits."""


class LLMFilteredResponseError(LLMProviderError):
    """Raised when the model output is blocked by a content filter."""


class LLMInsufficientResourceError(LLMProviderError):
    """Raised when the provider reports insufficient system resources after retry."""


class LLMNoChoicesError(LLMProviderError):
    """Raised when a model response contains no choices."""


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, config: AppConfig) -> None:
        self._model = config.model
        self._max_tokens = config.max_tokens
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=httpx.Timeout(connect=10.0, read=90.0, write=20.0, pool=10.0),
            max_retries=0,
        )

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> str:
        last_retryable_error: OpenAIError | None = None
        for attempt in range(2):
            try:
                return self._generate_json_once(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    attempt=attempt + 1,
                    use_json_mode=True,
                    analysis_mode=self._infer_analysis_mode(system_prompt),
                )
            except OpenAIError as exc:
                if not self._is_retryable_error(exc):
                    raise
                last_retryable_error = exc
                if attempt == 1:
                    break
                time.sleep(0.5)
            except LLMEmptyResponseError:
                if attempt == 1:
                    raise
                return self._generate_json_once(
                    system_prompt=self._build_fallback_system_prompt(system_prompt),
                    user_prompt=user_prompt,
                    attempt=attempt + 2,
                    use_json_mode=False,
                    analysis_mode=self._infer_analysis_mode(system_prompt),
                )
            except LLMInsufficientResourceError:
                if attempt == 1:
                    raise
                time.sleep(0.5)
            except LLMTruncatedResponseError:
                if self._is_core_action_system_prompt(system_prompt):
                    return self._generate_emergency_json(
                        source_text=user_prompt,
                        attempt=attempt + 1,
                    )
                if attempt == 1:
                    raise
                return self._generate_core_or_emergency_json(
                    source_text=user_prompt,
                    attempt=attempt + 2,
                )
        if isinstance(last_retryable_error, APITimeoutError):
            raise LLMProviderTimeoutError(
                f"LLM request timed out after retry: {last_retryable_error}"
            ) from last_retryable_error
        raise LLMProviderError(f"LLM request failed after retry: {last_retryable_error}") from last_retryable_error

    def _generate_json_once(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        attempt: int,
        use_json_mode: bool,
        analysis_mode: str,
    ) -> str:
        try:
            response = self._create_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                use_json_mode=use_json_mode,
                attempt=attempt,
                analysis_mode=analysis_mode,
            )
        except OpenAIError as exc:
            if self._is_retryable_error(exc):
                raise
            raise LLMProviderError(f"LLM request failed: {exc}") from exc

        return self._extract_json_content(
            response,
            attempt=attempt,
            use_json_mode=use_json_mode,
            analysis_mode=analysis_mode,
            system_prompt_chars=len(system_prompt),
            user_prompt_chars=len(user_prompt),
        )

    def _create_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        use_json_mode: bool,
        attempt: int,
        analysis_mode: str,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": self._max_tokens,
        }
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return self._client.chat.completions.create(**kwargs)

    def _generate_core_or_emergency_json(self, *, source_text: str, attempt: int) -> str:
        try:
            core_json = self._generate_json_once(
                system_prompt=build_core_action_system_prompt(),
                user_prompt=build_core_action_user_prompt(source_text),
                attempt=attempt,
                use_json_mode=True,
                analysis_mode="core",
            )
            return self._add_warning_to_json(core_json, CORE_ACTION_MODE_WARNING)
        except LLMTruncatedResponseError:
            return self._generate_emergency_json(source_text=source_text, attempt=attempt + 1)

    def _generate_emergency_json(self, *, source_text: str, attempt: int) -> str:
        emergency_json = self._generate_json_once(
            system_prompt=build_emergency_system_prompt(),
            user_prompt=build_emergency_user_prompt(source_text),
            attempt=attempt,
            use_json_mode=True,
            analysis_mode="emergency",
        )
        return self._convert_emergency_json_to_task_result(emergency_json)

    def _extract_json_content(
        self,
        response: Any,
        *,
        attempt: int,
        use_json_mode: bool,
        analysis_mode: str = "unknown",
        system_prompt_chars: int | None = None,
        user_prompt_chars: int | None = None,
    ) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            self._log_response_metadata(
                response,
                None,
                attempt,
                analysis_mode=analysis_mode,
                system_prompt_chars=system_prompt_chars,
                user_prompt_chars=user_prompt_chars,
                response_format_enabled=use_json_mode,
            )
            raise LLMNoChoicesError("LLM response contained no choices.")

        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        self._log_response_metadata(
            response,
            choice,
            attempt,
            analysis_mode=analysis_mode,
            system_prompt_chars=system_prompt_chars,
            user_prompt_chars=user_prompt_chars,
            response_format_enabled=use_json_mode,
        )

        if finish_reason == "length":
            raise LLMTruncatedResponseError("LLM output was truncated before valid JSON was completed.")
        if finish_reason == "content_filter":
            raise LLMFilteredResponseError("LLM output was blocked by the content filter.")
        if finish_reason == "insufficient_system_resource":
            raise LLMInsufficientResourceError("LLM provider reported insufficient system resources.")

        if not content:
            raise LLMEmptyResponseError("LLM returned an empty JSON response.")

        if use_json_mode:
            return content
        return self._extract_json_from_fallback(content)

    def _is_retryable_error(self, exc: OpenAIError) -> bool:
        if isinstance(exc, (APITimeoutError, APIConnectionError)):
            return True
        if isinstance(exc, APIStatusError):
            return exc.status_code >= 500
        return False

    def _log_response_metadata(
        self,
        response: Any,
        choice: Any | None,
        attempt: int,
        *,
        analysis_mode: str,
        system_prompt_chars: int | None,
        user_prompt_chars: int | None,
        response_format_enabled: bool,
    ) -> None:
        message = getattr(choice, "message", None) if choice is not None else None
        content = getattr(message, "content", None) if message is not None else None
        reasoning_content = getattr(message, "reasoning_content", None) if message is not None else None
        usage = getattr(response, "usage", None)
        logger.info(
            "LLM response metadata: request_attempt=%s analysis_mode=%s system_prompt_chars=%s "
            "user_prompt_chars=%s response_format_enabled=%s model=%s finish_reason=%s "
            "content_is_none=%s content_chars=%s reasoning_content_chars=%s prompt_tokens=%s "
            "completion_tokens=%s total_tokens=%s",
            attempt,
            analysis_mode,
            system_prompt_chars,
            user_prompt_chars,
            response_format_enabled,
            getattr(response, "model", None),
            getattr(choice, "finish_reason", None) if choice is not None else None,
            content is None,
            len(content or ""),
            len(reasoning_content or ""),
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
            getattr(usage, "total_tokens", None),
        )

    def _extract_json_from_fallback(self, content: str) -> str:
        stripped = content.strip()
        fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
        if fence_match:
            stripped = fence_match.group(1).strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise LLMEmptyResponseError("Fallback response did not contain a JSON object.")
        candidate = stripped[start : end + 1]
        try:
            json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise LLMEmptyResponseError("Fallback response did not contain valid JSON.") from exc
        return candidate

    def _build_fallback_system_prompt(self, system_prompt: str) -> str:
        return (
            f"{system_prompt}\n\n"
            "JSON MODE RECOVERY: Return one non-empty JSON object only. Do not use Markdown fences. "
            "The first character must be { and the last character must be }. "
            "If no actionable task is found, return {\"tasks\": [], \"source_language\": null, "
            "\"confidence\": 0.0, \"warnings\": [\"No actionable task was identified.\"]}."
        )

    def _is_compact_system_prompt(self, system_prompt: str) -> bool:
        return "COMPACT EXTRACTION MODE" in system_prompt

    def _is_core_action_system_prompt(self, system_prompt: str) -> bool:
        return "CORE ACTION MODE" in system_prompt

    def _infer_analysis_mode(self, system_prompt: str) -> str:
        if "EMERGENCY EXTRACTION MODE" in system_prompt:
            return "emergency"
        if self._is_core_action_system_prompt(system_prompt):
            return "core"
        if self._is_compact_system_prompt(system_prompt):
            return "compact"
        return "normal"

    def _add_warning_to_json(self, raw_json: str, warning: str) -> str:
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            return raw_json
        warnings = payload.setdefault("warnings", [])
        if isinstance(warnings, list) and warning not in warnings:
            warnings.append(warning)
        return json.dumps(payload, ensure_ascii=False)

    def _convert_emergency_json_to_task_result(self, raw_json: str) -> str:
        try:
            emergency = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise LLMProviderError("Emergency extraction returned invalid JSON.") from exc

        title = str(emergency.get("title") or "Untitled")
        deadline = emergency.get("deadline")
        deadlines_value = emergency.get("deadlines", deadline)
        must_do = self._string_list(emergency.get("must_do"))
        materials = self._string_list(emergency.get("materials"))
        questions = self._string_list(emergency.get("questions"))
        warnings = self._string_list(emergency.get("warnings"))
        if EMERGENCY_EXTRACTION_WARNING not in warnings:
            warnings.append(EMERGENCY_EXTRACTION_WARNING)

        deadlines = self._emergency_deadlines(
            deadlines_value,
            fallback_texts=[
                title,
                str(emergency.get("summary") or ""),
                *must_do,
                *materials,
                *warnings,
            ],
        )

        task_result = {
            "tasks": [
                {
                    "title": title,
                    "summary": title,
                    "deadlines": deadlines,
                    "prerequisites": [],
                    "materials": [
                        {"name": item, "description": None, "required": True, "evidence": None}
                        for item in materials
                    ],
                    "requirements": [
                        {"description": item, "priority": "must", "evidence": None}
                        for item in must_do
                    ],
                    "risks": [],
                    "confirmation_questions": [
                        {"question": item, "reason": "Emergency extraction question."}
                        for item in questions
                    ],
                }
            ],
            "source_language": None,
            "confidence": 0.0,
            "warnings": warnings,
        }
        return json.dumps(task_result, ensure_ascii=False)

    def _string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if item is not None and str(item).strip()]

    def _emergency_deadlines(self, value: Any, *, fallback_texts: list[str] | None = None) -> list[dict[str, Any]]:
        values = self._deadline_values(value)
        deadlines: list[dict[str, Any]] = []
        for item in values:
            if not isinstance(item, dict):
                iso_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", str(item))
                if len(iso_dates) > 1:
                    deadlines.extend(
                        {
                            "raw_text": iso_date,
                            "normalized_date": iso_date,
                            "timezone": None,
                            "status": "found",
                            "type": self._deadline_type(str(item)),
                            "evidence": None,
                        }
                        for iso_date in iso_dates
                    )
                    continue
            deadline = self._emergency_deadline_item(item)
            if deadline is not None:
                deadlines.append(deadline)
        if deadlines:
            return deadlines

        fallback_deadlines = self._extract_deadlines_from_texts(fallback_texts or [])
        return fallback_deadlines or [
            {
                "raw_text": None,
                "normalized_date": None,
                "timezone": None,
                "status": "missing",
                "type": "unknown",
                "evidence": None,
            }
        ]

    def _deadline_values(self, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _emergency_deadline_item(self, value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            raw_text = value.get("raw_text") or value.get("text") or value.get("deadline")
            normalized_date = value.get("normalized_date") or value.get("date")
            parsed_date = self._extract_iso_date(str(normalized_date or raw_text or ""))
            if parsed_date:
                raw_text_value = str(raw_text or normalized_date or parsed_date)
                return {
                    "raw_text": raw_text_value,
                    "normalized_date": parsed_date,
                    "timezone": None,
                    "status": "found",
                    "type": self._deadline_type(raw_text_value, explicit=value.get("type")),
                    "evidence": None,
                }
            if raw_text:
                raw_text_value = str(raw_text)
                return {
                    "raw_text": raw_text_value,
                    "normalized_date": None,
                    "timezone": None,
                    "status": "ambiguous",
                    "type": self._deadline_type(raw_text_value, explicit=value.get("type")),
                    "evidence": None,
                }
            return None

        text = str(value).strip()
        if not text:
            return None
        iso_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
        if iso_dates:
            return {
                "raw_text": text,
                "normalized_date": iso_dates[0],
                "timezone": None,
                "status": "found",
                "type": self._deadline_type(text),
                "evidence": None,
            }
        return {
            "raw_text": text,
            "normalized_date": None,
            "timezone": None,
            "status": "ambiguous",
            "type": self._deadline_type(text),
            "evidence": None,
        }

    def _extract_iso_date(self, value: str) -> str | None:
        match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", value)
        return match.group(0) if match else None

    def _extract_deadlines_from_texts(self, texts: list[str]) -> list[dict[str, Any]]:
        deadlines: list[dict[str, Any]] = []
        seen: set[str] = set()
        for text in texts:
            for raw_text, normalized_date in self._date_candidates_from_text(text):
                if normalized_date in seen:
                    continue
                seen.add(normalized_date)
                deadlines.append(
                    {
                        "raw_text": raw_text,
                        "normalized_date": normalized_date,
                        "timezone": None,
                        "status": "found",
                        "type": self._deadline_type(raw_text),
                        "evidence": None,
                    }
                )
        return deadlines

    def _date_candidates_from_text(self, text: str) -> list[tuple[str, str]]:
        if not text:
            return []
        candidates: list[tuple[str, str]] = []
        candidates.extend((match.group(0), match.group(0)) for match in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", text))
        occupied_spans: list[tuple[int, int]] = []

        range_pattern = re.compile(
            r"(?P<year>\d{4})年(?P<start_month>\d{1,2})月(?P<start_day>\d{1,2})日?\s*"
            r"(?:至|到|—|-|~|～)\s*(?:(?P<end_year>\d{4})年)?(?P<end_month>\d{1,2})月(?P<end_day>\d{1,2})日?"
        )
        for match in range_pattern.finditer(text):
            year = int(match.group("end_year") or match.group("year"))
            occupied_spans.append(match.span())
            candidates.append((text.strip(), f"{year:04d}-{int(match.group('end_month')):02d}-{int(match.group('end_day')):02d}"))

        chinese_date_pattern = re.compile(r"(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日?")
        for match in chinese_date_pattern.finditer(text):
            if self._span_inside_any(match.span(), occupied_spans):
                continue
            candidates.append((text.strip(), f"{int(match.group('year')):04d}-{int(match.group('month')):02d}-{int(match.group('day')):02d}"))

        month_day_pattern = re.compile(r"(?P<month>\d{1,2})月(?P<day>\d{1,2})日?(?:前|截止|之前)?")
        current_year = self._first_year_in_text(text)
        if current_year is not None:
            for match in month_day_pattern.finditer(text):
                if self._span_inside_any(match.span(), occupied_spans):
                    continue
                candidates.append((text.strip(), f"{current_year:04d}-{int(match.group('month')):02d}-{int(match.group('day')):02d}"))
        return candidates

    def _span_inside_any(self, span: tuple[int, int], occupied_spans: list[tuple[int, int]]) -> bool:
        start, end = span
        return any(start >= occupied_start and end <= occupied_end for occupied_start, occupied_end in occupied_spans)

    def _first_year_in_text(self, text: str) -> int | None:
        match = re.search(r"\b(20\d{2})\b|(?P<cn_year>20\d{2})年", text)
        if not match:
            return None
        value = match.group(1) or match.group("cn_year")
        return int(value) if value else None

    def _deadline_type(self, text: str, *, explicit: Any = None) -> str:
        explicit_value = str(explicit or "").strip().lower()
        if explicit_value in {"registration", "submission", "other", "unknown"}:
            return explicit_value

        lowered = text.lower()
        if any(keyword in lowered for keyword in ("报名", "注册", "signup", "sign up", "registration", "register")):
            return "registration"
        if any(
            keyword in lowered
            for keyword in ("作品", "提交", "材料", "邮件", "发送", "submission", "submit", "email", "work")
        ):
            return "submission"
        if any(keyword in lowered for keyword in ("初审", "终审", "评审", "review", "schedule", "time")):
            return "other"
        return "unknown"
