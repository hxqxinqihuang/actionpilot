from __future__ import annotations

from openai import OpenAI, OpenAIError

from src.config import AppConfig
from src.providers.base import LLMProvider


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider request fails."""


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, config: AppConfig) -> None:
        self._model = config.model
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
        except OpenAIError as exc:
            raise LLMProviderError(f"LLM request failed: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMProviderError("LLM returned an empty response.")
        return content
