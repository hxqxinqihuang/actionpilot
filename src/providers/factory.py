from __future__ import annotations

from src.config import AppConfig
from src.providers.base import LLMProvider
from src.providers.openai_compatible import OpenAICompatibleProvider


def create_llm_provider(config: AppConfig) -> LLMProvider:
    return OpenAICompatibleProvider(config)
