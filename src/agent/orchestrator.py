from __future__ import annotations

from src.agent.extractor import TaskExtractor
from src.providers.base import LLMProvider
from src.schemas.task import TaskExtractionResult


class ExtractionOrchestrator:
    def __init__(self, provider: LLMProvider) -> None:
        self._extractor = TaskExtractor(provider)

    def extract(self, text: str) -> TaskExtractionResult:
        return self._extractor.extract(text)
