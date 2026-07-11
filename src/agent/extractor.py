from __future__ import annotations

from pydantic import ValidationError

from src.agent.prompts import SYSTEM_PROMPT, build_extraction_prompt
from src.providers.base import LLMProvider
from src.schemas.task import Deadline, Material, Prerequisite, Requirement, TaskExtractionResult


class ExtractionError(RuntimeError):
    """Raised when model output cannot be converted into a task schema."""


class TaskExtractor:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def extract(self, text: str) -> TaskExtractionResult:
        raw_json = self._provider.generate_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_extraction_prompt(text),
        )
        try:
            result = TaskExtractionResult.model_validate_json(raw_json)
        except ValidationError as exc:
            raise ExtractionError("LLM response did not match the expected schema.") from exc
        self._validate_source_evidence(result, text)
        return result

    def _validate_source_evidence(self, result: TaskExtractionResult, source_text: str) -> None:
        for task_index, task in enumerate(result.tasks):
            for item_index, deadline in enumerate(task.deadlines):
                self._validate_deadline_source_text(deadline, source_text, task_index, item_index)
            for collection_name, items in (
                ("prerequisites", task.prerequisites),
                ("materials", task.materials),
                ("requirements", task.requirements),
            ):
                for item_index, item in enumerate(items):
                    evidence = self._get_evidence(item)
                    if evidence is not None and evidence not in source_text:
                        raise ExtractionError(
                            f"Evidence for tasks[{task_index}].{collection_name}[{item_index}] "
                            "was not found in the source text."
                        )

    def _validate_deadline_source_text(
        self,
        deadline: Deadline,
        source_text: str,
        task_index: int,
        item_index: int,
    ) -> None:
        if deadline.status == "missing":
            return

        if deadline.raw_text is not None and deadline.raw_text not in source_text:
            raise ExtractionError(
                f"Deadline raw_text for tasks[{task_index}].deadlines[{item_index}] "
                "was not found in the source text."
            )

        if deadline.evidence is not None and deadline.evidence not in source_text:
            raise ExtractionError(
                f"Deadline evidence for tasks[{task_index}].deadlines[{item_index}] "
                "was not found in the source text."
            )

    def _get_evidence(self, item: Prerequisite | Material | Requirement) -> str | None:
        return item.evidence
