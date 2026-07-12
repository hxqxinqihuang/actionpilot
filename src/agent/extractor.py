from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from pydantic import ValidationError

from src.agent.prompts import SYSTEM_PROMPT, build_compact_system_prompt, build_extraction_prompt
from src.analysis_policy import COMPACT_MODE_WARNING, choose_extraction_mode
from src.providers.base import LLMProvider
from src.schemas.task import Deadline, Material, Prerequisite, Requirement, TaskExtractionResult


class ExtractionError(RuntimeError):
    """Raised when model output cannot be converted into a task schema."""


class TaskExtractor:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def extract(self, text: str) -> TaskExtractionResult:
        mode = choose_extraction_mode(text)
        system_prompt = build_compact_system_prompt() if mode == "compact" else SYSTEM_PROMPT
        raw_json = self._provider.generate_json(
            system_prompt=system_prompt,
            user_prompt=build_extraction_prompt(text),
        )
        try:
            result = TaskExtractionResult.model_validate_json(raw_json)
        except ValidationError as exc:
            raise ExtractionError("LLM response did not match the expected schema.") from exc
        if mode == "compact" and COMPACT_MODE_WARNING not in result.warnings:
            result.warnings.append(COMPACT_MODE_WARNING)
        self._validate_source_evidence(result, text)
        return result

    def _validate_source_evidence(self, result: TaskExtractionResult, source_text: str) -> None:
        normalized_source = self._normalize_for_evidence(source_text)
        unverified_ordinary = 0
        verified_ordinary = 0
        deadline_failures = 0
        for task_index, task in enumerate(result.tasks):
            for item_index, deadline in enumerate(task.deadlines):
                if not self._validate_deadline_source_text(
                    deadline,
                    normalized_source,
                    result,
                    task_index,
                    item_index,
                ):
                    deadline_failures += 1
            for collection_name, items in (
                ("prerequisites", task.prerequisites),
                ("materials", task.materials),
                ("requirements", task.requirements),
            ):
                for item_index, item in enumerate(items):
                    evidence = self._get_evidence(item)
                    if evidence is None:
                        continue
                    if self._evidence_matches(evidence, normalized_source):
                        item.evidence_status = "verified"
                        verified_ordinary += 1
                    else:
                        item.evidence_status = "unverified"
                        item.evidence = None
                        unverified_ordinary += 1
                        result.warnings.append(
                            f"Evidence for tasks[{task_index}].{collection_name}[{item_index}] "
                            "could not be verified against the source text and was removed."
                        )

        if deadline_failures >= 3:
            raise ExtractionError("Too many deadline evidence items could not be verified.")
        if unverified_ordinary >= 5 and verified_ordinary == 0:
            raise ExtractionError("Too many evidence items could not be verified against the source text.")

    def _validate_deadline_source_text(
        self,
        deadline: Deadline,
        normalized_source: str,
        result: TaskExtractionResult,
        task_index: int,
        item_index: int,
    ) -> bool:
        if deadline.status == "missing":
            deadline.evidence_status = "verified"
            return True

        raw_text_valid = deadline.raw_text is None or self._evidence_matches(deadline.raw_text, normalized_source)
        evidence_valid = deadline.evidence is None or self._evidence_matches(deadline.evidence, normalized_source)

        if raw_text_valid and evidence_valid:
            deadline.evidence_status = "verified"
            return True

        deadline.evidence_status = "unverified"
        deadline.evidence = None
        deadline.raw_text = deadline.raw_text if raw_text_valid else None
        deadline.normalized_date = None
        deadline.timezone = None
        deadline.status = "ambiguous" if deadline.raw_text else "missing"
        result.warnings.append(
            f"Deadline evidence for tasks[{task_index}].deadlines[{item_index}] "
            "could not be verified and was downgraded."
        )
        return False

    def _get_evidence(self, item: Prerequisite | Material | Requirement) -> str | None:
        return item.evidence

    def _normalize_for_evidence(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"\s+", "", normalized)
        return normalized

    def _evidence_matches(self, evidence: str, normalized_source: str) -> bool:
        normalized_evidence = self._normalize_for_evidence(evidence)
        if not normalized_evidence:
            return True
        if normalized_evidence in normalized_source:
            return True
        return self._short_evidence_fuzzy_match(normalized_evidence, normalized_source)

    def _short_evidence_fuzzy_match(self, normalized_evidence: str, normalized_source: str) -> bool:
        evidence_length = len(normalized_evidence)
        if evidence_length < 8 or evidence_length > 80:
            return False
        if evidence_length > len(normalized_source):
            return False

        threshold = 0.95
        for start in range(0, len(normalized_source) - evidence_length + 1):
            window = normalized_source[start : start + evidence_length]
            if SequenceMatcher(None, normalized_evidence, window).ratio() >= threshold:
                return True
        return False
