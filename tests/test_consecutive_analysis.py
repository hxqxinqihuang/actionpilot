from __future__ import annotations

from src.agent.orchestrator import ExtractionOrchestrator
from src.providers.base import LLMProvider
from src.ui_state import apply_input_change


class SequentialProvider(LLMProvider):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(user_prompt)
        title = f"Task {len(self.calls)}"
        summary = "First input" if len(self.calls) == 1 else "Second input"
        return f"""
        {{
          "tasks": [
            {{
              "title": "{title}",
              "summary": "{summary}",
              "deadlines": [{{"raw_text": null, "normalized_date": null, "timezone": null, "status": "missing", "evidence": null}}],
              "prerequisites": [],
              "materials": [],
              "requirements": [],
              "risks": [],
              "confirmation_questions": []
            }}
          ],
          "source_language": "English",
          "confidence": 0.8
        }}
        """


def test_same_orchestrator_can_analyze_two_different_texts_without_reusing_result() -> None:
    provider = SequentialProvider()
    orchestrator = ExtractionOrchestrator(provider)

    first = orchestrator.extract("first text")
    second = orchestrator.extract("second text")

    assert len(provider.calls) == 2
    assert first.tasks[0].title == "Task 1"
    assert second.tasks[0].title == "Task 2"
    assert first.tasks[0].summary != second.tasks[0].summary
    assert "first text" in provider.calls[0]
    assert "second text" in provider.calls[1]


def test_input_change_clears_analysis_result_and_error_but_keeps_new_input() -> None:
    state = {
        "current_input_text": "old",
        "input_source": "paste",
        "analysis_result": object(),
        "analysis_error": "old error",
    }

    apply_input_change(state, "new", "paste")

    assert state["current_input_text"] == "new"
    assert state["input_source"] == "paste"
    assert state["analysis_result"] is None
    assert state["analysis_error"] is None


def test_same_input_and_source_does_not_clear_analysis_result() -> None:
    result = object()
    state = {
        "current_input_text": "same",
        "input_source": "paste",
        "analysis_result": result,
        "analysis_error": None,
    }

    apply_input_change(state, "same", "paste")

    assert state["current_input_text"] == "same"
    assert state["analysis_result"] is result


def test_paste_and_upload_sources_can_alternate_analysis_state() -> None:
    state = {
        "current_input_text": "",
        "input_source": None,
        "analysis_result": None,
        "analysis_error": None,
    }

    apply_input_change(state, "paste text", "paste")
    state["analysis_result"] = "paste result"
    apply_input_change(state, "upload text", "upload")

    assert state["current_input_text"] == "upload text"
    assert state["input_source"] == "upload"
    assert state["analysis_result"] is None

    state["analysis_result"] = "upload result"
    apply_input_change(state, "paste text again", "paste")

    assert state["current_input_text"] == "paste text again"
    assert state["input_source"] == "paste"
    assert state["analysis_result"] is None
