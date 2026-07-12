from __future__ import annotations

import json
from datetime import date

import pytest

from src.planner.prompts import PLANNER_SYSTEM_PROMPT
from src.planner.service import (
    PlannerInput,
    PlannerService,
    build_planner_payload,
    deadline_candidates,
    default_deadline_candidate,
    normalize_planner_output_payload,
)
from src.planner.validator import PlanValidationError, validate_plan_request
from src.providers.base import LLMProvider
from src.schemas.action_plan import ActionPlan
from src.schemas.task import ConfirmationQuestion, Deadline, ExtractedTask, Material, Requirement


class MockProvider(LLMProvider):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, str]] = []

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        if not self.responses:
            raise AssertionError("Provider was called more times than expected.")
        return self.responses.pop(0)


def test_single_clear_future_deadline_can_prefill() -> None:
    deadline = Deadline(raw_text="7月20日前提交", normalized_date="2026-07-20", status="found")

    assert default_deadline_candidate([deadline], today=date(2026, 7, 12)) == deadline


def test_multiple_deadlines_need_user_selection() -> None:
    deadlines = [
        Deadline(raw_text="7月20日前报名", normalized_date="2026-07-20", status="found"),
        Deadline(raw_text="7月30日前提交", normalized_date="2026-07-30", status="found"),
    ]

    assert len(deadline_candidates(deadlines, today=date(2026, 7, 12))) == 2
    assert default_deadline_candidate(deadlines, today=date(2026, 7, 12)) is None


def test_ambiguous_missing_and_past_deadlines_are_not_auto_candidates() -> None:
    deadlines = [
        Deadline(raw_text="日期待确认", normalized_date=None, status="ambiguous"),
        Deadline(raw_text=None, normalized_date=None, status="missing"),
        Deadline(raw_text="昨天截止", normalized_date="2026-07-11", status="found"),
    ]

    assert deadline_candidates(deadlines, today=date(2026, 7, 12)) == []
    assert default_deadline_candidate(deadlines, today=date(2026, 7, 12)) is None


def test_invalid_target_date_is_rejected() -> None:
    result = validate_plan_request(
        start_date=date(2026, 7, 20),
        target_date=date(2026, 7, 19),
        available_hours_per_day=2,
        available_days_per_week=7,
        buffer_days=1,
    )

    assert result.allowed is False
    assert "Target date" in result.errors[0]


def test_invalid_daily_hours_are_rejected() -> None:
    result = validate_plan_request(
        start_date=date(2026, 7, 12),
        target_date=date(2026, 7, 20),
        available_hours_per_day=17,
        available_days_per_week=7,
        buffer_days=1,
    )

    assert result.allowed is False
    assert "Daily available hours" in result.errors[0]


def test_buffer_days_must_fit_available_days() -> None:
    result = validate_plan_request(
        start_date=date(2026, 7, 12),
        target_date=date(2026, 7, 12),
        available_hours_per_day=2,
        available_days_per_week=7,
        buffer_days=1,
    )

    assert result.allowed is False
    assert "buffer" in result.errors[0]


def test_planner_payload_uses_required_materials_and_buffer() -> None:
    planner_input = make_planner_input()

    payload = build_planner_payload(planner_input)

    assert payload["task"]["materials"][0]["required"] is True
    assert payload["task"]["materials"][1]["required"] is False
    assert payload["constraints"]["final_check_buffer_days"] == 1
    assert "execution_blueprint" in payload
    assert payload["execution_blueprint"]["recommended_phases"]


def test_planner_payload_does_not_pass_emergency_extraction_fields() -> None:
    planner_input = make_planner_input()

    payload = build_planner_payload(planner_input)

    assert "tasks" not in payload
    assert "must_do" not in payload
    assert "warnings" not in payload["task"]


def test_generate_plan_calls_provider_once_on_valid_response() -> None:
    provider = MockProvider([valid_plan_json()])
    service = PlannerService(provider)

    plan = service.generate_plan(make_planner_input())

    assert isinstance(plan, ActionPlan)
    assert len(provider.calls) == 1
    assert provider.calls[0]["system_prompt"] == PLANNER_SYSTEM_PROMPT


def test_emergency_extractor_result_can_generate_plan() -> None:
    task = ExtractedTask(
        title="Competition",
        summary="Competition",
        deadlines=[Deadline(raw_text=None, normalized_date=None, status="missing")],
        materials=[Material(name="作品", required=True)],
        requirements=[Requirement(description="完成报名", priority="must")],
        confirmation_questions=[ConfirmationQuestion(question="确认提交日期")],
    )
    planner_input = make_planner_input()
    planner_input = PlannerInput(
        task=task,
        source_language=None,
        verified_deadlines=[],
        completed_materials=[],
        start_date=planner_input.start_date,
        target_date=planner_input.target_date,
        available_hours_per_day=planner_input.available_hours_per_day,
        available_days_per_week=planner_input.available_days_per_week,
        current_progress=None,
        buffer_days=1,
    )
    provider = MockProvider([valid_plan_json()])

    plan = PlannerService(provider).generate_plan(planner_input)

    assert plan.title == "ActionPilot project"
    assert len(provider.calls) == 1


def test_click_generate_equivalent_service_call_happens_once() -> None:
    provider = MockProvider([valid_plan_json()])

    PlannerService(provider).generate_plan(make_planner_input())

    assert len(provider.calls) == 1


def test_completed_material_is_not_scheduled_as_main_creation_task() -> None:
    provider = MockProvider([plan_with_completed_material_creation(), valid_plan_json()])
    planner_input = make_planner_input(completed_materials=["README.md"])

    plan = PlannerService(provider).generate_plan(planner_input)

    assert plan.title == "ActionPilot project"
    assert len(provider.calls) == 2


def test_daily_total_time_limit_is_validated_and_repaired() -> None:
    provider = MockProvider([plan_with_too_many_hours(), valid_plan_json()])

    plan = PlannerService(provider).generate_plan(make_planner_input())

    assert plan.daily_tasks[0].estimated_hours <= 2
    assert len(provider.calls) == 2


def test_task_dates_must_stay_in_range() -> None:
    provider = MockProvider([plan_with_out_of_range_task(), valid_plan_json()])

    plan = PlannerService(provider).generate_plan(make_planner_input())

    assert plan.daily_tasks[0].date == date(2026, 7, 12)
    assert len(provider.calls) == 2


def test_invalid_json_and_schema_errors_allow_one_repair() -> None:
    provider = MockProvider(["not json", valid_plan_json()])

    plan = PlannerService(provider).generate_plan(make_planner_input())

    assert plan.title == "ActionPilot project"
    assert len(provider.calls) == 2
    assert "Invalid response to repair" in provider.calls[1]["user_prompt"]


def test_tasks_style_planner_output_is_converted_without_traceback() -> None:
    provider = MockProvider(['{"tasks": [{"title": "test"}]}'])

    plan = PlannerService(provider).generate_plan(make_planner_input())

    assert plan.title == "ActionPilot project"
    assert plan.goal == "Build a structured action app."
    assert plan.start_date == date(2026, 7, 12)
    assert plan.target_date == date(2026, 7, 20)
    assert plan.daily_tasks[0].title != "test"
    assert plan.daily_tasks[0].deliverable
    assert len(provider.calls) == 1


def test_tasks_style_conversion_supplements_required_action_plan_fields() -> None:
    payload = normalize_planner_output_payload({"tasks": [{"title": "test"}]}, make_planner_input())

    assert payload["title"] == "ActionPilot project"
    assert payload["goal"] == "Build a structured action app."
    assert payload["start_date"] == "2026-07-12"
    assert payload["target_date"] == "2026-07-20"
    assert payload["total_days"] == 9
    assert payload["available_hours_per_day"] == 2.0
    assert payload["daily_tasks"][0]["title"] == "test"


def test_second_invalid_response_stops() -> None:
    provider = MockProvider(["not json", '{"title": "missing fields"}'])

    with pytest.raises(Exception):
        PlannerService(provider).generate_plan(make_planner_input())

    assert len(provider.calls) == 2


def test_illegal_planner_json_still_raises_after_repair_attempt() -> None:
    provider = MockProvider(["not json", "still not json"])

    with pytest.raises(json.JSONDecodeError):
        PlannerService(provider).generate_plan(make_planner_input())

    assert len(provider.calls) == 2


def test_fenced_json_plan_can_be_parsed() -> None:
    provider = MockProvider([f"```json\n{valid_plan_json()}\n```"])

    plan = PlannerService(provider).generate_plan(make_planner_input())

    assert plan.title == "ActionPilot project"


def test_shallow_title_copy_plan_is_replaced_with_executable_scaffold() -> None:
    provider = MockProvider([title_copy_plan_json()])
    planner_input = make_competition_planner_input()

    plan = PlannerService(provider).generate_plan(planner_input)

    assert all(task.title != planner_input.task.title for task in plan.daily_tasks)
    assert any("方案" in task.title or "设计" in task.title for task in plan.daily_tasks)
    assert any("Demo" in task.title or "演示" in task.description for task in plan.daily_tasks)
    assert any("提交" in task.title for task in plan.daily_tasks)
    assert all(task.deliverable and task.deliverable != task.title for task in plan.daily_tasks)
    assert plan.phases


def test_planner_prompt_requires_action_tasks_not_title_repetition() -> None:
    assert "do not repeat the task title" in PLANNER_SYSTEM_PROMPT
    assert "Every daily task title must start with a concrete action verb" in PLANNER_SYSTEM_PROMPT
    assert "registration form, PPT, demo, code, and validation report" in PLANNER_SYSTEM_PROMPT


def make_planner_input(completed_materials: list[str] | None = None) -> PlannerInput:
    task = ExtractedTask(
        title="ActionPilot project",
        summary="Build a structured action app.",
        deadlines=[Deadline(raw_text="Submit by July 20", normalized_date="2026-07-20", status="found")],
        materials=[
            Material(name="README.md", required=True),
            Material(name="Demo video", required=False),
        ],
        requirements=[Requirement(description="Use an LLM API", priority="must")],
        confirmation_questions=[ConfirmationQuestion(question="Confirm exact submission channel.")],
    )
    return PlannerInput(
        task=task,
        source_language="English",
        verified_deadlines=task.deadlines,
        completed_materials=completed_materials or [],
        start_date=date(2026, 7, 12),
        target_date=date(2026, 7, 20),
        available_hours_per_day=2.0,
        available_days_per_week=7,
        current_progress=None,
        buffer_days=1,
    )


def make_competition_planner_input() -> PlannerInput:
    task = ExtractedTask(
        title="面向一流学科建设的学科垂类大模型与创新应用开发比赛",
        summary="完成竞赛作品并提交材料。",
        deadlines=[Deadline(raw_text="作品提交截止2026年9月5日", normalized_date="2026-09-05", status="found", type="submission")],
        materials=[
            Material(name="报名表", required=True),
            Material(name="PPT", required=True),
            Material(name="Demo", required=True),
            Material(name="代码", required=True),
            Material(name="效果验证报告", required=True),
        ],
        requirements=[Requirement(description="完成大模型创新应用开发", priority="must")],
        confirmation_questions=[],
    )
    return PlannerInput(
        task=task,
        source_language="中文",
        verified_deadlines=task.deadlines,
        completed_materials=[],
        start_date=date(2026, 8, 25),
        target_date=date(2026, 9, 5),
        available_hours_per_day=3.0,
        available_days_per_week=7,
        current_progress=None,
        buffer_days=1,
    )


def valid_plan_json() -> str:
    return json.dumps(
        {
            "title": "ActionPilot project",
            "goal": "Finish required submission materials.",
            "start_date": "2026-07-12",
            "target_date": "2026-07-20",
            "total_days": 9,
            "available_hours_per_day": 2.0,
            "assumptions": ["Submission channel still needs confirmation."],
            "phases": [
                {
                    "name": "Build",
                    "start_date": "2026-07-12",
                    "end_date": "2026-07-18",
                    "objective": "Create required deliverables.",
                }
            ],
            "daily_tasks": [
                {
                    "date": "2026-07-12",
                    "title": "Draft implementation outline",
                    "description": "Plan the app workflow and required materials.",
                    "estimated_hours": 2.0,
                    "priority": "high",
                    "deliverable": "Implementation outline",
                    "related_materials": ["README.md"],
                }
            ],
            "milestones": [
                {
                    "date": "2026-07-18",
                    "name": "Materials ready",
                    "success_criteria": "Required files are prepared.",
                }
            ],
            "final_checklist": ["Confirm submission channel", "Review README.md"],
            "warnings": ["The plan depends on one unresolved question."],
        },
        ensure_ascii=False,
    )


def plan_with_completed_material_creation() -> str:
    payload = json.loads(valid_plan_json())
    payload["daily_tasks"][0]["title"] = "Create README.md"
    payload["daily_tasks"][0]["description"] = "Write README.md from scratch."
    payload["daily_tasks"][0]["deliverable"] = "README.md"
    return json.dumps(payload)


def plan_with_too_many_hours() -> str:
    payload = json.loads(valid_plan_json())
    payload["daily_tasks"][0]["estimated_hours"] = 3.0
    return json.dumps(payload)


def plan_with_out_of_range_task() -> str:
    payload = json.loads(valid_plan_json())
    payload["daily_tasks"][0]["date"] = "2026-07-21"
    return json.dumps(payload)


def title_copy_plan_json() -> str:
    return json.dumps(
        {
            "title": "面向一流学科建设的学科垂类大模型与创新应用开发比赛",
            "goal": "完成竞赛",
            "start_date": "2026-08-25",
            "target_date": "2026-09-05",
            "total_days": 12,
            "available_hours_per_day": 3.0,
            "assumptions": [],
            "phases": [],
            "daily_tasks": [
                {
                    "date": "2026-08-25",
                    "title": "面向一流学科建设的学科垂类大模型与创新应用开发比赛",
                    "description": "面向一流学科建设的学科垂类大模型与创新应用开发比赛",
                    "estimated_hours": 2.0,
                    "priority": "high",
                    "deliverable": "面向一流学科建设的学科垂类大模型与创新应用开发比赛",
                    "related_materials": [],
                }
            ],
            "milestones": [],
            "final_checklist": [],
            "warnings": [],
        },
        ensure_ascii=False,
    )
