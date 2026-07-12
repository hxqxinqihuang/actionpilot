from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from pydantic import ValidationError

from src.planner.prompts import PLANNER_SYSTEM_PROMPT, build_plan_user_prompt, build_repair_user_prompt
from src.planner.validator import PlanValidationError, validate_generated_plan
from src.providers.base import LLMProvider
from src.schemas.action_plan import ActionPlan
from src.schemas.task import Deadline, ExtractedTask, Material, Requirement


@dataclass(frozen=True)
class PlannerInput:
    task: ExtractedTask
    source_language: str | None
    verified_deadlines: list[Deadline]
    completed_materials: list[str]
    start_date: date
    target_date: date
    available_hours_per_day: float
    available_days_per_week: int
    current_progress: str | None
    buffer_days: int


class PlannerService:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def generate_plan(self, planner_input: PlannerInput) -> ActionPlan:
        payload = build_planner_payload(planner_input)
        raw = self._provider.generate_json(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=build_plan_user_prompt(payload),
        )
        try:
            return self._parse_and_validate(raw, planner_input)
        except (json.JSONDecodeError, ValidationError, PlanValidationError) as exc:
            repaired = self._provider.generate_json(
                system_prompt=PLANNER_SYSTEM_PROMPT,
                user_prompt=build_repair_user_prompt(raw, str(exc), payload),
            )
            return self._parse_and_validate(repaired, planner_input)

    def _parse_and_validate(self, raw_json: str, planner_input: PlannerInput) -> ActionPlan:
        payload = json.loads(extract_json_object(raw_json))
        payload = normalize_planner_output_payload(payload, planner_input)
        plan = ActionPlan.model_validate(payload)
        plan = ensure_executable_action_plan(plan, planner_input)
        plan = polish_action_plan(plan, planner_input)
        validate_generated_plan(
            plan,
            start_date=planner_input.start_date,
            target_date=planner_input.target_date,
            available_hours_per_day=planner_input.available_hours_per_day,
            completed_materials=planner_input.completed_materials,
        )
        return plan


def normalize_planner_output_payload(payload: dict[str, Any], planner_input: PlannerInput) -> dict[str, Any]:
    if "tasks" not in payload or "daily_tasks" in payload:
        return payload

    tasks_value = payload.get("tasks")
    if not isinstance(tasks_value, list):
        return payload

    daily_tasks: list[dict[str, Any]] = []
    for item in tasks_value:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("name") or "Task").strip()
        if not title:
            continue
        daily_tasks.append(
            {
                "date": planner_input.start_date.isoformat(),
                "title": title,
                "description": str(item.get("description") or item.get("summary") or title),
                "estimated_hours": min(planner_input.available_hours_per_day, 1.0),
                "priority": _normalize_priority(item.get("priority")),
                "deliverable": str(item.get("deliverable") or title),
                "related_materials": _string_list(item.get("related_materials")),
            }
        )

    warnings = _string_list(payload.get("warnings"))
    warnings.append("已根据文档要求生成阶段化执行计划。")
    return {
        "title": str(payload.get("title") or planner_input.task.title),
        "goal": str(payload.get("goal") or planner_input.task.summary or planner_input.task.title),
        "start_date": planner_input.start_date.isoformat(),
        "target_date": planner_input.target_date.isoformat(),
        "total_days": (planner_input.target_date - planner_input.start_date).days + 1,
        "available_hours_per_day": planner_input.available_hours_per_day,
        "assumptions": _string_list(payload.get("assumptions")),
        "phases": [],
        "daily_tasks": daily_tasks,
        "milestones": [],
        "final_checklist": _string_list(payload.get("final_checklist")),
        "warnings": warnings,
    }


def build_planner_payload(planner_input: PlannerInput) -> dict[str, Any]:
    task = planner_input.task
    return {
        "task": {
            "title": task.title,
            "summary": task.summary,
            "source_language": planner_input.source_language,
            "verified_deadlines": [_deadline_payload(deadline) for deadline in planner_input.verified_deadlines],
            "materials": [_material_payload(material) for material in task.materials],
            "requirements": [_requirement_payload(requirement) for requirement in task.requirements],
            "confirmation_questions": [
                {"question": question.question, "reason": question.reason}
                for question in task.confirmation_questions
            ],
        },
        "completed_materials": planner_input.completed_materials,
        "execution_blueprint": build_execution_blueprint(task.materials, task.requirements, planner_input.source_language),
        "constraints": {
            "start_date": planner_input.start_date.isoformat(),
            "target_date": planner_input.target_date.isoformat(),
            "total_days": (planner_input.target_date - planner_input.start_date).days + 1,
            "available_hours_per_day": planner_input.available_hours_per_day,
            "available_days_per_week": planner_input.available_days_per_week,
            "current_progress": planner_input.current_progress or "",
            "final_check_buffer_days": planner_input.buffer_days,
        },
    }


def ensure_executable_action_plan(plan: ActionPlan, planner_input: PlannerInput) -> ActionPlan:
    if not _is_shallow_action_plan(plan, planner_input):
        return plan
    return build_template_action_plan(planner_input, source_plan=plan)


def build_template_action_plan(planner_input: PlannerInput, *, source_plan: ActionPlan | None = None) -> ActionPlan:
    task = planner_input.task
    total_days = (planner_input.target_date - planner_input.start_date).days + 1
    blueprint = build_execution_blueprint(task.materials, task.requirements, planner_input.source_language)
    phase_specs = blueprint["recommended_phases"]
    phase_dates = _spread_dates(planner_input.start_date, planner_input.target_date, len(phase_specs))
    phase_payloads: list[dict[str, Any]] = []
    daily_task_payloads: list[dict[str, Any]] = []
    milestone_payloads: list[dict[str, Any]] = []
    available_hours = max(0.5, min(planner_input.available_hours_per_day, planner_input.available_hours_per_day))

    for index, phase in enumerate(phase_specs):
        phase_date = phase_dates[index]
        next_date = phase_dates[index + 1] if index + 1 < len(phase_dates) else planner_input.target_date
        phase_payloads.append(
            {
                "name": phase["name"],
                "start_date": phase_date,
                "end_date": max(phase_date, next_date),
                "objective": phase["objective"],
                "key_actions": phase["key_actions"],
                "deliverable": phase["deliverable"],
            }
        )
        daily_task_payloads.append(
            {
                "date": phase_date,
                "title": phase["task_title"],
                "description": phase["task_description"],
                "estimated_hours": min(available_hours, max(0.5, available_hours)),
                "priority": "high" if index < 2 else "medium",
                "deliverable": phase["deliverable"],
                "related_materials": phase["related_materials"],
            }
        )
        milestone_payloads.append(
            {
                "date": phase_date,
                "name": phase["milestone"],
                "success_criteria": phase["success_criteria"],
            }
        )

    warnings = list(source_plan.warnings if source_plan is not None else [])
    warnings.append("文档信息较复杂，已提取核心目标并生成行动方案。")
    plan_type = _plan_type(total_days)
    first_phase = phase_specs[0] if phase_specs else None
    return ActionPlan.model_validate(
        {
            "title": (source_plan.title if source_plan is not None else None) or task.title,
            "goal": _action_goal(task),
            "start_date": planner_input.start_date,
            "target_date": planner_input.target_date,
            "total_days": total_days,
            "available_hours_per_day": planner_input.available_hours_per_day,
            "plan_type": plan_type,
            "current_focus": first_phase["name"] if first_phase and plan_type == "phase" else None,
            "next_actions": first_phase["key_actions"] if first_phase and plan_type == "phase" else [],
            "assumptions": list(source_plan.assumptions if source_plan is not None else []),
            "phases": phase_payloads,
            "daily_tasks": daily_task_payloads,
            "milestones": milestone_payloads,
            "final_checklist": _final_checklist(task.materials, planner_input.completed_materials),
            "warnings": warnings,
        }
    )


def polish_action_plan(plan: ActionPlan, planner_input: PlannerInput) -> ActionPlan:
    payload = plan.model_dump()
    total_days = (planner_input.target_date - planner_input.start_date).days + 1
    plan_type = _plan_type(total_days)
    payload["plan_type"] = plan_type
    if _goal_copies_title(plan.goal, planner_input.task.title):
        payload["goal"] = _action_goal(planner_input.task)
    payload["warnings"] = _friendly_warnings(plan.warnings)

    if plan_type == "phase":
        phases = payload.get("phases") or []
        if phases:
            first_phase = phases[0]
            payload["current_focus"] = payload.get("current_focus") or first_phase.get("name")
            payload["next_actions"] = payload.get("next_actions") or first_phase.get("key_actions") or _default_next_actions(
                planner_input.source_language
            )
        else:
            payload["current_focus"] = payload.get("current_focus") or "需求理解与方案设计"
            payload["next_actions"] = payload.get("next_actions") or _default_next_actions(planner_input.source_language)
    return ActionPlan.model_validate(payload)


def build_execution_blueprint(
    materials: list[Material],
    requirements: list[Requirement],
    source_language: str | None,
) -> dict[str, Any]:
    material_names = [material.name for material in materials]
    chinese = _prefers_chinese(source_language, material_names + [requirement.description for requirement in requirements])
    phases = _chinese_phase_blueprint(materials) if chinese else _english_phase_blueprint(materials)
    return {
        "instruction": (
            "Use these phases as a scaffold. Convert materials and hard requirements into concrete execution tasks, "
            "not task-title repetition."
        ),
        "recommended_phases": phases,
    }


def _is_shallow_action_plan(plan: ActionPlan, planner_input: PlannerInput) -> bool:
    if not plan.daily_tasks:
        return True
    task_title = planner_input.task.title.strip().lower()
    plan_title = plan.title.strip().lower()
    for daily_task in plan.daily_tasks:
        title = daily_task.title.strip().lower()
        deliverable = daily_task.deliverable.strip().lower()
        description = daily_task.description.strip().lower()
        if title in {task_title, plan_title}:
            return True
        if deliverable in {task_title, plan_title, title}:
            return True
        if title == description:
            return True
    return False


def _chinese_phase_blueprint(materials: list[Material]) -> list[dict[str, Any]]:
    return [
        {
            "name": "需求理解与方案设计",
            "objective": "明确任务目标、应用场景、技术路线和功能模块。",
            "key_actions": ["分析比赛功能要求", "确定目标用户和应用场景", "设计Agent核心流程", "完成初版技术方案"],
            "task_title": "完成项目方案设计",
            "task_description": "梳理任务要求，确定应用场景、技术路线、功能模块和验收口径。",
            "deliverable": "项目方案设计草稿",
            "related_materials": _related_materials(materials, ("PPT", "方案", "报告")),
            "milestone": "方案框架确定",
            "success_criteria": "已形成可用于开发和展示的方案框架。",
        },
        {
            "name": "技术开发与模型调用",
            "objective": "完成核心功能、模型/API调用和基础数据流程。",
            "key_actions": ["实现主要工作流", "完成模型调用", "调试输入输出流程", "处理关键异常场景"],
            "task_title": "开发核心功能并接入模型调用",
            "task_description": "实现主要功能链路，完成模型/API调用、输入输出处理和基础错误处理。",
            "deliverable": "可运行的核心功能代码",
            "related_materials": _related_materials(materials, ("代码", "仓库", "API", "模型")),
            "milestone": "核心功能可运行",
            "success_criteria": "核心流程可以端到端运行并产出结果。",
        },
        {
            "name": "Demo实现",
            "objective": "把核心功能整理为可演示的应用流程。",
            "key_actions": ["完善交互流程", "准备演示样例", "录制或整理Demo说明", "检查演示稳定性"],
            "task_title": "实现可演示Demo流程",
            "task_description": "完善交互流程、示例输入和展示结果，确保演示过程稳定。",
            "deliverable": "可演示Demo",
            "related_materials": _related_materials(materials, ("Demo", "视频", "演示")),
            "milestone": "Demo可演示",
            "success_criteria": "Demo可以展示核心价值和完整使用流程。",
        },
        {
            "name": "测试与用户反馈",
            "objective": "验证效果、记录问题并完成必要修改。",
            "key_actions": ["设计测试样例", "记录效果对比", "收集用户反馈", "整理验证结论"],
            "task_title": "完成测试验证与反馈整理",
            "task_description": "使用典型样例测试功能效果，记录问题、反馈和改进结论。",
            "deliverable": "测试与反馈记录",
            "related_materials": _related_materials(materials, ("报告", "验证", "测试")),
            "milestone": "测试结论完成",
            "success_criteria": "已确认主要功能稳定，并整理可写入材料的验证结论。",
        },
        {
            "name": "材料整理与提交",
            "objective": "完成提交材料检查、格式整理和最终提交。",
            "key_actions": ["检查报名表和声明文件", "整理PPT与技术报告", "打包代码和Demo材料", "完成最终提交确认"],
            "task_title": "整理并提交最终材料",
            "task_description": "检查报名表、PPT、Demo、代码和报告等材料，按要求命名、打包并提交。",
            "deliverable": "最终提交材料包",
            "related_materials": [material.name for material in materials],
            "milestone": "材料提交完成",
            "success_criteria": "所有必交材料已检查、打包并按截止要求提交。",
        },
    ]


def _english_phase_blueprint(materials: list[Material]) -> list[dict[str, Any]]:
    return [
        {
            "name": "Requirement Understanding and Solution Design",
            "objective": "Clarify the goal, use case, technical route, and modules.",
            "key_actions": ["Analyze functional requirements", "Define target users and use case", "Design the agent workflow", "Draft the technical solution"],
            "task_title": "Complete solution design",
            "task_description": "Review requirements and define the use case, technical route, functional modules, and acceptance criteria.",
            "deliverable": "Solution design draft",
            "related_materials": _related_materials(materials, ("PPT", "proposal", "report")),
            "milestone": "Solution framework ready",
            "success_criteria": "A concrete design framework is ready for implementation and presentation.",
        },
        {
            "name": "Technical Development and Model Integration",
            "objective": "Build core functions, model/API integration, and data flow.",
            "key_actions": ["Implement the main workflow", "Integrate model calls", "Debug input and output flow", "Handle key error cases"],
            "task_title": "Implement core functions and model calls",
            "task_description": "Build the main workflow, integrate model/API calls, and handle inputs, outputs, and basic errors.",
            "deliverable": "Runnable core code",
            "related_materials": _related_materials(materials, ("code", "repo", "API", "model")),
            "milestone": "Core workflow running",
            "success_criteria": "The core flow runs end to end and produces usable output.",
        },
        {
            "name": "Demo Implementation",
            "objective": "Package the core function into a stable demo flow.",
            "key_actions": ["Polish the interaction flow", "Prepare demo examples", "Create demo notes or recording", "Check demo stability"],
            "task_title": "Implement demo workflow",
            "task_description": "Prepare sample inputs, user flow, and output presentation so the demo is stable.",
            "deliverable": "Working demo",
            "related_materials": _related_materials(materials, ("demo", "video", "presentation")),
            "milestone": "Demo ready",
            "success_criteria": "The demo shows the core value and complete usage flow.",
        },
        {
            "name": "Testing and Feedback",
            "objective": "Validate results, record issues, and make needed improvements.",
            "key_actions": ["Design test cases", "Record result comparisons", "Collect user feedback", "Summarize validation findings"],
            "task_title": "Run tests and organize feedback",
            "task_description": "Test with typical cases, record issues and feedback, and summarize improvement conclusions.",
            "deliverable": "Test and feedback notes",
            "related_materials": _related_materials(materials, ("report", "validation", "test")),
            "milestone": "Validation complete",
            "success_criteria": "Main functions are stable and validation conclusions are ready for materials.",
        },
        {
            "name": "Material Packaging and Submission",
            "objective": "Check, format, package, and submit final materials.",
            "key_actions": ["Check forms and statements", "Prepare slides and technical report", "Package code and demo materials", "Confirm final submission"],
            "task_title": "Prepare and submit final materials",
            "task_description": "Check forms, slides, demo, code, reports, names, and packaging before final submission.",
            "deliverable": "Final submission package",
            "related_materials": [material.name for material in materials],
            "milestone": "Submission complete",
            "success_criteria": "All required materials are checked, packaged, and submitted before the deadline.",
        },
    ]


def _spread_dates(start_date: date, target_date: date, count: int) -> list[date]:
    if count <= 1:
        return [start_date]
    total_span = max((target_date - start_date).days, 0)
    return [start_date + timedelta(days=round(total_span * index / (count - 1))) for index in range(count)]


def _related_materials(materials: list[Material], keywords: tuple[str, ...]) -> list[str]:
    matched = [
        material.name
        for material in materials
        if any(keyword.lower() in material.name.lower() for keyword in keywords)
    ]
    return matched


def _final_checklist(materials: list[Material], completed_materials: list[str]) -> list[str]:
    checklist = []
    completed = {material.lower() for material in completed_materials}
    for material in materials:
        if material.name.lower() in completed:
            checklist.append(f"Review completed material: {material.name}")
        else:
            checklist.append(f"Prepare and verify: {material.name}")
    return checklist


def _prefers_chinese(source_language: str | None, texts: list[str]) -> bool:
    if source_language and ("中" in source_language or "chinese" in source_language.lower()):
        return True
    return any(any("\u4e00" <= char <= "\u9fff" for char in text) for text in texts)


def _plan_type(total_days: int) -> str:
    if total_days <= 7:
        return "daily"
    if total_days <= 30:
        return "weekly"
    return "phase"


def _goal_copies_title(goal: str, title: str) -> bool:
    normalized_goal = re.sub(r"\s+", "", goal).lower()
    normalized_title = re.sub(r"\s+", "", title).lower()
    return not normalized_goal or normalized_goal == normalized_title or normalized_title in normalized_goal


def _action_goal(task: ExtractedTask) -> str:
    names = [material.name for material in task.materials]
    chinese = _prefers_chinese(None, [task.title, task.summary, *names])
    if chinese:
        if names:
            core_materials = "、".join(_short_material_name(name) for name in names[:3])
            return f"完成作品开发，并完成{core_materials}等任务材料提交。"
        return "完成任务要求的核心成果，并按截止时间提交所需材料。"
    if names:
        core_materials = ", ".join(_short_material_name(name) for name in names[:3])
        return f"Complete the project deliverable and submit required materials including {core_materials}."
    return "Complete the required work and submit all required materials before the deadline."


def _short_material_name(name: str) -> str:
    return re.split(r"[（(]", name, maxsplit=1)[0].strip() or name


def _friendly_warnings(warnings: list[str]) -> list[str]:
    internal_markers = (
        "Planner returned extractor-style tasks",
        "Planner result was too generic",
        "converted them to daily_tasks",
        "executable scaffold",
    )
    friendly: list[str] = []
    for warning in warnings:
        if any(marker in warning for marker in internal_markers):
            continue
        if warning not in friendly:
            friendly.append(warning)
    if not friendly:
        friendly.append("已根据文档要求生成阶段化执行计划。")
    return friendly


def _default_next_actions(source_language: str | None) -> list[str]:
    if source_language and ("中" in source_language or "chinese" in source_language.lower()):
        return ["分析任务要求", "确定目标用户和应用场景", "设计核心流程", "完成初版方案"]
    return ["Analyze task requirements", "Define target users and use case", "Design the core workflow", "Draft the initial solution"]


def extract_json_object(raw: str) -> str:
    stripped = raw.strip()
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        stripped = fence_match.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise json.JSONDecodeError("No JSON object found", raw, 0)
    return stripped[start : end + 1]


def deadline_candidates(deadlines: list[Deadline], *, today: date) -> list[Deadline]:
    candidates: list[Deadline] = []
    for deadline in deadlines:
        if deadline.status != "found" or not deadline.normalized_date:
            continue
        parsed = parse_iso_date(deadline.normalized_date)
        if parsed is not None and parsed >= today:
            candidates.append(deadline)
    return candidates


def default_deadline_candidate(deadlines: list[Deadline], *, today: date) -> Deadline | None:
    candidates = deadline_candidates(deadlines, today=today)
    return candidates[0] if len(candidates) == 1 else None


def parse_iso_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _deadline_payload(deadline: Deadline) -> dict[str, Any]:
    return {
        "raw_text": deadline.raw_text,
        "normalized_date": deadline.normalized_date,
        "status": deadline.status,
        "evidence_status": deadline.evidence_status,
        "evidence": deadline.evidence,
    }


def _material_payload(material: Material) -> dict[str, Any]:
    return {
        "name": material.name,
        "description": material.description,
        "required": material.required,
        "evidence_status": material.evidence_status,
        "evidence": material.evidence,
    }


def _requirement_payload(requirement: Requirement) -> dict[str, Any]:
    return {
        "description": requirement.description,
        "priority": requirement.priority,
        "evidence_status": requirement.evidence_status,
        "evidence": requirement.evidence,
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None and str(item).strip()]


def _normalize_priority(value: Any) -> str:
    priority = str(value or "medium").lower()
    return priority if priority in {"high", "medium", "low"} else "medium"
