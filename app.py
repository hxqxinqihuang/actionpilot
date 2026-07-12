from __future__ import annotations

import json
import logging
from datetime import date
from json import JSONDecodeError

from pydantic import ValidationError
import streamlit as st

from src.agent.extractor import ExtractionError
from src.agent.orchestrator import ExtractionOrchestrator
from src.config import AppConfig, ConfigError
from src.checklist import calculate_progress, completed_material_names, initialize_material_checklist
from src.parsers import FileParseError, ParseResult, parse_uploaded_file
from src.parsers.text_parser import parse_text
from src.planner.service import (
    PlannerInput,
    PlannerService,
    deadline_candidates,
    default_deadline_candidate,
    parse_iso_date,
)
from src.planner.validator import PlanValidationError, validate_plan_request
from src.providers.factory import create_llm_provider
from src.providers.openai_compatible import LLMProviderError, LLMProviderTimeoutError
from src.schemas.action_plan import ActionPlan
from src.schemas.task import (
    ConfirmationQuestion,
    Deadline,
    ExtractedTask,
    Material,
    Requirement,
    Risk,
    TaskExtractionResult,
)
from src.ui_state import (
    apply_input_change,
    build_file_signature,
    can_call_provider,
    check_input_length,
    format_provider_error,
    format_plan_error,
    format_parse_error,
    has_analyzable_text,
    input_sha12,
)


logger = logging.getLogger(__name__)

st.set_page_config(page_title="ActionPilot", page_icon="AP", layout="wide")

st.title("ActionPilot")
st.caption("Paste text or upload a notice file, then manually start structured analysis.")


def build_orchestrator() -> ExtractionOrchestrator:
    config = AppConfig.from_env()
    provider = create_llm_provider(config)
    return ExtractionOrchestrator(provider=provider)


def initialize_state() -> None:
    defaults: dict[str, object] = {
        "current_input_text": "",
        "input_source": None,
        "paste_source_text": "",
        "file_parse_result": None,
        "file_signature": None,
        "analysis_result": None,
        "analysis_error": None,
        "action_plan_results": {},
        "action_plan_errors": {},
        "is_generating_plan": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_analysis() -> None:
    st.session_state.analysis_result = None
    st.session_state.analysis_error = None


def set_current_input(text: str, source: str) -> None:
    apply_input_change(st.session_state, text, source)


def log_analysis_exception(exc: Exception, text: str, source: str) -> None:
    logger.exception(
        "Analysis failed: type=%s message=%s input_chars=%s input_sha12=%s input_source=%s",
        type(exc).__name__,
        str(exc),
        len(text),
        input_sha12(text),
        source,
    )


def log_plan_exception(exc: Exception, text: str, source: str) -> None:
    logger.exception(
        "Plan generation failed: type=%s message=%s input_chars=%s input_sha12=%s input_source=%s",
        type(exc).__name__,
        str(exc),
        len(text),
        input_sha12(text),
        source,
    )


def analyze_input(text: str, source: str) -> None:
    set_current_input(text, source)
    if not has_analyzable_text(text):
        st.warning("Please provide non-empty text before starting analysis.")
        return

    length_check = check_input_length(text)
    if not length_check.allowed:
        st.session_state.analysis_error = length_check.error
        st.error(length_check.error)
        return
    if length_check.warning:
        st.warning(length_check.warning)

    try:
        orchestrator = build_orchestrator()
        spinner_text = (
            "Document is long; analyzing with compact mode..."
            if length_check.mode == "compact"
            else "Analyzing with standard mode..."
        )
        with st.spinner(spinner_text):
            st.session_state.analysis_result = orchestrator.extract(text)
        st.session_state.analysis_error = None
        st.success("Analysis complete.")
    except ConfigError as exc:
        st.session_state.analysis_error = str(exc)
        st.error(str(exc))
        st.info("Create a .env file or export environment variables based on .env.example.")
    except LLMProviderTimeoutError as exc:
        st.session_state.analysis_error = str(exc)
        log_analysis_exception(exc, text, source)
        st.error(format_provider_error(exc))
        st.info("You can retry once, or remove unrelated sections before analyzing again.")
    except LLMProviderError as exc:
        st.session_state.analysis_error = str(exc)
        log_analysis_exception(exc, text, source)
        st.error(format_provider_error(exc))
    except ExtractionError as exc:
        st.session_state.analysis_error = str(exc)
        log_analysis_exception(exc, text, source)
        st.error("Analysis failed because too much evidence could not be verified against the source text.")
    except Exception as exc:
        st.session_state.analysis_error = str(exc)
        log_analysis_exception(exc, text, source)
        st.error("Analysis failed. Please check your model settings and try again.")


def render_parse_result(parse_result: ParseResult) -> None:
    st.write(f"File name: {parse_result.file_name}")
    st.write(f"File type: {parse_result.file_type}")
    st.write(f"Character count: {parse_result.char_count}")
    if parse_result.page_count is not None:
        st.write(f"PDF pages: {parse_result.page_count}")
    st.write(f"Warnings: {parse_result.warnings or []}")
    with st.expander("Text preview"):
        st.text(parse_result.text[:3000])


def render_analysis_result() -> None:
    result = st.session_state.analysis_result
    if isinstance(result, TaskExtractionResult):
        st.subheader("Analysis Result")
        if not result.tasks:
            st.info("No actionable tasks were identified.")
            if result.warnings:
                render_warnings(result.warnings)
            with st.expander("View raw JSON"):
                st.json(json.loads(result.model_dump_json()))
            return

        for task_index, task in enumerate(result.tasks):
            st.markdown(f"### Task {task_index + 1}: {task.title}")
            render_task_overview(result, task.summary)
            render_deadlines(task.deadlines)
            render_materials(task.title, task.materials)
            render_requirements(task.requirements)
            render_risks_warnings_questions(task.risks, result.warnings, task.confirmation_questions)
            render_action_plan_section(result, task, task_index)

        with st.expander("View raw JSON"):
            st.json(json.loads(result.model_dump_json()))


def render_task_overview(result: TaskExtractionResult, summary: str) -> None:
    st.markdown("#### Task Overview")
    st.write(f"Summary: {summary}")
    st.write(f"Confidence: {result.confidence:.2f}")
    st.write(f"Source language: {result.source_language or 'Unknown'}")


def render_deadlines(deadlines: list[Deadline]) -> None:
    if not deadlines:
        return
    st.markdown("#### Deadlines")
    for deadline in deadlines:
        status = getattr(deadline, "status", "unknown")
        if status == "ambiguous":
            st.warning("Ambiguous deadline. Please confirm this date.")
        st.write(f"Raw text: {getattr(deadline, 'raw_text', None) or '-'}")
        st.write(f"Normalized date: {getattr(deadline, 'normalized_date', None) or '-'}")
        st.write(f"Status: {status}")
        st.write(f"Evidence status: {getattr(deadline, 'evidence_status', 'unknown')}")
        st.caption(f"Evidence: {getattr(deadline, 'evidence', None) or '-'}")


def render_materials(task_title: str, materials: list[Material]) -> None:
    st.markdown("#### Submission Materials")
    if not materials:
        st.info("No submission materials were identified.")
        return

    input_signature = input_sha12(st.session_state.current_input_text)
    keys = initialize_material_checklist(st.session_state, input_signature, task_title, materials)
    progress = calculate_progress(st.session_state, keys)
    st.write(f"Completed: {progress.completed}/{progress.total} ({progress.percent:.0%})")
    st.progress(progress.percent)

    for material, key in zip(materials, keys, strict=False):
        optional_label = " (optional)" if not getattr(material, "required", True) else ""
        st.checkbox(f"{getattr(material, 'name', 'Material')}{optional_label}", key=key)
        description = getattr(material, "description", None)
        if description:
            st.write(description)
        st.write(f"Evidence status: {getattr(material, 'evidence_status', 'unknown')}")
        st.caption(f"Evidence: {getattr(material, 'evidence', None) or '-'}")


def render_requirements(requirements: list[Requirement]) -> None:
    if not requirements:
        return
    st.markdown("#### Must Requirements")
    for requirement in requirements:
        priority = getattr(requirement, "priority", "unknown")
        st.write(f"[{priority}] {getattr(requirement, 'description', '')}")
        st.write(f"Evidence status: {getattr(requirement, 'evidence_status', 'unknown')}")
        st.caption(f"Evidence: {getattr(requirement, 'evidence', None) or '-'}")


def render_risks_warnings_questions(risks: list[Risk], warnings: list[str], questions: list[ConfirmationQuestion]) -> None:
    if risks or warnings or questions:
        st.markdown("#### Risks, Warnings, and Questions")
    if risks:
        for risk in risks:
            st.write(f"Risk: {getattr(risk, 'description', '')}")
            st.write(f"Severity: {getattr(risk, 'severity', 'unknown')}")
            mitigation = getattr(risk, "mitigation", None)
            if mitigation:
                st.caption(f"Mitigation: {mitigation}")
    if warnings:
        render_warnings(warnings)
    if questions:
        for question in questions:
            st.write(f"Question: {getattr(question, 'question', '')}")
            reason = getattr(question, "reason", None)
            if reason:
                st.caption(f"Reason: {reason}")


def render_warnings(warnings: list[str]) -> None:
    for warning in warnings:
        st.warning(warning)


def render_action_plan_section(result: TaskExtractionResult, task: ExtractedTask, task_index: int) -> None:
    st.markdown("#### Generate Action Plan")
    input_signature = input_sha12(st.session_state.current_input_text)
    plan_key = f"{input_signature}:{task_index}:{task.title}"
    today = date.today()
    candidates = deadline_candidates(task.deadlines, today=today)
    default_deadline = default_deadline_candidate(task.deadlines, today=today)

    if candidates:
        if len(candidates) == 1 and default_deadline is not None:
            st.caption("One clear future deadline was found and prefilled.")
            target_default = parse_iso_date(default_deadline.normalized_date or "") or today
        else:
            labels = ["Manual date"] + [_format_deadline_option(candidate) for candidate in candidates]
            selected_label = st.selectbox("Choose a target deadline", labels, key=f"plan_deadline_select_{plan_key}")
            selected_index = labels.index(selected_label)
            selected_deadline = candidates[selected_index - 1] if selected_index > 0 else None
            target_default = parse_iso_date(selected_deadline.normalized_date or "") if selected_deadline else None
            if target_default is None:
                target_default = today
    else:
        st.caption("No clear future deadline was found. Please enter the target date manually.")
        target_default = today

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Plan start date", value=today, key=f"plan_start_{plan_key}")
        target_date = st.date_input("Target deadline", value=target_default, key=f"plan_target_{plan_key}")
        available_hours = st.number_input(
            "Daily available hours",
            min_value=0.5,
            max_value=16.0,
            value=2.0,
            step=0.5,
            key=f"plan_hours_{plan_key}",
        )
    with col2:
        available_days = st.number_input(
            "Available days per week",
            min_value=1,
            max_value=7,
            value=7,
            step=1,
            key=f"plan_days_{plan_key}",
        )
        buffer_days = st.radio(
            "Final check buffer",
            options=[1, 2],
            horizontal=True,
            key=f"plan_buffer_{plan_key}",
        )
        current_progress = st.text_area(
            "Current progress (optional)",
            height=80,
            key=f"plan_progress_{plan_key}",
        )

    request_check = validate_plan_request(
        start_date=start_date,
        target_date=target_date,
        available_hours_per_day=float(available_hours),
        available_days_per_week=int(available_days),
        buffer_days=int(buffer_days),
    )
    for warning in request_check.warnings:
        st.warning(warning)
    for error in request_check.errors:
        st.error(error)

    if st.button(
        "Generate action plan",
        type="primary",
        disabled=not request_check.allowed or bool(st.session_state.is_generating_plan),
        key=f"generate_plan_{plan_key}",
    ):
        generate_action_plan(
            result=result,
            task=task,
            plan_key=plan_key,
            start_date=start_date,
            target_date=target_date,
            available_hours_per_day=float(available_hours),
            available_days_per_week=int(available_days),
            current_progress=current_progress,
            buffer_days=int(buffer_days),
        )

    plan = st.session_state.action_plan_results.get(plan_key)
    error = st.session_state.action_plan_errors.get(plan_key)
    if error:
        st.error(error)
    if isinstance(plan, ActionPlan):
        render_action_plan(plan)


def generate_action_plan(
    *,
    result: TaskExtractionResult,
    task: ExtractedTask,
    plan_key: str,
    start_date: date,
    target_date: date,
    available_hours_per_day: float,
    available_days_per_week: int,
    current_progress: str,
    buffer_days: int,
) -> None:
    st.session_state.is_generating_plan = True
    try:
        provider = create_llm_provider(AppConfig.from_env())
        service = PlannerService(provider=provider)
        input_signature = input_sha12(st.session_state.current_input_text)
        completed = completed_material_names(st.session_state, input_signature, task.title, task.materials)
        verified_deadlines = [
            deadline
            for deadline in task.deadlines
            if deadline.status == "found" and deadline.evidence_status == "verified"
        ]
        planner_input = PlannerInput(
            task=task,
            source_language=result.source_language,
            verified_deadlines=verified_deadlines,
            completed_materials=completed,
            start_date=start_date,
            target_date=target_date,
            available_hours_per_day=available_hours_per_day,
            available_days_per_week=available_days_per_week,
            current_progress=current_progress,
            buffer_days=buffer_days,
        )
        with st.spinner("Generating action plan..."):
            plan = service.generate_plan(planner_input)
        st.session_state.action_plan_results[plan_key] = plan
        st.session_state.action_plan_errors.pop(plan_key, None)
        st.success("Action plan generated.")
    except ConfigError as exc:
        st.session_state.action_plan_errors[plan_key] = str(exc)
        st.error(str(exc))
    except (LLMProviderError, JSONDecodeError, ValidationError, PlanValidationError) as exc:
        st.session_state.action_plan_errors[plan_key] = format_plan_error(exc)
        log_plan_exception(exc, st.session_state.current_input_text, st.session_state.input_source or "unknown")
    except Exception as exc:
        st.session_state.action_plan_errors[plan_key] = "Action plan generation failed. Please retry."
        log_plan_exception(exc, st.session_state.current_input_text, st.session_state.input_source or "unknown")
    finally:
        st.session_state.is_generating_plan = False


def render_action_plan(plan: ActionPlan) -> None:
    st.markdown("#### Action Plan")
    st.write(f"Goal: {plan.goal}")
    st.write(f"Start date: {plan.start_date.isoformat()}")
    st.write(f"Target date: {plan.target_date.isoformat()}")
    st.write(f"Total days: {plan.total_days}")
    st.write(f"Daily time: {plan.available_hours_per_day:g} hours")

    if plan.assumptions or plan.warnings:
        st.markdown("##### Assumptions and Warnings")
        for assumption in plan.assumptions:
            st.info(assumption)
        for warning in plan.warnings:
            st.warning(warning)

    if plan.phases:
        st.markdown("##### Phases")
        for phase in plan.phases:
            st.write(f"{phase.start_date.isoformat()} - {phase.end_date.isoformat()}: {phase.name}")
            st.caption(phase.objective)

    if plan.daily_tasks:
        st.markdown("##### Daily Timeline")
        for daily_task in plan.daily_tasks:
            st.write(
                f"{daily_task.date.isoformat()} [{daily_task.priority}] "
                f"{daily_task.title} ({daily_task.estimated_hours:g}h)"
            )
            st.caption(daily_task.description)
            if daily_task.deliverable:
                st.write(f"Deliverable: {daily_task.deliverable}")
            if daily_task.related_materials:
                st.write(f"Related materials: {', '.join(daily_task.related_materials)}")

    if plan.milestones:
        st.markdown("##### Milestones")
        for milestone in plan.milestones:
            st.write(f"{milestone.date.isoformat()}: {milestone.name}")
            st.caption(milestone.success_criteria)

    if plan.final_checklist:
        st.markdown("##### Final Checklist")
        for item in plan.final_checklist:
            st.checkbox(item, value=False, key=f"final_check_{input_sha12(plan.title + item)}")

    with st.expander("View raw plan JSON"):
        st.json(json.loads(plan.model_dump_json()))


def _format_deadline_option(deadline: Deadline) -> str:
    date_text = deadline.normalized_date or "-"
    raw_text = deadline.raw_text or deadline.evidence or ""
    return f"{date_text} - {raw_text[:80]}"


initialize_state()

paste_tab, upload_tab = st.tabs(["Paste text", "Upload file"])

with paste_tab:
    notice_text = st.text_area(
        "Notification text",
        height=280,
        placeholder="Paste course notices, competition announcements, job descriptions, or other task-related text here.",
        key="paste_text_area",
    )
    cleaned_text = parse_text(notice_text)
    if st.session_state.paste_source_text != cleaned_text:
        st.session_state.paste_source_text = cleaned_text
        set_current_input(cleaned_text, "paste")

    if st.button("Start analysis", type="primary", disabled=not can_call_provider(cleaned_text), key="analyze_paste"):
        analyze_input(cleaned_text, "paste")

with upload_tab:
    uploaded_file = st.file_uploader("Upload a notice file", type=["txt", "md", "docx", "pdf"])
    if uploaded_file is None:
        if st.session_state.input_source == "upload":
            st.session_state.file_signature = None
            st.session_state.file_parse_result = None
            set_current_input("", "upload")
    else:
        file_bytes = uploaded_file.getvalue()
        signature = build_file_signature(uploaded_file.name, file_bytes)
        if st.session_state.file_signature != signature:
            st.session_state.file_signature = signature
            st.session_state.file_parse_result = None
            clear_analysis()
            try:
                parse_result = parse_uploaded_file(uploaded_file.name, file_bytes)
                st.session_state.file_parse_result = parse_result
                set_current_input(parse_result.text, "upload")
            except FileParseError as exc:
                st.error(format_parse_error(exc))
                set_current_input("", "upload")

        parse_result = st.session_state.file_parse_result
        if isinstance(parse_result, ParseResult):
            render_parse_result(parse_result)
            if st.button(
                "Start analysis",
                type="primary",
                disabled=not can_call_provider(parse_result.text),
                key="analyze_file",
            ):
                analyze_input(parse_result.text, "upload")

render_analysis_result()
