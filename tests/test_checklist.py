from __future__ import annotations

from src.checklist import (
    build_material_key,
    calculate_progress,
    completed_material_names,
    initialize_material_checklist,
    set_checklist_item,
)
from src.schemas.task import Material


def test_empty_material_list() -> None:
    state: dict[str, object] = {}

    keys = initialize_material_checklist(state, "input-a", "Task", [])
    progress = calculate_progress(state, keys)

    assert keys == []
    assert progress.completed == 0
    assert progress.total == 0
    assert progress.percent == 0.0


def test_required_and_optional_materials_get_distinct_stable_keys() -> None:
    required = Material(name="README.md", required=True)
    optional = Material(name="Demo video", required=False)

    required_key = build_material_key("input-a", "Task", required)
    optional_key = build_material_key("input-a", "Task", optional)

    assert required_key == build_material_key("input-a", "Task", required)
    assert optional_key == build_material_key("input-a", "Task", optional)
    assert required_key != optional_key


def test_input_signature_change_isolates_checklists() -> None:
    material = Material(name="README.md")

    first_key = build_material_key("input-a", "Task", material)
    second_key = build_material_key("input-b", "Task", material)

    assert first_key != second_key


def test_progress_zero_partial_and_complete() -> None:
    state: dict[str, object] = {}
    materials = [Material(name="README.md"), Material(name="PDF")]
    keys = initialize_material_checklist(state, "input-a", "Task", materials)

    assert calculate_progress(state, keys).percent == 0.0

    set_checklist_item(state, keys[0], True)
    partial = calculate_progress(state, keys)
    assert partial.completed == 1
    assert partial.total == 2
    assert partial.percent == 0.5

    set_checklist_item(state, keys[1], True)
    complete = calculate_progress(state, keys)
    assert complete.completed == 2
    assert complete.percent == 1.0


def test_rerun_initialization_does_not_reset_existing_state() -> None:
    state: dict[str, object] = {}
    materials = [Material(name="README.md"), Material(name="PDF")]
    keys = initialize_material_checklist(state, "input-a", "Task", materials)
    set_checklist_item(state, keys[0], True)

    rerun_keys = initialize_material_checklist(state, "input-a", "Task", materials)

    assert rerun_keys == keys
    assert state[keys[0]] is True
    assert state[keys[1]] is False


def test_completed_material_names_does_not_call_provider_or_reset_plan() -> None:
    state: dict[str, object] = {"action_plan_results": {"existing": object()}}
    materials = [Material(name="README.md"), Material(name="PDF")]
    keys = initialize_material_checklist(state, "input-a", "Task", materials)
    set_checklist_item(state, keys[1], True)

    completed = completed_material_names(state, "input-a", "Task", materials)

    assert completed == ["PDF"]
    assert "action_plan_results" in state
