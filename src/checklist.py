from __future__ import annotations

import hashlib
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from src.schemas.task import Material


@dataclass(frozen=True)
class ChecklistProgress:
    completed: int
    total: int
    percent: float


def build_material_key(input_signature: str, task_title: str, material: Material) -> str:
    raw = "|".join(
        [
            input_signature,
            task_title,
            material.name,
            material.description or "",
            str(material.required),
        ]
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"material_done_{digest}"


def initialize_material_checklist(
    state: MutableMapping[str, Any],
    input_signature: str,
    task_title: str,
    materials: list[Material],
) -> list[str]:
    keys = [build_material_key(input_signature, task_title, material) for material in materials]
    known_key = f"material_keys_{input_signature}"
    if state.get(known_key) != keys:
        state[known_key] = keys
        for key in keys:
            state.setdefault(key, False)
    return keys


def calculate_progress(state: MutableMapping[str, Any], keys: list[str]) -> ChecklistProgress:
    total = len(keys)
    completed = sum(1 for key in keys if bool(state.get(key, False)))
    percent = completed / total if total else 0.0
    return ChecklistProgress(completed=completed, total=total, percent=percent)


def set_checklist_item(state: MutableMapping[str, Any], key: str, checked: bool) -> None:
    state[key] = checked


def completed_material_names(
    state: MutableMapping[str, Any],
    input_signature: str,
    task_title: str,
    materials: list[Material],
) -> list[str]:
    keys = initialize_material_checklist(state, input_signature, task_title, materials)
    return [material.name for material, key in zip(materials, keys, strict=False) if bool(state.get(key, False))]
