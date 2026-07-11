from __future__ import annotations

from src.agent.prompts import SYSTEM_PROMPT, build_extraction_prompt


def test_prompt_requires_matching_output_language() -> None:
    assert "same language" in SYSTEM_PROMPT
    assert "same dominant language" in build_extraction_prompt("测试通知")


def test_prompt_requires_missing_deadline_message() -> None:
    assert 'status "missing"' in SYSTEM_PROMPT
    assert "raw_text null" in SYSTEM_PROMPT
    assert "evidence null" in SYSTEM_PROMPT


def test_prompt_separates_prerequisites_materials_and_requirements() -> None:
    assert "prerequisites: skills, tools, accounts, APIs" in SYSTEM_PROMPT
    assert "materials: final deliverables" in SYSTEM_PROMPT
    assert "requirements: rules" in SYSTEM_PROMPT
    assert "Do not classify development tools" in SYSTEM_PROMPT


def test_prompt_does_not_treat_parenthetical_model_list_as_all_required() -> None:
    assert "candidate options by default" in SYSTEM_PROMPT
    assert "Do not infer that every listed API must be used" in SYSTEM_PROMPT
    assert "at least one API from the listed candidates" in SYSTEM_PROMPT


def test_prompt_requires_readme_md_and_pdf_as_separate_materials() -> None:
    assert "two separate material items" in SYSTEM_PROMPT
    assert "README.md and README PDF" in SYSTEM_PROMPT


def test_prompt_normalizes_team_size_rule() -> None:
    assert "completed by 1 person independently" in SYSTEM_PROMPT
    assert "2 people as a team" in SYSTEM_PROMPT
