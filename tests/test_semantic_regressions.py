from __future__ import annotations

from src.agent.extractor import TaskExtractor
from src.providers.base import LLMProvider


class FakeProvider(LLMProvider):
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> str:
        return self._payload


def test_split_readme_md_and_pdf_materials_with_shared_evidence() -> None:
    evidence = "\u7f16\u5199 README.md \u5e76\u8f6c\u6362\u6210 PDF \u63d0\u4ea4"
    payload = f"""
    {{
      "tasks": [
        {{
          "title": "\u8bfe\u7a0b\u9879\u76ee",
          "summary": "\u63d0\u4ea4 README \u6750\u6599\u3002",
          "deadlines": [{{"raw_text": null, "normalized_date": null, "timezone": null, "status": "missing", "evidence": null}}],
          "prerequisites": [],
          "materials": [
            {{"name": "README.md", "description": null, "required": true, "evidence": "{evidence}"}},
            {{"name": "README PDF", "description": null, "required": true, "evidence": "{evidence}"}}
          ],
          "requirements": [],
          "risks": [],
          "confirmation_questions": []
        }}
      ],
      "source_language": "\u4e2d\u6587",
      "confidence": 0.8
    }}
    """
    extractor = TaskExtractor(FakeProvider(payload))

    result = extractor.extract(evidence)

    material_names = [material.name for material in result.tasks[0].materials]
    assert material_names == ["README.md", "README PDF"]
    assert result.tasks[0].materials[0].evidence == evidence
    assert result.tasks[0].materials[1].evidence == evidence


def test_model_provider_list_is_not_interpreted_as_all_required() -> None:
    evidence = "\u8c03\u7528\u5927\u6a21\u578b\uff08Qwen\u3001Kimi\u3001DeepSeek\u3001\u667a\u8c31\uff09\u7684 API \u505a\u4e00\u4e2a\u5927\u6a21\u578b\u6216 Agent \u5e94\u7528/\u7cfb\u7edf"
    payload = f"""
    {{
      "tasks": [
        {{
          "title": "Agent \u5e94\u7528",
          "summary": "\u4f7f\u7528\u5927\u6a21\u578b API \u5b8c\u6210\u5e94\u7528\u3002",
          "deadlines": [{{"raw_text": null, "normalized_date": null, "timezone": null, "status": "missing", "evidence": null}}],
          "prerequisites": [
            {{"name": "\u5927\u6a21\u578b API \u4f7f\u7528\u80fd\u529b", "description": "\u9700\u8981\u8c03\u7528 Qwen\u3001Kimi\u3001DeepSeek \u6216\u667a\u8c31\u7b49\u81f3\u5c11\u4e00\u79cd\u5927\u6a21\u578b API\u3002", "required": true, "evidence": "{evidence}"}}
          ],
          "materials": [],
          "requirements": [
            {{"description": "\u9879\u76ee\u9700\u8981\u8c03\u7528 Qwen\u3001Kimi\u3001DeepSeek \u6216\u667a\u8c31\u7b49\u81f3\u5c11\u4e00\u79cd\u5927\u6a21\u578b API\u3002", "priority": "must", "evidence": "{evidence}"}}
          ],
          "risks": [],
          "confirmation_questions": []
        }}
      ],
      "source_language": "\u4e2d\u6587",
      "confidence": 0.8
    }}
    """
    extractor = TaskExtractor(FakeProvider(payload))

    result = extractor.extract(evidence)

    requirement = result.tasks[0].requirements[0].description
    assert "\u81f3\u5c11\u4e00\u79cd" in requirement
    assert "\u5168\u90e8" not in requirement
    assert "\u6bcf\u4e00\u4e2a" not in requirement


def test_team_size_rule_is_normalized_without_losing_evidence() -> None:
    evidence = "\u9879\u76ee\u4eba\u6570\u4e3a1\u4eba\u62161-2\u4eba"
    payload = f"""
    {{
      "tasks": [
        {{
          "title": "\u8bfe\u7a0b\u9879\u76ee",
          "summary": "\u5b8c\u6210\u8bfe\u7a0b\u9879\u76ee\u3002",
          "deadlines": [{{"raw_text": null, "normalized_date": null, "timezone": null, "status": "missing", "evidence": null}}],
          "prerequisites": [],
          "materials": [],
          "requirements": [
            {{"description": "\u9879\u76ee\u53ef\u75311\u4eba\u72ec\u7acb\u5b8c\u6210\uff0c\u6216\u75312\u4eba\u7ec4\u961f\u5b8c\u6210\u3002", "priority": "must", "evidence": "{evidence}"}}
          ],
          "risks": [],
          "confirmation_questions": []
        }}
      ],
      "source_language": "\u4e2d\u6587",
      "confidence": 0.8
    }}
    """
    extractor = TaskExtractor(FakeProvider(payload))

    result = extractor.extract(evidence)

    requirement = result.tasks[0].requirements[0]
    assert requirement.priority == "must"
    assert requirement.evidence == evidence
    assert "\u72ec\u7acb\u5b8c\u6210" in requirement.description
    assert "\u7ec4\u961f\u5b8c\u6210" in requirement.description
