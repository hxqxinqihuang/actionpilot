from __future__ import annotations

from src.analysis_policy import CORE_ACTION_MODE_WARNING
from src.providers.openai_compatible import OpenAICompatibleProvider


def test_core_warning_is_added_to_core_result_json() -> None:
    provider = object.__new__(OpenAICompatibleProvider)
    raw_json = """
    {
      "tasks": [
        {
          "title": "Competition",
          "summary": "核心行动信息",
          "deadlines": [
            {"raw_text": "9月5日前发送所有相关材料", "normalized_date": null, "timezone": null, "status": "found", "evidence": "9月5日前发送所有相关材料"},
            {"raw_text": "9月15日前提交作品", "normalized_date": null, "timezone": null, "status": "found", "evidence": "9月15日前提交作品"}
          ],
          "prerequisites": [],
          "materials": [{"name": "作品材料", "description": null, "required": true, "evidence": "提交作品"}],
          "requirements": [{"description": "符合报名资格", "priority": "must", "evidence": "报名资格"}],
          "risks": [],
          "confirmation_questions": [{"question": "文件中存在两个作品提交相关日期，请确认各自对应的提交环节。", "reason": "存在多个提交相关日期。"}]
        }
      ],
      "source_language": "中文",
      "confidence": 0.8,
      "warnings": []
    }
    """

    result = provider._add_warning_to_json(raw_json, CORE_ACTION_MODE_WARNING)

    assert CORE_ACTION_MODE_WARNING in result
    assert '"deadlines"' in result
    assert '"materials"' in result
    assert '"requirements"' in result
    assert "文件中存在两个作品提交相关日期" in result
