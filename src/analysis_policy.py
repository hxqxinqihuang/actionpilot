from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


STANDARD_MODE_MAX_CHARS = 8000
COMPACT_RECOMMENDED_MAX_CHARS = 15000
MAX_INPUT_CHARS = 30000
EMERGENCY_EXTRACTION_WARNING = "文档信息量较大，本次使用简化提取模式，仅保留核心执行信息。"
COMPACT_MODE_WARNING = "文档内容较多，本次使用紧凑模式，仅保留核心行动信息。"
CORE_ACTION_MODE_WARNING = "文档信息密度较高，本次使用核心行动模式，仅保留截止日期、资格、提交材料和必须完成的事项。"

ExtractionMode = Literal["standard", "compact"]


@dataclass(frozen=True)
class InputLengthCheck:
    allowed: bool
    mode: ExtractionMode
    warning: str | None = None
    error: str | None = None


def choose_extraction_mode(text: str) -> ExtractionMode:
    return "compact" if len(text) > STANDARD_MODE_MAX_CHARS else "standard"


def check_input_length(text: str) -> InputLengthCheck:
    char_count = len(text)
    if char_count > MAX_INPUT_CHARS:
        return InputLengthCheck(
            allowed=False,
            mode="compact",
            error=(
                f"Input is too long ({char_count} characters). Please remove unrelated sections "
                f"and keep it within {MAX_INPUT_CHARS} characters."
            ),
        )
    if char_count > COMPACT_RECOMMENDED_MAX_CHARS:
        return InputLengthCheck(
            allowed=True,
            mode="compact",
            warning=(
                f"Input is {char_count} characters. Compact mode will be used; removing appendices, "
                "reference lists, product descriptions, or unrelated sections is recommended."
            ),
        )
    if char_count > STANDARD_MODE_MAX_CHARS:
        return InputLengthCheck(
            allowed=True,
            mode="compact",
            warning=f"Input is {char_count} characters. Compact mode will be used because the document is long.",
        )
    return InputLengthCheck(allowed=True, mode="standard")
