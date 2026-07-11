from __future__ import annotations

import re


_THREE_OR_MORE_BLANK_LINES = re.compile(r"\n[ \t]*\n[ \t]*\n+")


def clean_extracted_text(text: str) -> str:
    cleaned = text.replace("\x00", "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = "\n".join(line.rstrip(" \t") for line in cleaned.split("\n"))
    cleaned = _THREE_OR_MORE_BLANK_LINES.sub("\n\n", cleaned)
    return cleaned.strip()
