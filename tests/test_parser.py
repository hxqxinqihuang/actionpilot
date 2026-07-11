from __future__ import annotations

from src.parsers.text_parser import parse_text


def test_parse_text_strips_whitespace() -> None:
    assert parse_text("  hello  ") == "hello"
