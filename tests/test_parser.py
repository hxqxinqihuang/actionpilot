from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from src.parsers import (
    CorruptedFileError,
    EmptyFileError,
    NoExtractableTextError,
    ParseResult,
    UnsupportedFileTypeError,
    parse_uploaded_file,
)
from src.parsers.text_parser import parse_text


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _fixture_bytes(file_name: str) -> bytes:
    return (FIXTURES_DIR / file_name).read_bytes()


def _blank_pdf_bytes() -> bytes:
    document = fitz.open()
    document.new_page()
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def _compact_text(text: str) -> str:
    return text.replace(" ", "")


def test_parse_text_strips_whitespace() -> None:
    assert parse_text("  hello  ") == "hello"


@pytest.mark.parametrize(
    ("file_name", "expected_type", "expected_page_count"),
    [
        ("course_project.txt", "txt", None),
        ("course_project.md", "md", None),
        ("course_project.docx", "docx", None),
        ("course_project.pdf", "pdf", 1),
    ],
)
def test_parse_uploaded_course_project_files(
    file_name: str,
    expected_type: str,
    expected_page_count: int | None,
) -> None:
    result = parse_uploaded_file(file_name, _fixture_bytes(file_name))

    assert isinstance(result, ParseResult)
    assert result.file_name == file_name
    assert result.file_type == expected_type
    assert result.char_count == len(result.text)
    assert result.page_count == expected_page_count
    assert result.warnings == []
    assert "大模型/Agent应用" in _compact_text(result.text)
    assert "README.md" in result.text
    assert "Agent" in result.text


def test_parse_utf8_chinese_txt_contains_key_course_project_content() -> None:
    result = parse_uploaded_file("course_project.txt", _fixture_bytes("course_project.txt"))

    assert result.file_type == "txt"
    assert "项目方向 3" in result.text
    assert "大模型/Agent应用" in _compact_text(result.text)
    assert "调用大模型" in result.text
    assert result.char_count == len(result.text)


def test_parse_markdown_preserves_list_symbols_and_urls() -> None:
    result = parse_uploaded_file("course_project.md", _fixture_bytes("course_project.md"))

    assert "- 实用性为主" in result.text
    assert "https://www.producthunt.com" in result.text
    assert "https://openai.com/zh-Hans-CN/news/product-releases/" in result.text
    assert "\n" in result.text


def test_parse_docx_contains_course_project_key_text() -> None:
    result = parse_uploaded_file("course_project.docx", _fixture_bytes("course_project.docx"))

    assert result.file_type == "docx"
    assert "大模型/Agent应用" in _compact_text(result.text)
    assert "GitHub" in result.text
    assert "Hugging Face" in result.text
    assert result.char_count == len(result.text)


def test_parse_text_pdf_returns_reasonable_page_count() -> None:
    result = parse_uploaded_file("course_project.pdf", _fixture_bytes("course_project.pdf"))

    assert result.file_type == "pdf"
    assert result.page_count is not None
    assert result.page_count >= 1
    assert "大模型/Agent应用" in _compact_text(result.text)
    assert result.char_count == len(result.text)


def test_extension_matching_is_case_insensitive_for_pdf() -> None:
    result = parse_uploaded_file("NOTICE.PDF", _fixture_bytes("course_project.pdf"))

    assert result.file_name == "NOTICE.PDF"
    assert result.file_type == "pdf"
    assert result.page_count is not None
    assert result.page_count >= 1


def test_empty_file_raises_project_exception() -> None:
    with pytest.raises(EmptyFileError, match="empty"):
        parse_uploaded_file("empty.txt", b"")


def test_unsupported_extension_raises_project_exception() -> None:
    with pytest.raises(UnsupportedFileTypeError, match="Unsupported file type"):
        parse_uploaded_file("notice.xlsx", b"not empty")


def test_corrupted_pdf_raises_project_exception() -> None:
    with pytest.raises(CorruptedFileError, match="PDF could not be opened"):
        parse_uploaded_file("broken.pdf", b"not a valid pdf")


def test_corrupted_docx_raises_project_exception() -> None:
    with pytest.raises(CorruptedFileError, match="DOCX could not be opened"):
        parse_uploaded_file("broken.docx", b"not a valid docx")


def test_no_text_pdf_reports_scanned_pdf_without_ocr() -> None:
    with pytest.raises(NoExtractableTextError, match="scanned PDF"):
        parse_uploaded_file("scan.pdf", _blank_pdf_bytes())


def test_cleaning_compresses_blank_lines_without_breaking_structure_or_urls() -> None:
    raw_markdown = (
        "# Title\r\n\r\n\r\n"
        "- item one   \r\n"
        "- item two\t\r\n\r\n\r\n\r\n"
        "https://example.com/path?q=1\r"
        "中文标点：保留。"
    )

    result = parse_uploaded_file("notice.md", raw_markdown.encode("utf-8"))

    assert "\r" not in result.text
    assert "\n\n\n" not in result.text
    assert "- item one" in result.text
    assert "- item two" in result.text
    assert "https://example.com/path?q=1" in result.text
    assert "中文标点：保留。" in result.text
    assert result.char_count == len(result.text)
