from __future__ import annotations

from io import BytesIO

import fitz
import pytest
from docx import Document

from src.parsers import (
    CorruptedFileError,
    EmptyFileError,
    NoExtractableTextError,
    UnsupportedFileTypeError,
    parse_uploaded_file,
)


def _build_docx_bytes(text: str) -> bytes:
    buffer = BytesIO()
    document = Document()
    document.add_paragraph(text)
    document.save(buffer)
    return buffer.getvalue()


def _build_pdf_bytes(text: str | None) -> bytes:
    document = fitz.open()
    page = document.new_page()
    if text is not None:
        page.insert_text((72, 72), text)
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_parse_utf8_chinese_txt() -> None:
    text = "\u8bfe\u7a0b\u9879\u76ee\u901a\u77e5\n\u63d0\u4ea4 README.md"

    result = parse_uploaded_file("notice.TXT", text.encode("utf-8"))

    assert result.file_type == "txt"
    assert result.text == text
    assert result.char_count == len(text)
    assert result.page_count is None


def test_parse_markdown_preserves_list_and_url() -> None:
    markdown = "- README.md\n- Repository: https://example.com/repo\n"

    result = parse_uploaded_file("project.md", markdown.encode("utf-8"))

    assert "- README.md" in result.text
    assert "https://example.com/repo" in result.text
    assert "\n" in result.text


def test_parse_docx() -> None:
    docx_bytes = _build_docx_bytes("Submit README.md and README PDF.")

    result = parse_uploaded_file("notice.docx", docx_bytes)

    assert result.file_type == "docx"
    assert "Submit README.md and README PDF." in result.text
    assert result.page_count is None


def test_parse_text_pdf() -> None:
    pdf_bytes = _build_pdf_bytes("ActionPilot PDF notice")

    result = parse_uploaded_file("notice.PDF", pdf_bytes)

    assert result.file_type == "pdf"
    assert "ActionPilot PDF notice" in result.text
    assert result.page_count == 1


def test_empty_file_raises_clear_exception() -> None:
    with pytest.raises(EmptyFileError, match="empty"):
        parse_uploaded_file("empty.txt", b"")


def test_unsupported_file_type_raises_clear_exception() -> None:
    with pytest.raises(UnsupportedFileTypeError, match="Unsupported file type"):
        parse_uploaded_file("image.png", b"not empty")


def test_pdf_without_text_reports_possible_scanned_pdf() -> None:
    pdf_bytes = _build_pdf_bytes(None)

    with pytest.raises(NoExtractableTextError, match="scanned PDF"):
        parse_uploaded_file("scan.pdf", pdf_bytes)


def test_cleaning_removes_nul_normalizes_newlines_and_compresses_blank_lines() -> None:
    raw = "line 1\x00  \r\n\r\n\r\nline 2\t\rline 3"

    result = parse_uploaded_file("notice.txt", raw.encode("utf-8"))

    assert result.text == "line 1\n\nline 2\nline 3"


def test_corrupted_pdf_raises_user_friendly_exception() -> None:
    with pytest.raises(CorruptedFileError, match="PDF could not be opened"):
        parse_uploaded_file("broken.pdf", b"this is not a pdf")


def test_corrupted_docx_raises_user_friendly_exception() -> None:
    with pytest.raises(CorruptedFileError, match="DOCX could not be opened"):
        parse_uploaded_file("broken.docx", b"this is not a docx")
