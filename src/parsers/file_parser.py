from __future__ import annotations

from pathlib import Path

from src.parsers.docx_parser import parse_docx
from src.parsers.exceptions import EmptyFileError, UnsupportedFileTypeError
from src.parsers.models import ParseResult
from src.parsers.pdf_parser import parse_pdf
from src.parsers.text_parser import parse_text_bytes


SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}


def parse_uploaded_file(file_name: str, file_bytes: bytes) -> ParseResult:
    if not file_bytes:
        raise EmptyFileError("Uploaded file is empty.")

    extension = Path(file_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileTypeError(f"Unsupported file type '{extension or '<none>'}'. Supported types: {supported}.")

    if extension in {".txt", ".md"}:
        return parse_text_bytes(file_name=file_name, file_bytes=file_bytes, file_type=extension.removeprefix("."))
    if extension == ".docx":
        return parse_docx(file_name=file_name, file_bytes=file_bytes)
    return parse_pdf(file_name=file_name, file_bytes=file_bytes)
