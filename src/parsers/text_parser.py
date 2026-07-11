from __future__ import annotations

from src.parsers.cleaning import clean_extracted_text
from src.parsers.exceptions import TextDecodeError
from src.parsers.models import ParseResult


def parse_text(text: str) -> str:
    return clean_extracted_text(text)


def parse_text_bytes(file_name: str, file_bytes: bytes, file_type: str) -> ParseResult:
    try:
        decoded = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            decoded = file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise TextDecodeError(
                "Text file could not be decoded as UTF-8. Please save it as UTF-8 and upload again."
            ) from exc

    if decoded.startswith("\ufeff"):
        decoded = decoded.removeprefix("\ufeff")

    text = clean_extracted_text(decoded)
    return ParseResult(
        text=text,
        file_name=file_name,
        file_type=file_type,
        char_count=len(text),
        page_count=None,
        warnings=[],
    )
