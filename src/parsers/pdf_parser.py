from __future__ import annotations

import fitz

from src.parsers.cleaning import clean_extracted_text
from src.parsers.exceptions import CorruptedFileError, NoExtractableTextError
from src.parsers.models import ParseResult


def parse_pdf(file_name: str, file_bytes: bytes) -> ParseResult:
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise CorruptedFileError("PDF could not be opened. The file may be damaged or not a valid PDF.") from exc

    try:
        page_texts = [page.get_text("text") for page in document]
        page_count = document.page_count
    except Exception as exc:
        raise CorruptedFileError("PDF text could not be extracted. The file may be damaged.") from exc
    finally:
        document.close()

    text = clean_extracted_text("\n\n".join(page_texts))
    if not text:
        raise NoExtractableTextError(
            "PDF has no extractable text. It may be a scanned PDF, and OCR is not supported yet."
        )

    return ParseResult(
        text=text,
        file_name=file_name,
        file_type="pdf",
        char_count=len(text),
        page_count=page_count,
        warnings=[],
    )
