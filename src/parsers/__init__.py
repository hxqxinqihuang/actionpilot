"""Input parser modules."""

from src.parsers.exceptions import (
    CorruptedFileError,
    EmptyFileError,
    FileParseError,
    NoExtractableTextError,
    TextDecodeError,
    UnsupportedFileTypeError,
)
from src.parsers.file_parser import parse_uploaded_file
from src.parsers.models import ParseResult

__all__ = [
    "CorruptedFileError",
    "EmptyFileError",
    "FileParseError",
    "NoExtractableTextError",
    "ParseResult",
    "TextDecodeError",
    "UnsupportedFileTypeError",
    "parse_uploaded_file",
]
