from __future__ import annotations


class FileParseError(RuntimeError):
    """Base exception for user-facing file parsing errors."""


class EmptyFileError(FileParseError):
    """Raised when the uploaded file is empty."""


class UnsupportedFileTypeError(FileParseError):
    """Raised when a file extension is not supported."""


class TextDecodeError(FileParseError):
    """Raised when a text file cannot be decoded safely."""


class CorruptedFileError(FileParseError):
    """Raised when a PDF or DOCX file appears to be damaged."""


class NoExtractableTextError(FileParseError):
    """Raised when a file has no extractable text."""
