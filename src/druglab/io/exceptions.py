"""Custom exceptions for druglab.io."""

from __future__ import annotations


class DrugLabIOError(IOError):
    """Base class for all druglab.io errors."""


class UnsupportedFormatError(DrugLabIOError):
    """Raised when a file format is not recognised or supported."""

    def __init__(self, fmt: str) -> None:
        super().__init__(
            f"Unsupported format: '{fmt}'. "
            "Check druglab.io.SUPPORTED_FORMATS for the list of recognised formats."
        )
        self.fmt = fmt


class RecordParseError(DrugLabIOError):
    """Raised when a single record within a file cannot be parsed."""

    def __init__(self, source: str, record_index: int, reason: str) -> None:
        super().__init__(
            f"Failed to parse record #{record_index} in '{source}': {reason}"
        )
        self.source = source
        self.record_index = record_index
        self.reason = reason