"""
Format detection and handler registry for druglab.io.

Every file format is registered in ``_FORMAT_REGISTRY`` with both reader and
writer classes.  New formats can be added via :func:`register_format`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Type

from druglab.io.exceptions import UnsupportedFormatError

if TYPE_CHECKING:
    from druglab.io.readers import BaseFormatReader
    from druglab.io.writers import BaseFormatWriter


# ---------------------------------------------------------------------------
# Supported extension set (lower-case, without leading dot)
# ---------------------------------------------------------------------------

SUPPORTED_FORMATS: frozenset[str] = frozenset(
    {"sdf", "sd", "smi", "smiles", "csv", "tsv", "rxn", "mol"}
)

# Internal registry: ext -> (ReaderClass, WriterClass)
# Populated lazily to avoid circular imports; finalised by _ensure_registry().
_FORMAT_REGISTRY: Dict[str, Tuple[Optional[Type], Optional[Type]]] = {}
_REGISTRY_READY = False


def _ensure_registry() -> None:
    """Populate the registry on first access (avoids circular imports)."""
    global _REGISTRY_READY
    if _REGISTRY_READY:
        return

    # Import concrete classes here to break the circular dependency cycle.
    from druglab.io.readers import (
        SDFFormatReader,
        SMILESFormatReader,
        CSVFormatReader,
        RXNFormatReader,
        MOLFormatReader,
    )
    from druglab.io.writers import (
        SDFWriter,
        SMILESWriter,
        CSVWriter,
        RXNWriter,
        MOLWriter,
    )

    _FORMAT_REGISTRY.update(
        {
            "sdf": (SDFFormatReader, SDFWriter),
            "sd": (SDFFormatReader, SDFWriter),
            "smi": (SMILESFormatReader, SMILESWriter),
            "smiles": (SMILESFormatReader, SMILESWriter),
            "csv": (CSVFormatReader, CSVWriter),
            "tsv": (CSVFormatReader, CSVWriter),
            "rxn": (RXNFormatReader, RXNWriter),
            "mol": (MOLFormatReader, MOLWriter),
        }
    )
    _REGISTRY_READY = True


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def detect_format(path: str) -> str:
    """Return the lower-cased extension (without dot) for *path*.

    Parameters
    ----------
    path:
        File path whose extension determines the format.

    Returns
    -------
    str
        Lower-cased extension, e.g. ``"sdf"``, ``"smi"``, ``"csv"``.

    Raises
    ------
    UnsupportedFormatError
        If the extension is not in :data:`SUPPORTED_FORMATS`.
    """
    _, ext = os.path.splitext(path)
    fmt = ext.lstrip(".").lower()
    if not fmt or fmt not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(fmt or f"(no extension) for '{path}'")
    return fmt


def get_reader_cls(fmt: str) -> "Type[BaseFormatReader]":
    """Return the reader class registered for *fmt*.

    Parameters
    ----------
    fmt:
        Lower-cased format string (e.g. ``"sdf"``).

    Raises
    ------
    UnsupportedFormatError
        If *fmt* has no registered reader.
    """
    _ensure_registry()
    entry = _FORMAT_REGISTRY.get(fmt)
    if entry is None or entry[0] is None:
        raise UnsupportedFormatError(fmt)
    return entry[0]


def get_writer_cls(fmt: str) -> "Type[BaseFormatWriter]":
    """Return the writer class registered for *fmt*.

    Parameters
    ----------
    fmt:
        Lower-cased format string (e.g. ``"sdf"``).

    Raises
    ------
    UnsupportedFormatError
        If *fmt* has no registered writer.
    """
    _ensure_registry()
    entry = _FORMAT_REGISTRY.get(fmt)
    if entry is None or entry[1] is None:
        raise UnsupportedFormatError(fmt)
    return entry[1]


def register_format(
    ext: str,
    reader_cls: "Optional[Type[BaseFormatReader]]" = None,
    writer_cls: "Optional[Type[BaseFormatWriter]]" = None,
) -> None:
    """Register a custom format handler.

    Call this **after** importing ``druglab.io`` to extend the registry with
    your own format reader/writer pair.

    Parameters
    ----------
    ext:
        File extension (without dot, case-insensitive) to register.
    reader_cls:
        A :class:`~druglab.io.readers.BaseFormatReader` subclass, or *None*
        if the format is write-only.
    writer_cls:
        A :class:`~druglab.io.writers.BaseFormatWriter` subclass, or *None*
        if the format is read-only.
    """
    _ensure_registry()
    global SUPPORTED_FORMATS
    ext = ext.lower()
    _FORMAT_REGISTRY[ext] = (reader_cls, writer_cls)
    SUPPORTED_FORMATS = frozenset(SUPPORTED_FORMATS | {ext})