"""
druglab.db.indexing
~~~~~~~~~~~~~~~~~~~
Backend-agnostic index normalisation and the formal ``RowSelection`` type.

This module is the single source of truth for all row-addressing in DrugLab.
Every backend and table that needs to resolve a user-supplied index should
import from here rather than implementing its own logic.

Public API
----------
INDEX_LIKE
    Type alias for accepted raw index inputs.

normalize_row_index(idx, n) -> np.ndarray | None
    Core helper.  Converts any INDEX_LIKE to a 1-D integer ndarray (dtype
    np.intp).  Returns ``None`` when ``idx`` is ``None`` (meaning "all rows").
    Raises ``TypeError`` for unsupported types and ``IndexError`` for
    out-of-bounds positions.  Rejects lossy float/object arrays unless the
    caller opts in via ``allow_float_cast``.

coerce_bool_mask(mask, n) -> np.ndarray
    Strictly validates and converts a boolean mask to a positional index.

validate_take_index(arr, n) -> np.ndarray
    Validates an integer array for use as a fancy index, resolving negatives
    and raising on out-of-bounds entries.

RowSelection
    Dataclass that wraps a resolved positional index array and exposes
    convenience properties (is_scalar, is_full, positions).  Constructed
    via ``RowSelection.from_raw(idx, n)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

INDEX_LIKE = Union[int, slice, List[int], np.ndarray]

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def coerce_bool_mask(mask: np.ndarray, n: int) -> np.ndarray:
    """
    Validate a boolean mask and return a 1-D integer position array.

    Parameters
    ----------
    mask : np.ndarray
        Must have boolean dtype and exactly ``n`` elements.
    n : int
        Expected length of the parent dimension.

    Returns
    -------
    np.ndarray
        1-D ``np.intp`` array of positions where ``mask`` is True.

    Raises
    ------
    IndexError
        If the mask length does not match ``n``.
    TypeError
        If ``mask`` does not have a boolean dtype.
    """
    if not np.issubdtype(mask.dtype, np.bool_):
        raise TypeError(
            f"Boolean mask must have dtype bool; got {mask.dtype}. "
            "Use validate_take_index() for integer arrays."
        )
    if len(mask) != n:
        raise IndexError(
            f"Boolean mask length {len(mask)} does not match "
            f"parent dimension size {n}."
        )
    return np.where(mask)[0].astype(np.intp)


def validate_take_index(
    arr: np.ndarray,
    n: int,
    *,
    allow_float_cast: bool = False,
) -> np.ndarray:
    """
    Validate an integer array for use as a positional (fancy) index.

    Negative values are resolved to their positive equivalents relative to
    ``n``.  Duplicate entries are allowed (they produce duplicate rows, which
    is valid NumPy fancy-indexing behaviour).

    Parameters
    ----------
    arr : np.ndarray
        1-D array of index values.
    n : int
        Size of the parent dimension.
    allow_float_cast : bool, default False
        If True, float arrays whose values are all whole numbers are silently
        cast to ``np.intp``.  If False (the default), float arrays raise a
        ``TypeError`` unconditionally so that lossy accidental casts are caught
        early.

    Returns
    -------
    np.ndarray
        Validated 1-D ``np.intp`` array with negatives resolved.

    Raises
    ------
    TypeError
        If ``arr`` has a non-integer (and non-castable float) dtype, or if it
        has an object dtype.
    IndexError
        If any resolved index is out of bounds for ``n``.
    """
    intp_info = np.iinfo(np.intp)

    if np.issubdtype(arr.dtype, np.bool_):
        raise TypeError(
            "Received a boolean array in validate_take_index(). "
            "Use coerce_bool_mask() for boolean masks."
        )

    if np.issubdtype(arr.dtype, np.floating):
        if not allow_float_cast:
            raise TypeError(
                f"Received float array (dtype={arr.dtype}) as a positional index. "
                "Pass allow_float_cast=True if you are certain the values are "
                "whole numbers, or cast explicitly to an integer dtype first."
            )
        # Check for lossiness before casting
        rounded = np.round(arr)
        if not np.array_equal(arr, rounded):
            raise TypeError(
                f"Float array contains non-integer values and cannot be safely "
                f"cast to a positional index (e.g. {arr[arr != rounded][0]!r})."
            )
        if rounded.size and ((rounded > intp_info.max).any() or (rounded < intp_info.min).any()):
            raise OverflowError(
                "Float index values exceed platform pointer-size integer bounds."
            )
        arr = rounded.astype(np.intp)

    elif arr.dtype == object:
        raise TypeError(
            "Object-dtype arrays are not accepted as positional indices. "
            "Ensure the array contains integers before indexing."
        )

    elif not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(
            f"Positional index array must have an integer dtype; got {arr.dtype}."
        )
    else:
        if arr.size and ((arr > intp_info.max).any() or (arr < intp_info.min).any()):
            raise OverflowError(
                "Index values exceed platform pointer-size integer bounds."
            )
        arr = arr.astype(np.intp, copy=False)

    # Resolve negatives
    neg_mask = arr < 0
    if neg_mask.any():
        arr = arr.copy()
        arr[neg_mask] = n + arr[neg_mask]

    # Bounds check
    if n == 0:
        if len(arr) > 0:
            raise IndexError(
                f"Cannot index into an empty dimension (size 0)."
            )
        return arr

    oob = (arr < 0) | (arr >= n)
    if oob.any():
        bad = arr[oob]
        raise IndexError(
            f"Positional index out of bounds for dimension of size {n}: "
            f"{bad.tolist()!r}"
        )

    return arr


def normalize_row_index(
    idx: Optional[INDEX_LIKE],
    n: int,
    *,
    allow_float_cast: bool = False,
) -> Optional[np.ndarray]:
    """
    Canonical entry-point: convert any supported index representation to a
    1-D ``np.intp`` array, or return ``None`` for "all rows".

    This is the primary function every backend should call.  It supersedes the
    private ``_resolve_idx`` helper in ``backend/memory.py`` (which is now a
    thin compatibility shim delegating here).

    Parameters
    ----------
    idx : None | int | slice | List[int] | np.ndarray
        The user-supplied row selector.
    n : int
        Total number of rows in the parent dimension.
    allow_float_cast : bool, default False
        Forwarded to ``validate_take_index`` when ``idx`` is a float array.

    Returns
    -------
    np.ndarray of dtype np.intp, or None
        ``None``  → caller should use all rows.
        array     → explicit row positions (validated, negatives resolved).

    Raises
    ------
    TypeError
        For unsupported index types, object-dtype arrays, or float arrays
        (when ``allow_float_cast=False``).
    IndexError
        For out-of-bounds or length-mismatched indices.
    """
    # --- None: caller wants all rows ---
    if idx is None:
        return None

    # --- Scalar integer ---
    if isinstance(idx, (int, np.integer)):
        i = int(idx)
        # Resolve negative
        if i < 0:
            if i + n < 0:
                raise IndexError(
                    f"Index {idx} is out of bounds for dimension of size {n}."
                )
            i = n + i
        if i >= n:
            raise IndexError(
                f"Index {idx} is out of bounds for dimension of size {n}."
            )
        return np.array([i], dtype=np.intp)

    # --- Slice ---
    if isinstance(idx, slice):
        return np.arange(*idx.indices(n), dtype=np.intp)

    # --- List ---
    if isinstance(idx, list):
        if len(idx) == 0:
            return np.array([], dtype=np.intp)
        arr = np.asarray(idx)
        # Delegate to the appropriate sub-helper based on dtype
        if np.issubdtype(arr.dtype, np.bool_):
            return coerce_bool_mask(arr, n)
        return validate_take_index(arr, n, allow_float_cast=allow_float_cast)

    # --- NumPy array ---
    if isinstance(idx, np.ndarray):
        if idx.ndim == 0:
            # 0-d array — treat as scalar
            return normalize_row_index(idx.item(), n, allow_float_cast=allow_float_cast)
        if np.issubdtype(idx.dtype, np.bool_):
            return coerce_bool_mask(idx, n)
        return validate_take_index(idx, n, allow_float_cast=allow_float_cast)

    raise TypeError(
        f"Unsupported index type {type(idx).__name__!r}. "
        "Expected None, int, slice, List[int], or np.ndarray."
    )


# ---------------------------------------------------------------------------
# RowSelection dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowSelection:
    """
    Immutable, validated wrapper around a resolved positional index array.

    A ``RowSelection`` maps zero or more *requested* row positions to their
    positions inside a parent array of known size ``n``.  Once constructed,
    callers can interrogate it without worrying about raw index types or
    validation.

    Attributes
    ----------
    positions : np.ndarray
        1-D ``np.intp`` array of resolved row positions, *or* ``None`` when
        the selection covers all rows (i.e. the raw index was ``None``).
    n : int
        Total number of rows in the parent dimension at construction time.
    scalar_input : bool
        True when the caller supplied a bare integer index.  Useful for
        deciding whether to return a single object vs. a list.

    Construction
    ------------
    Use the class-method factory::

        sel = RowSelection.from_raw(idx, n)

    Do **not** call ``__init__`` directly; the ``positions`` field is
    a post-validated array and should never be set manually.
    """

    positions: Optional[np.ndarray]
    n: int
    scalar_input: bool = field(default=False)
    _positions_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_raw(
        cls,
        idx: Optional[INDEX_LIKE],
        n: int,
        *,
        allow_float_cast: bool = False,
    ) -> "RowSelection":
        """
        Validate ``idx`` against dimension size ``n`` and return a
        ``RowSelection``.

        Parameters
        ----------
        idx : INDEX_LIKE or None
            Raw user-supplied index.
        n : int
            Parent dimension size.
        allow_float_cast : bool, default False
            Forwarded to ``normalize_row_index``.

        Returns
        -------
        RowSelection
        """
        scalar = isinstance(idx, (int, np.integer))
        resolved = normalize_row_index(idx, n, allow_float_cast=allow_float_cast)
        return cls(positions=resolved, n=n, scalar_input=scalar)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_full(self) -> bool:
        """True when this selection covers all rows (positions is None)."""
        return self.positions is None

    @property
    def is_empty(self) -> bool:
        """True when the selection resolves to zero rows."""
        if self.positions is None:
            return self.n == 0
        return len(self.positions) == 0

    @property
    def is_scalar(self) -> bool:
        """True when the original input was a bare integer."""
        return self.scalar_input

    @property
    def count(self) -> int:
        """Number of rows selected."""
        if self.positions is None:
            return self.n
        return len(self.positions)

    @property
    def positions_or_all(self) -> np.ndarray:
        """Explicit positions, cached for repeated all-row access."""
        if self.positions is not None:
            return self.positions
        if self._positions_cache is None:
            object.__setattr__(self, "_positions_cache", np.arange(self.n, dtype=np.intp))
        return self._positions_cache

    def apply_to(self, arr: np.ndarray) -> np.ndarray:
        """
        Apply this selection to a NumPy array along axis 0.

        Parameters
        ----------
        arr : np.ndarray
            Source array whose first axis is the row dimension.

        Returns
        -------
        np.ndarray
            Subset view or copy.  When ``is_full``, returns a copy of the
            full array.  When ``is_empty``, returns an empty array preserving
            the trailing shape.
        """
        if self.is_full:
            return arr.copy()
        if self.is_empty:
            return arr[0:0]
        return arr[self.positions]

    def apply_to_list(self, lst: list) -> list:
        """
        Apply this selection to a Python list.

        Parameters
        ----------
        lst : list
            Source list.

        Returns
        -------
        list
            Selected elements.  When ``is_full``, returns a shallow copy.
        """
        if self.is_full:
            return list(lst)
        return [lst[i] for i in self.positions]

    def __repr__(self) -> str:  # pragma: no cover
        if self.is_full:
            return f"RowSelection(all {self.n} rows)"
        return (
            f"RowSelection(positions={self.positions.tolist()!r}, "
            f"n={self.n}, scalar={self.scalar_input})"
        )