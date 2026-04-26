"""
tests/unit/db/test_indexing.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive tests for druglab.db.indexing — the shared, backend-agnostic
index-normalisation module.

Covers:
1.  normalize_row_index — all input shapes, edge cases, error paths
2.  coerce_bool_mask — validation, length mismatch, wrong dtype
3.  validate_take_index — integer arrays, negative indices, float rejection,
    float allow_float_cast, object dtype rejection, bounds checking
4.  RowSelection — construction via from_raw(), all properties, apply_to(),
    apply_to_list(), repr
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from druglab.db.indexing import (
    RowSelection,
    coerce_bool_mask,
    normalize_row_index,
    validate_take_index,
)
# from druglab.db.backend.memory import _resolve_idx
from druglab.db.backend import EagerMemoryBackend
from druglab.db.table import BaseTable, HistoryEntry

# ===========================================================================
# Section 1: normalize_row_index
# ===========================================================================

class TestNormalizeRowIndex:
    """Unit tests for the core normalisation helper."""

    # --- None ---
    def test_none_returns_none(self):
        assert normalize_row_index(None, 10) is None

    def test_none_with_zero_n(self):
        assert normalize_row_index(None, 0) is None

    # --- Scalar int ---
    def test_scalar_int_positive(self):
        result = normalize_row_index(3, 10)
        assert result.tolist() == [3]
        assert result.dtype == np.intp

    def test_scalar_int_zero(self):
        result = normalize_row_index(0, 5)
        assert result.tolist() == [0]

    def test_scalar_int_negative(self):
        result = normalize_row_index(-1, 10)
        assert result.tolist() == [9]

    def test_scalar_int_negative_boundary(self):
        result = normalize_row_index(-10, 10)
        assert result.tolist() == [0]

    def test_scalar_int_out_of_bounds_positive(self):
        with pytest.raises(IndexError):
            normalize_row_index(10, 10)

    def test_scalar_int_out_of_bounds_negative(self):
        with pytest.raises(IndexError):
            normalize_row_index(-11, 10)

    def test_numpy_integer_scalar(self):
        result = normalize_row_index(np.intp(4), 10)
        assert result.tolist() == [4]

    # --- Slice ---
    def test_slice_basic(self):
        result = normalize_row_index(slice(2, 5), 10)
        assert result.tolist() == [2, 3, 4]

    def test_slice_step(self):
        result = normalize_row_index(slice(0, 10, 2), 10)
        assert result.tolist() == [0, 2, 4, 6, 8]

    def test_slice_open_ends(self):
        result = normalize_row_index(slice(None, None), 5)
        assert result.tolist() == [0, 1, 2, 3, 4]

    def test_slice_empty(self):
        result = normalize_row_index(slice(5, 5), 10)
        assert result.tolist() == []

    def test_slice_negative_start(self):
        result = normalize_row_index(slice(-3, None), 6)
        assert result.tolist() == [3, 4, 5]

    # --- List ---
    def test_list_of_ints(self):
        result = normalize_row_index([0, 2, 4], 10)
        assert result.tolist() == [0, 2, 4]
        assert result.dtype == np.intp

    def test_list_negative_ints(self):
        result = normalize_row_index([-1, -2], 10)
        assert result.tolist() == [9, 8]

    def test_list_empty(self):
        result = normalize_row_index([], 10)
        assert result is not None
        assert len(result) == 0

    def test_list_bool_delegates_to_bool_mask(self):
        result = normalize_row_index([True, False, True], 3)
        assert result.tolist() == [0, 2]

    def test_list_out_of_bounds(self):
        with pytest.raises(IndexError):
            normalize_row_index([0, 99], 10)

    # --- NumPy array ---
    def test_numpy_int_array(self):
        result = normalize_row_index(np.array([1, 3, 5]), 10)
        assert result.tolist() == [1, 3, 5]

    def test_numpy_bool_mask(self):
        mask = np.array([True, False, True, False])
        result = normalize_row_index(mask, 4)
        assert result.tolist() == [0, 2]

    def test_numpy_0d_array(self):
        result = normalize_row_index(np.array(3), 10)
        assert result.tolist() == [3]

    def test_numpy_float_array_rejected_by_default(self):
        with pytest.raises(TypeError, match="float"):
            normalize_row_index(np.array([1.0, 2.0]), 10)

    def test_numpy_float_array_allowed_with_flag(self):
        result = normalize_row_index(np.array([1.0, 2.0]), 10, allow_float_cast=True)
        assert result.tolist() == [1, 2]

    def test_numpy_float_array_lossy_rejected_even_with_flag(self):
        with pytest.raises(TypeError):
            normalize_row_index(np.array([1.5, 2.5]), 10, allow_float_cast=True)

    def test_numpy_object_array_rejected(self):
        with pytest.raises(TypeError, match="[Oo]bject"):
            normalize_row_index(np.array([1, 2], dtype=object), 10)

    # --- Type errors ---
    def test_string_raises_type_error(self):
        with pytest.raises(TypeError):
            normalize_row_index("bad", 10)

    def test_dict_raises_type_error(self):
        with pytest.raises(TypeError):
            normalize_row_index({"a": 1}, 10)

    def test_float_scalar_raises_type_error(self):
        with pytest.raises(TypeError):
            normalize_row_index(3.0, 10)

    # --- Return dtype ---
    def test_output_dtype_is_intp(self):
        for idx in [2, [0, 1], np.array([0, 1]), slice(0, 3)]:
            result = normalize_row_index(idx, 10)
            assert result.dtype == np.intp, f"Expected np.intp for idx={idx!r}"


# ===========================================================================
# Section 2: coerce_bool_mask
# ===========================================================================


class TestCoerceBoolMask:
    def test_basic_true_false(self):
        mask = np.array([True, False, True, False, True])
        result = coerce_bool_mask(mask, 5)
        assert result.tolist() == [0, 2, 4]
        assert result.dtype == np.intp

    def test_all_false(self):
        mask = np.zeros(5, dtype=bool)
        result = coerce_bool_mask(mask, 5)
        assert result.tolist() == []

    def test_all_true(self):
        mask = np.ones(5, dtype=bool)
        result = coerce_bool_mask(mask, 5)
        assert result.tolist() == [0, 1, 2, 3, 4]

    def test_wrong_length_raises_index_error(self):
        with pytest.raises(IndexError, match="length"):
            coerce_bool_mask(np.array([True, False]), 5)

    def test_non_bool_dtype_raises_type_error(self):
        with pytest.raises(TypeError, match="bool"):
            coerce_bool_mask(np.array([1, 0, 1]), 3)

    def test_float_bool_like_raises_type_error(self):
        with pytest.raises(TypeError):
            coerce_bool_mask(np.array([1.0, 0.0, 1.0]), 3)


# ===========================================================================
# Section 3: validate_take_index
# ===========================================================================


class TestValidateTakeIndex:
    def test_basic_int_array(self):
        arr = np.array([0, 2, 4], dtype=np.int64)
        result = validate_take_index(arr, 10)
        assert result.tolist() == [0, 2, 4]
        assert result.dtype == np.intp

    def test_negative_indices_resolved(self):
        arr = np.array([-1, -2], dtype=np.int64)
        result = validate_take_index(arr, 10)
        assert result.tolist() == [9, 8]

    def test_out_of_bounds_positive_raises(self):
        with pytest.raises(IndexError):
            validate_take_index(np.array([0, 99]), 10)

    def test_out_of_bounds_negative_raises(self):
        with pytest.raises(IndexError):
            validate_take_index(np.array([-11]), 10)

    def test_empty_array(self):
        result = validate_take_index(np.array([], dtype=np.int64), 10)
        assert len(result) == 0

    def test_float_array_rejected_by_default(self):
        with pytest.raises(TypeError, match="float"):
            validate_take_index(np.array([1.0, 2.0]), 10)

    def test_float_array_whole_numbers_allowed_with_flag(self):
        result = validate_take_index(np.array([1.0, 2.0]), 10, allow_float_cast=True)
        assert result.tolist() == [1, 2]

    def test_float_array_non_integer_values_rejected(self):
        with pytest.raises(TypeError):
            validate_take_index(np.array([1.5, 2.0]), 10, allow_float_cast=True)

    def test_object_dtype_raises(self):
        with pytest.raises(TypeError, match="[Oo]bject"):
            validate_take_index(np.array([1, 2], dtype=object), 10)

    def test_bool_array_raises(self):
        with pytest.raises(TypeError, match="[Bb]oolean"):
            validate_take_index(np.array([True, False]), 10)

    def test_duplicate_indices_allowed(self):
        result = validate_take_index(np.array([0, 0, 2]), 10)
        assert result.tolist() == [0, 0, 2]

    def test_empty_parent_with_nonempty_index_raises(self):
        with pytest.raises(IndexError):
            validate_take_index(np.array([0]), 0)

    def test_empty_parent_with_empty_index_ok(self):
        result = validate_take_index(np.array([], dtype=np.int64), 0)
        assert len(result) == 0

    def test_rejects_uint64_values_above_intp(self):
        big = np.array([np.iinfo(np.intp).max], dtype=np.uint64) + np.uint64(1)
        with pytest.raises(OverflowError):
            validate_take_index(big, n=10)

    def test_rejects_float_cast_values_above_intp(self):
        big = np.array([float(np.iinfo(np.intp).max) * 2.0], dtype=np.float64)
        with pytest.raises(OverflowError):
            validate_take_index(big, n=10, allow_float_cast=True)

# ===========================================================================
# Section 4: RowSelection
# ===========================================================================


class TestRowSelection:
    # --- Construction ---
    def test_from_raw_none(self):
        sel = RowSelection.from_raw(None, 10)
        assert sel.is_full
        assert sel.positions is None
        assert sel.n == 10
        assert not sel.scalar_input

    def test_from_raw_int(self):
        sel = RowSelection.from_raw(3, 10)
        assert not sel.is_full
        assert sel.positions.tolist() == [3]
        assert sel.scalar_input

    def test_from_raw_negative_int(self):
        sel = RowSelection.from_raw(-1, 10)
        assert sel.positions.tolist() == [9]
        assert sel.scalar_input

    def test_from_raw_slice(self):
        sel = RowSelection.from_raw(slice(1, 4), 10)
        assert sel.positions.tolist() == [1, 2, 3]
        assert not sel.scalar_input

    def test_from_raw_list(self):
        sel = RowSelection.from_raw([0, 2, 4], 10)
        assert sel.positions.tolist() == [0, 2, 4]

    def test_from_raw_bool_mask(self):
        mask = np.array([True, False, True])
        sel = RowSelection.from_raw(mask, 3)
        assert sel.positions.tolist() == [0, 2]

    def test_from_raw_float_rejected(self):
        with pytest.raises(TypeError):
            RowSelection.from_raw(np.array([1.0, 2.0]), 10)

    def test_from_raw_float_allowed_with_flag(self):
        sel = RowSelection.from_raw(np.array([1.0, 2.0]), 10, allow_float_cast=True)
        assert sel.positions.tolist() == [1, 2]

    def test_positions_or_all_caches_for_full_selection(self):
        sel = RowSelection.from_raw(None, 4)
        first = sel.positions_or_all
        second = sel.positions_or_all
        assert first.tolist() == [0, 1, 2, 3]
        assert first is second

    # --- Properties ---
    def test_is_full_true(self):
        sel = RowSelection.from_raw(None, 5)
        assert sel.is_full

    def test_is_full_false(self):
        sel = RowSelection.from_raw([0, 1], 5)
        assert not sel.is_full

    def test_is_empty_true(self):
        sel = RowSelection.from_raw([], 5)
        assert sel.is_empty

    def test_is_empty_false(self):
        sel = RowSelection.from_raw([0], 5)
        assert not sel.is_empty

    def test_is_empty_full_with_zero_n(self):
        sel = RowSelection.from_raw(None, 0)
        assert sel.is_empty

    def test_is_scalar_true(self):
        sel = RowSelection.from_raw(2, 10)
        assert sel.is_scalar

    def test_is_scalar_false_for_list(self):
        sel = RowSelection.from_raw([2], 10)
        assert not sel.is_scalar

    def test_count_full(self):
        sel = RowSelection.from_raw(None, 7)
        assert sel.count == 7

    def test_count_partial(self):
        sel = RowSelection.from_raw([0, 1, 2], 7)
        assert sel.count == 3

    def test_count_empty(self):
        sel = RowSelection.from_raw([], 7)
        assert sel.count == 0

    # --- apply_to ---
    def test_apply_to_full(self):
        arr = np.arange(12).reshape(4, 3)
        sel = RowSelection.from_raw(None, 4)
        result = sel.apply_to(arr)
        assert np.array_equal(result, arr)

    def test_apply_to_partial(self):
        arr = np.arange(12).reshape(4, 3)
        sel = RowSelection.from_raw([1, 3], 4)
        result = sel.apply_to(arr)
        assert result.shape == (2, 3)
        assert np.array_equal(result[0], arr[1])
        assert np.array_equal(result[1], arr[3])

    def test_apply_to_empty(self):
        arr = np.arange(12).reshape(4, 3)
        sel = RowSelection.from_raw([], 4)
        result = sel.apply_to(arr)
        assert result.shape == (0, 3)

    def test_apply_to_returns_copy_for_full(self):
        arr = np.arange(6).reshape(2, 3)
        sel = RowSelection.from_raw(None, 2)
        result = sel.apply_to(arr)
        result[0, 0] = -999
        assert arr[0, 0] != -999

    # --- apply_to_list ---
    def test_apply_to_list_full(self):
        lst = ["a", "b", "c"]
        sel = RowSelection.from_raw(None, 3)
        result = sel.apply_to_list(lst)
        assert result == ["a", "b", "c"]

    def test_apply_to_list_partial(self):
        lst = ["a", "b", "c", "d"]
        sel = RowSelection.from_raw([0, 2], 4)
        result = sel.apply_to_list(lst)
        assert result == ["a", "c"]

    def test_apply_to_list_empty(self):
        lst = ["a", "b", "c"]
        sel = RowSelection.from_raw([], 3)
        assert sel.apply_to_list(lst) == []

    def test_apply_to_list_is_shallow_copy(self):
        lst = [{"x": 1}, {"x": 2}]
        sel = RowSelection.from_raw(None, 2)
        result = sel.apply_to_list(lst)
        # Shallow: same object references
        assert result[0] is lst[0]

    # --- Immutability ---
    def test_row_selection_is_frozen(self):
        sel = RowSelection.from_raw([0, 1], 5)
        with pytest.raises((AttributeError, TypeError)):
            sel.n = 999

# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])