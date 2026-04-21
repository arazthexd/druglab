"""
tests/test_indexing.py
~~~~~~~~~~~~~~~~~~~~~~
Comprehensive tests for druglab.db.indexing — the shared, backend-agnostic
index-normalisation module.

Covers:
1.  normalize_row_index — all input shapes, edge cases, error paths
2.  coerce_bool_mask — validation, length mismatch, wrong dtype
3.  validate_take_index — integer arrays, negative indices, float rejection,
    float allow_float_cast, object dtype rejection, bounds checking
4.  RowSelection — construction via from_raw(), all properties, apply_to(),
    apply_to_list(), repr
5.  Integration smoke-tests verifying that EagerMemoryBackend and BaseTable
    delegate through normalize_row_index (i.e. the shim works and stricter
    errors propagate correctly)
6.  Backward-compatibility: _resolve_idx shim produces identical output to
    normalize_row_index for all accepted input types
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
    INDEX_LIKE,
    RowSelection,
    coerce_bool_mask,
    normalize_row_index,
    validate_take_index,
)
from druglab.db.backend.memory import _resolve_idx
from druglab.db.backend import EagerMemoryBackend
from druglab.db.table import BaseTable, HistoryEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(n: int = 6) -> EagerMemoryBackend:
    return EagerMemoryBackend(
        objects=[{"id": i} for i in range(n)],
        metadata=pd.DataFrame({"val": list(range(n))}),
        features={"fp": np.arange(n * 4, dtype=np.float32).reshape(n, 4)},
    )


class DummyTable(BaseTable[dict]):
    def _serialize_object(self, obj):
        return json.dumps(obj).encode()

    def _deserialize_object(self, raw):
        return json.loads(raw.decode())

    def _object_type_name(self):
        return "dict"


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
# Section 5: _resolve_idx backward-compatibility shim
# ===========================================================================


class TestResolveIdxShim:
    """
    The private ``_resolve_idx`` function in memory.py is now a thin wrapper
    around ``normalize_row_index``.  These tests confirm identical behaviour.
    """

    @pytest.mark.parametrize("idx,n,expected", [
        (None, 10, None),
        (3, 10, [3]),
        (-1, 10, [9]),
        (slice(1, 4), 10, [1, 2, 3]),
        ([0, 2, 4], 10, [0, 2, 4]),
        (np.array([1, 3]), 10, [1, 3]),
    ])
    def test_shim_matches_normalize(self, idx, n, expected):
        shim_result = _resolve_idx(idx, n)
        norm_result = normalize_row_index(idx, n)

        if expected is None:
            assert shim_result is None
            assert norm_result is None
        else:
            assert shim_result.tolist() == expected
            assert norm_result.tolist() == expected

    def test_shim_invalid_type_raises(self):
        with pytest.raises(TypeError):
            _resolve_idx("bad", 10)

    def test_shim_out_of_bounds_raises(self):
        with pytest.raises(IndexError):
            _resolve_idx(99, 10)

    def test_shim_bool_mask(self):
        mask = np.array([True, False, True])
        result = _resolve_idx(mask, 3)
        assert result.tolist() == [0, 2]


# ===========================================================================
# Section 6: Integration — EagerMemoryBackend uses normalize_row_index
# ===========================================================================


class TestBackendIntegration:
    """
    Smoke tests confirming the backend delegates index resolution through
    the shared indexing module (strict error model propagates correctly).
    """

    def test_get_feature_float_array_rejected(self):
        b = _make_backend(6)
        with pytest.raises(TypeError, match="float"):
            b.get_feature("fp", idx=np.array([1.0, 2.0]))

    def test_get_metadata_float_array_rejected(self):
        b = _make_backend(6)
        with pytest.raises(TypeError, match="float"):
            b.get_metadata(idx=np.array([0.0, 1.0]))

    def test_get_objects_float_array_rejected(self):
        b = _make_backend(6)
        with pytest.raises(TypeError):
            b.get_objects(idx=np.array([0.0, 1.0]))

    def test_get_feature_object_array_rejected(self):
        b = _make_backend(6)
        with pytest.raises(TypeError, match="[Oo]bject"):
            b.get_feature("fp", idx=np.array([0, 1], dtype=object))

    def test_get_feature_out_of_bounds_raises(self):
        b = _make_backend(4)
        with pytest.raises(IndexError):
            b.get_feature("fp", idx=np.array([0, 99]))

    def test_get_metadata_bool_mask_wrong_length_raises(self):
        b = _make_backend(6)
        with pytest.raises(IndexError, match="length"):
            b.get_metadata(idx=np.array([True, False]))  # wrong length

    def test_row_selection_is_used_in_create_view(self):
        """create_view routes through RowSelection internally."""
        b = _make_backend(6)
        view = b.create_view([1, 3, 5])
        assert len(view) == 3
        assert view._objects[0] == {"id": 1}
        assert view._metadata["val"].tolist() == [1, 3, 5]


# ===========================================================================
# Section 7: Integration — BaseTable uses normalize_row_index via backend
# ===========================================================================


class TestTableIntegration:
    """
    Confirm that table-level multi-axis indexing propagates strict errors
    from the new indexing module.
    """

    def _make_table(self, n=6):
        objects = [{"id": i} for i in range(n)]
        meta = pd.DataFrame({"val": list(range(n))})
        features = {"fp": np.arange(n * 4, dtype=np.float32).reshape(n, 4)}
        return DummyTable(objects=objects, metadata=meta, features=features)

    def test_feat_pushdown_float_array_rejected(self):
        t = self._make_table()
        from druglab.db.table.base import FEAT
        with pytest.raises(TypeError, match="float"):
            _ = t[FEAT, "fp", np.array([0.0, 1.0])]

    def test_feat_pushdown_object_array_rejected(self):
        t = self._make_table()
        from druglab.db.table.base import FEAT
        with pytest.raises(TypeError):
            _ = t[FEAT, "fp", np.array([0, 1], dtype=object)]

    def test_meta_pushdown_out_of_bounds_raises(self):
        t = self._make_table(4)
        from druglab.db.table.base import META
        with pytest.raises(IndexError):
            _ = t[META, 99]

    def test_row_selection_scalar_input_flag(self):
        """RowSelection.scalar_input is True when a bare int was passed."""
        sel = RowSelection.from_raw(2, 10)
        assert sel.is_scalar is True

        sel_list = RowSelection.from_raw([2], 10)
        assert sel_list.is_scalar is False

    def test_bool_mask_with_wrong_length_raises_at_table_level(self):
        """Table.subset enforces boolean mask length independently."""
        t = self._make_table(4)
        with pytest.raises(ValueError):
            t.subset(np.array([True, False]))  # length 2, table is 4

    def test_normalize_row_index_importable_from_db(self):
        """Top-level db package should export the indexing utilities."""
        from druglab.db import normalize_row_index, RowSelection, coerce_bool_mask, validate_take_index
        result = normalize_row_index([0, 2], 5)
        assert result.tolist() == [0, 2]


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])