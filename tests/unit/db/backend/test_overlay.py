"""
tests/unit/db/backend/test_overlay.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive tests for OverlayBackend.

Coverage map
------------
1.  Zero-Copy Subsetting              - length, metadata, features, objects
2.  Flattening Nested Overlays        - index composition, data correctness
3.  CoW Feature Addition              - local delta isolation
4.  CoW Mutation                      - metadata / objects / features
5.  Commit                            - flush & clear all three domains
6.  Clone                             - deep-copy independence
7.  Materialize - Features            - full replace, partial, drop
8.  Materialize - Metadata            - full replace, partial, drop
9.  Materialize - Objects             - single and bulk object mutations
10. Materialize - Combined Deltas     - all three domains in one shot
11. Drop + Materialize                - feature/metadata drop survives materialize
12. Save via Overlay                  - round-trip through materialize().save()
13. Nested Overlay Data Correctness   - features / metadata / objects through two levels
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from druglab.db.backend import EagerMemoryBackend
from druglab.db.backend.overlay import OverlayBackend
from druglab.db.table.base import BaseTable, HistoryEntry
from tests.shared.make_dummy_db import (
    BackendContext,
    DictTable,
    TableContext,
    _make_dict_objects,
    _make_features,
    _make_metadata,
    _make_dummy_dict_memory_backend,
    _make_dummy_dict_memory_backend_context,
)


# ---------------------------------------------------------------------------
# Local fixtures (mirror conftest so the file is self-contained when run
# directly; pytest will prefer conftest.py fixtures when both are present)
# ---------------------------------------------------------------------------

@pytest.fixture(name="bctx")
def backend_context() -> BackendContext:
    return _make_dummy_dict_memory_backend_context(
        n_rows=40,
        meta_cols=("col1", "col2"),
        feat_sizes={"feat1": 4, "feat2": 8},
    )


# ===========================================================================
# 1. Zero-Copy Subsetting
# ===========================================================================

class TestZeroCopySubsetting:
    """OverlayBackend created with an explicit index_map gives a coherent view."""

    def test_overlay_length_matches_index_map(self, bctx: BackendContext):
        indices = np.array([0, 5, 10, 15], dtype=np.intp)
        overlay = OverlayBackend(bctx.backend, indices)
        assert len(overlay) == len(indices)

    def test_full_identity_overlay_length(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        assert len(overlay) == bctx.num_rows

    def test_empty_overlay_length(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.array([], dtype=np.intp))
        assert len(overlay) == 0

    # --- metadata ---

    def test_subset_metadata_rows_are_correct(self, bctx: BackendContext):
        indices = np.array([2, 7, 15], dtype=np.intp)
        overlay = OverlayBackend(bctx.backend, indices)
        generated = _make_metadata(bctx.num_rows, *bctx.meta_cols)

        result = overlay.get_metadata()
        assert len(result) == len(indices)

        first_col = bctx.meta_cols[0]
        expected = generated[first_col].iloc[indices].tolist()
        assert result[first_col].tolist() == expected

    def test_subset_metadata_has_correct_columns(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.array([0, 1], dtype=np.intp))
        result_cols = set(overlay.get_metadata().columns)
        expected_cols = set(bctx.meta_cols)
        assert result_cols == expected_cols

    def test_subset_metadata_index_is_reset(self, bctx: BackendContext):
        """Returned DataFrame must have a 0-based RangeIndex."""
        indices = np.array([10, 20, 30], dtype=np.intp)
        overlay = OverlayBackend(bctx.backend, indices)
        result = overlay.get_metadata()
        assert list(result.index) == [0, 1, 2]

    # --- features ---

    def test_subset_feature_shape_is_correct(self, bctx: BackendContext):
        indices = np.array([0, 3, 6], dtype=np.intp)
        overlay = OverlayBackend(bctx.backend, indices)

        for feat_name, feat_dim in bctx.feat_sizes.items():
            arr = overlay.get_feature(feat_name)
            assert arr.shape == (len(indices), feat_dim)

    def test_subset_feature_values_match_base(self, bctx: BackendContext):
        indices = np.array([1, 4, 9], dtype=np.intp)
        overlay = OverlayBackend(bctx.backend, indices)

        feat_name = bctx.feat_names[0]
        base_arr = bctx.backend.get_feature(feat_name)
        expected = base_arr[indices]
        result = overlay.get_feature(feat_name)
        np.testing.assert_array_equal(result, expected)

    def test_get_feature_names_equals_base(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.array([0, 1], dtype=np.intp))
        assert set(overlay.get_feature_names()) == set(bctx.feat_names)

    def test_feature_shape_reported_correctly(self, bctx: BackendContext):
        indices = np.array([0, 1, 2], dtype=np.intp)
        overlay = OverlayBackend(bctx.backend, indices)

        feat_name = bctx.feat_names[0]
        shape = overlay.get_feature_shape(feat_name)
        assert shape == (len(indices), bctx.feat_sizes[feat_name])

    # --- objects ---

    def test_subset_objects_are_correct(self, bctx: BackendContext):
        indices = np.array([0, 2, 4], dtype=np.intp)
        overlay = OverlayBackend(bctx.backend, indices)

        result = overlay.get_objects()
        assert len(result) == len(indices)
        for i, obj in enumerate(result):
            assert obj == {"id": int(indices[i])}

    def test_get_single_object_by_scalar_idx(self, bctx: BackendContext):
        indices = np.array([5, 10, 15], dtype=np.intp)
        overlay = OverlayBackend(bctx.backend, indices)
        # overlay position 1 → base position 10
        obj = overlay.get_objects(idx=1)
        assert obj == {"id": 10}

    def test_subset_does_not_copy_base_objects(self, bctx: BackendContext):
        """OverlayBackend must not hold a copy of all base objects."""
        overlay = OverlayBackend(bctx.backend, np.array([0, 1], dtype=np.intp))
        # local object store should start empty
        assert len(overlay._local_objects) == 0


# ===========================================================================
# 2. Flattening Nested Overlays
# ===========================================================================

class TestNestedOverlayFlattening:

    def test_nested_overlay_points_to_absolute_base(self, bctx: BackendContext):
        base = bctx.backend
        outer = OverlayBackend(base, np.array([0, 1, 2, 3, 4], dtype=np.intp))
        inner = OverlayBackend(outer, np.array([1, 3], dtype=np.intp))

        assert isinstance(inner._base, EagerMemoryBackend)
        assert inner._base is base

    def test_nested_overlay_not_an_overlay_of_overlay(self, bctx: BackendContext):
        base = bctx.backend
        outer = OverlayBackend(base, np.array([0, 2, 4, 6, 8], dtype=np.intp))
        inner = OverlayBackend(outer, np.array([0, 2], dtype=np.intp))

        assert not isinstance(inner._base, OverlayBackend)

    def test_nested_index_map_composition(self, bctx: BackendContext):
        """
        outer maps [0,2,4,6,8] from base (even rows).
        inner selects [1, 3] from outer → should map to [2, 6] in base.
        """
        base = bctx.backend
        outer = OverlayBackend(base, np.array([0, 2, 4, 6, 8], dtype=np.intp))
        inner = OverlayBackend(outer, np.array([1, 3], dtype=np.intp))

        np.testing.assert_array_equal(inner._index_map, np.array([2, 6], dtype=np.intp))

    def test_nested_overlay_data_correct(self, bctx: BackendContext):
        base = bctx.backend
        outer = OverlayBackend(base, np.array([0, 2, 4, 6, 8], dtype=np.intp))
        inner = OverlayBackend(outer, np.array([0, 2], dtype=np.intp))

        meta = inner.get_metadata()
        first_col = bctx.meta_cols[0]

        generated_meta = _make_metadata(bctx.num_rows, *bctx.meta_cols)
        expected_vals = generated_meta.iloc[[0, 4]][first_col].tolist()

        assert meta[first_col].tolist() == expected_vals

    def test_triple_nesting_flattens_correctly(self, bctx: BackendContext):
        if bctx.num_rows < 20:
            pytest.skip("Test requires at least 20 rows for triple nesting verification")

        base = bctx.backend
        o1 = OverlayBackend(base, np.arange(10, dtype=np.intp))        # rows 0-9
        o2 = OverlayBackend(o1, np.arange(5, dtype=np.intp))           # rows 0-4
        o3 = OverlayBackend(o2, np.array([1, 3], dtype=np.intp))       # rows 1, 3

        assert isinstance(o3._base, EagerMemoryBackend)
        assert o3._base is base
        np.testing.assert_array_equal(o3._index_map, np.array([1, 3], dtype=np.intp))


# ===========================================================================
# 3. CoW Feature Addition
# ===========================================================================

class TestCoWFeatureAddition:

    def test_new_feature_readable_from_overlay(self, bctx: BackendContext):
        base = bctx.backend
        indices = np.array([0, 1, 2], dtype=np.intp)
        overlay = OverlayBackend(base, indices)

        new_feat = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
        overlay.update_feature("new_feat", new_feat)
        result = overlay.get_feature("new_feat")
        np.testing.assert_array_equal(result, new_feat)

    def test_new_feature_not_in_base(self, bctx: BackendContext):
        base = bctx.backend
        overlay = OverlayBackend(base, np.array([0, 1, 2], dtype=np.intp))

        overlay.update_feature("new_feat", np.ones((3, 1), dtype=np.float32))
        assert "new_feat" not in base.get_feature_names()

    def test_new_feature_raises_on_base(self, bctx: BackendContext):
        base = bctx.backend
        overlay = OverlayBackend(base, np.array([0, 1, 2], dtype=np.intp))

        overlay.update_feature("new_feat", np.ones((3, 1), dtype=np.float32))

        with pytest.raises(KeyError):
            base.get_feature("new_feat")

    def test_new_feature_stored_in_local_delta(self, bctx: BackendContext):
        base = bctx.backend
        overlay = OverlayBackend(base, np.array([0, 1, 2], dtype=np.intp))

        overlay.update_feature("my_feat", np.zeros((3, 1), dtype=np.float32))
        assert "my_feat" in overlay._local_features

    def test_feature_visible_in_names(self, bctx: BackendContext):
        base = bctx.backend
        overlay = OverlayBackend(base, np.arange(bctx.num_rows, dtype=np.intp))

        name = "extra_col"
        overlay.update_feature(name, np.zeros((bctx.num_rows, 1), dtype=np.float32))

        assert name in overlay.get_feature_names()
        assert name not in base.get_feature_names()


# ===========================================================================
# 4. CoW Mutation
# ===========================================================================

class TestCoWMutation:

    def test_metadata_update_visible_in_overlay(self, bctx: BackendContext):
        base = bctx.backend
        overlay = OverlayBackend(base, np.array([1, 2, 3], dtype=np.intp))

        first_col = bctx.meta_cols[0]
        update_vals = [99, 98, 97]

        overlay.update_metadata(pd.Series(update_vals, name=first_col))
        result = overlay.get_metadata()[first_col].tolist()
        assert result == update_vals

    def test_metadata_update_does_not_touch_base(self, bctx: BackendContext):
        base = bctx.backend
        first_col = bctx.meta_cols[0]
        original_vals = base.get_metadata()[first_col].tolist()

        overlay = OverlayBackend(base, np.array([1, 2, 3], dtype=np.intp))
        overlay.update_metadata(pd.Series([99, 98, 97], name=first_col))

        assert base.get_metadata()[first_col].tolist() == original_vals

    def test_partial_row_update_in_overlay(self, bctx: BackendContext):
        base = bctx.backend
        first_col = bctx.meta_cols[0]
        overlay = OverlayBackend(base, np.arange(bctx.num_rows, dtype=np.intp))

        target_idx = 2
        new_val = 42
        overlay.update_metadata(pd.Series([new_val], name=first_col), idx=np.array([target_idx]))

        assert overlay.get_metadata()[first_col].iloc[target_idx] == new_val
        generated_meta = _make_metadata(bctx.num_rows, *bctx.meta_cols)
        assert base.get_metadata()[first_col].iloc[target_idx] == generated_meta[first_col].iloc[target_idx]

    def test_object_update_visible_in_overlay(self, bctx: BackendContext):
        base = bctx.backend
        overlay = OverlayBackend(base, np.array([0, 1], dtype=np.intp))

        new_obj = {"id": 999}
        overlay.update_objects(new_obj, idx=0)
        assert overlay.get_objects(idx=0) == new_obj

    def test_object_update_does_not_touch_base(self, bctx: BackendContext):
        base = bctx.backend
        overlay = OverlayBackend(base, np.array([0, 1], dtype=np.intp))

        overlay.update_objects({"id": 999}, idx=0)
        assert base.get_objects(idx=0) == {"id": 0}

    def test_feature_update_isolated_to_overlay(self, bctx: BackendContext):
        base = bctx.backend
        feat_name = bctx.feat_names[0]
        feat_dim = bctx.feat_sizes[feat_name]

        orig_feat_data = base.get_feature(feat_name).copy()

        overlay = OverlayBackend(base, np.array([0, 1, 2], dtype=np.intp))
        new_vals = np.full((3, feat_dim), -1.0, dtype=np.float32)

        overlay.update_feature(feat_name, new_vals)

        np.testing.assert_array_equal(base.get_feature(feat_name), orig_feat_data)
        np.testing.assert_array_equal(overlay.get_feature(feat_name), new_vals)


# ===========================================================================
# 5. Commit
# ===========================================================================

class TestCommit:

    def test_commit_flushes_feature_to_base(self, bctx: BackendContext):
        base = bctx.backend
        feat_name = bctx.feat_names[0]
        feat_dim = bctx.feat_sizes[feat_name]

        overlay = OverlayBackend(base, np.arange(bctx.num_rows, dtype=np.intp))

        new_data = np.full((bctx.num_rows, feat_dim), 7.0, dtype=np.float32)
        overlay.update_feature(feat_name, new_data)

        overlay.commit()
        np.testing.assert_array_equal(base.get_feature(feat_name), new_data)

    def test_commit_clears_local_features(self, bctx: BackendContext):
        base = bctx.backend
        feat_name = bctx.feat_names[0]
        overlay = OverlayBackend(base, np.arange(bctx.num_rows, dtype=np.intp))

        overlay.update_feature(feat_name, np.zeros((bctx.num_rows, bctx.feat_sizes[feat_name]), dtype=np.float32))
        overlay.commit()

        assert len(overlay._local_features) == 0

    def test_commit_flushes_metadata_to_base(self, bctx: BackendContext):
        base = bctx.backend
        first_col = bctx.meta_cols[0]
        overlay = OverlayBackend(base, np.arange(bctx.num_rows, dtype=np.intp))

        new_vals = [i + 100 for i in range(bctx.num_rows)]
        overlay.update_metadata(pd.Series(new_vals, name=first_col))

        overlay.commit()
        assert base.get_metadata()[first_col].tolist() == new_vals

    def test_commit_clears_local_metadata(self, bctx: BackendContext):
        base = bctx.backend
        first_col = bctx.meta_cols[0]
        overlay = OverlayBackend(base, np.arange(bctx.num_rows, dtype=np.intp))

        overlay.update_metadata(pd.Series([1] * bctx.num_rows, name=first_col))
        overlay.commit()

        assert overlay._local_metadata is None

    def test_commit_flushes_objects_to_base(self, bctx: BackendContext):
        base = bctx.backend
        overlay = OverlayBackend(base, np.arange(bctx.num_rows, dtype=np.intp))

        new_obj = {"id": 999}
        overlay.update_objects(new_obj, idx=0)

        overlay.commit()
        assert base.get_objects(idx=0) == new_obj

    def test_commit_clears_local_objects(self, bctx: BackendContext):
        base = bctx.backend
        overlay = OverlayBackend(base, np.arange(bctx.num_rows, dtype=np.intp))

        overlay.update_objects({"id": 999}, idx=0)
        overlay.commit()

        assert len(overlay._local_objects) == 0

    def test_commit_partial_index_flushes_correctly(self, bctx: BackendContext):
        """Overlay covers a subset of the base; only those rows should be flushed."""
        if bctx.num_rows < 6:
            pytest.skip("Test requires at least 6 rows")

        base = bctx.backend
        first_col = bctx.meta_cols[0]

        indices = np.array([2, 3, 4], dtype=np.intp)
        overlay = OverlayBackend(base, indices)

        update_vals = [200, 300, 400]
        overlay.update_metadata(pd.Series(update_vals, name=first_col))

        overlay.commit()

        generated_meta = _make_metadata(bctx.num_rows, *bctx.meta_cols)
        full_base_meta = base.get_metadata()[first_col].tolist()

        assert full_base_meta[:2] == generated_meta[first_col].iloc[:2].tolist()
        assert full_base_meta[2:5] == update_vals
        assert full_base_meta[5:] == generated_meta[first_col].iloc[5:].tolist()


# ===========================================================================
# 6. Clone
# ===========================================================================

class TestOverlayClone:

    def test_clone_copies_local_state_without_touching_base(self, bctx: BackendContext):
        base = bctx.backend
        overlay = OverlayBackend(base, np.array([0, 1, 2], dtype=np.intp))

        overlay.update_feature("new_feat", np.array([[1.0], [2.0], [3.0]], dtype=np.float32))
        overlay.add_metadata_column("local_val", np.array([10, 20, 30]))
        overlay.update_objects({"id": 123}, idx=0)
        overlay.drop_feature(bctx.feat_names[0])
        overlay.drop_metadata_columns(bctx.meta_cols[0])

        cloned = overlay.clone()

        assert isinstance(cloned, OverlayBackend)
        assert cloned._base is base
        assert "new_feat" not in base.get_feature_names()

        np.testing.assert_array_equal(
            cloned._local_features["new_feat"],
            overlay._local_features["new_feat"],
        )
        assert cloned._local_objects == overlay._local_objects
        assert cloned._deleted_features == overlay._deleted_features
        assert cloned._deleted_metadata_cols == overlay._deleted_metadata_cols

    def test_clone_is_independent_from_source_overlay(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.array([0, 1, 2], dtype=np.intp))
        overlay.update_feature("new_feat", np.array([[1.0], [2.0], [3.0]], dtype=np.float32))

        cloned = overlay.clone()
        cloned.update_feature("new_feat", np.array([[7.0], [8.0], [9.0]], dtype=np.float32))
        cloned.update_objects({"id": 999}, idx=1)

        np.testing.assert_array_equal(
            overlay.get_feature("new_feat"),
            np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        )
        assert 1 not in overlay._local_objects

    def test_clone_preserves_index_map(self, bctx: BackendContext):
        indices = np.array([3, 7, 11], dtype=np.intp)
        overlay = OverlayBackend(bctx.backend, indices)
        cloned = overlay.clone()
        np.testing.assert_array_equal(cloned._index_map, indices)

    def test_clone_index_map_is_a_copy(self, bctx: BackendContext):
        """Mutating the clone's index_map must not affect the original."""
        indices = np.array([0, 1, 2], dtype=np.intp)
        overlay = OverlayBackend(bctx.backend, indices.copy())
        cloned = overlay.clone()
        cloned._index_map[0] = 99
        assert overlay._index_map[0] == 0


# ===========================================================================
# 7. Materialize – Features
# ===========================================================================

class TestMaterializeFeatures:
    """_apply_materialized_deltas must correctly replay all feature mutations."""

    def test_materialize_includes_existing_base_features(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        concrete = overlay.materialize()

        assert set(concrete.get_feature_names()) == set(bctx.feat_names)

    def test_materialize_new_feature_present_in_concrete(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        new_arr = np.ones((bctx.num_rows, 3), dtype=np.float32) * 42.0
        overlay.update_feature("brand_new", new_arr)

        concrete = overlay.materialize()
        assert "brand_new" in concrete.get_feature_names()
        np.testing.assert_array_equal(concrete.get_feature("brand_new"), new_arr)

    def test_materialize_full_feature_replace(self, bctx: BackendContext):
        feat_name = bctx.feat_names[0]
        feat_dim = bctx.feat_sizes[feat_name]

        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        replacement = np.full((bctx.num_rows, feat_dim), -5.0, dtype=np.float32)
        overlay.update_feature(feat_name, replacement)

        concrete = overlay.materialize()
        np.testing.assert_array_equal(concrete.get_feature(feat_name), replacement)

    def test_materialize_partial_feature_update_correct_rows(self, bctx: BackendContext):
        """Only the rows touched by the overlay should differ; others keep base values."""
        feat_name = bctx.feat_names[0]
        feat_dim = bctx.feat_sizes[feat_name]
        indices = np.array([0, 1, 2], dtype=np.intp)

        overlay = OverlayBackend(bctx.backend, indices)
        patch = np.full((3, feat_dim), 99.0, dtype=np.float32)
        overlay.update_feature(feat_name, patch)

        concrete = overlay.materialize()
        # Concrete covers only the 3 rows in the index map
        assert len(concrete) == len(indices)
        np.testing.assert_array_equal(concrete.get_feature(feat_name), patch)

    def test_materialize_dropped_feature_absent(self, bctx: BackendContext):
        feat_name = bctx.feat_names[0]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.drop_feature(feat_name)

        concrete = overlay.materialize()
        assert feat_name not in concrete.get_feature_names()

    def test_materialize_dropped_feature_not_in_base_after(self, bctx: BackendContext):
        """materialize() must not mutate the original base backend."""
        feat_name = bctx.feat_names[0]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.drop_feature(feat_name)

        overlay.materialize()
        # Base should still have the feature
        assert feat_name in bctx.backend.get_feature_names()

    def test_materialize_base_not_mutated_by_feature_add(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.update_feature("new_one", np.zeros((bctx.num_rows, 2), dtype=np.float32))

        overlay.materialize()
        assert "new_one" not in bctx.backend.get_feature_names()


# ===========================================================================
# 8. Materialize – Metadata
# ===========================================================================

class TestMaterializeMetadata:

    def test_materialize_preserves_base_metadata_columns(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        concrete = overlay.materialize()
        assert set(bctx.meta_cols).issubset(set(concrete.get_metadata().columns))

    def test_materialize_new_metadata_column_present(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        vals = np.arange(bctx.num_rows, dtype=np.int64)
        overlay.add_metadata_column("new_col", vals)

        concrete = overlay.materialize()
        assert "new_col" in concrete.get_metadata().columns
        np.testing.assert_array_equal(concrete.get_metadata()["new_col"].values, vals)

    def test_materialize_metadata_full_replace(self, bctx: BackendContext):
        first_col = bctx.meta_cols[0]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        new_vals = np.arange(bctx.num_rows, dtype=np.int64) + 1000
        overlay.update_metadata(pd.Series(new_vals.tolist(), name=first_col))

        concrete = overlay.materialize()
        np.testing.assert_array_equal(
            concrete.get_metadata()[first_col].values, new_vals
        )

    def test_materialize_dropped_metadata_col_absent(self, bctx: BackendContext):
        first_col = bctx.meta_cols[0]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.drop_metadata_columns(first_col)

        concrete = overlay.materialize()
        assert first_col not in concrete.get_metadata().columns

    def test_materialize_metadata_drop_does_not_affect_base(self, bctx: BackendContext):
        first_col = bctx.meta_cols[0]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.drop_metadata_columns(first_col)

        overlay.materialize()
        assert first_col in bctx.backend.get_metadata().columns

    def test_materialize_subset_metadata_only_covers_subset_rows(self, bctx: BackendContext):
        indices = np.array([5, 10, 15, 20], dtype=np.intp)
        overlay = OverlayBackend(bctx.backend, indices)
        concrete = overlay.materialize()
        assert len(concrete.get_metadata()) == len(indices)

    def test_materialize_metadata_values_subset_correct(self, bctx: BackendContext):
        indices = np.array([0, 3, 7], dtype=np.intp)
        first_col = bctx.meta_cols[0]
        overlay = OverlayBackend(bctx.backend, indices)
        concrete = overlay.materialize()

        generated = _make_metadata(bctx.num_rows, *bctx.meta_cols)
        expected = generated[first_col].iloc[indices].tolist()
        assert concrete.get_metadata()[first_col].tolist() == expected


# ===========================================================================
# 9. Materialize – Objects
# ===========================================================================

class TestMaterializeObjects:

    def test_materialize_objects_correct_for_full_overlay(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        concrete = overlay.materialize()
        assert concrete.get_objects() == bctx.backend.get_objects()

    def test_materialize_mutated_object_present_in_concrete(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        new_obj = {"id": 777, "extra": "x"}
        overlay.update_objects(new_obj, idx=5)

        concrete = overlay.materialize()
        assert concrete.get_objects(idx=5) == new_obj

    def test_materialize_objects_subset_order_preserved(self, bctx: BackendContext):
        indices = np.array([9, 3, 1], dtype=np.intp)
        overlay = OverlayBackend(bctx.backend, indices)
        concrete = overlay.materialize()

        result = concrete.get_objects()
        expected = [{"id": 9}, {"id": 3}, {"id": 1}]
        assert result == expected

    def test_materialize_does_not_mutate_base_objects(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.update_objects({"id": 555}, idx=0)

        overlay.materialize()
        assert bctx.backend.get_objects(idx=0) == {"id": 0}

    def test_materialize_bulk_object_update(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.array([0, 1, 2], dtype=np.intp))
        new_objs = [{"id": 100}, {"id": 101}, {"id": 102}]
        overlay.update_objects(new_objs)

        concrete = overlay.materialize()
        assert concrete.get_objects() == new_objs


# ===========================================================================
# 10. Materialize – Combined Deltas (all three domains)
# ===========================================================================

class TestMaterializeCombinedDeltas:
    """Ensure _apply_materialized_deltas fires the full cooperative chain."""

    def test_all_three_domains_reflected_in_concrete(self, bctx: BackendContext):
        feat_name = bctx.feat_names[0]
        feat_dim = bctx.feat_sizes[feat_name]
        first_col = bctx.meta_cols[0]

        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))

        # Feature delta
        new_feat = np.full((bctx.num_rows, feat_dim), 3.14, dtype=np.float32)
        overlay.update_feature(feat_name, new_feat)

        # Metadata delta
        new_meta_vals = list(range(bctx.num_rows))
        overlay.update_metadata(pd.Series(new_meta_vals, name=first_col))

        # Object delta
        overlay.update_objects({"id": -1}, idx=0)

        concrete = overlay.materialize()

        np.testing.assert_array_almost_equal(concrete.get_feature(feat_name), new_feat)
        assert concrete.get_metadata()[first_col].tolist() == new_meta_vals
        assert concrete.get_objects(idx=0) == {"id": -1}

    def test_combined_deltas_do_not_bleed_into_base(self, bctx: BackendContext):
        feat_name = bctx.feat_names[0]
        first_col = bctx.meta_cols[0]
        feat_dim = bctx.feat_sizes[feat_name]

        original_feat = bctx.backend.get_feature(feat_name).copy()
        original_meta = bctx.backend.get_metadata()[first_col].tolist()
        original_obj0 = bctx.backend.get_objects(idx=0)

        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.update_feature(feat_name, np.zeros((bctx.num_rows, feat_dim), dtype=np.float32))
        overlay.update_metadata(pd.Series([999] * bctx.num_rows, name=first_col))
        overlay.update_objects({"id": -99}, idx=0)

        overlay.materialize()

        np.testing.assert_array_equal(bctx.backend.get_feature(feat_name), original_feat)
        assert bctx.backend.get_metadata()[first_col].tolist() == original_meta
        assert bctx.backend.get_objects(idx=0) == original_obj0

    def test_concrete_is_an_eager_memory_backend(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.update_feature("x", np.ones((bctx.num_rows, 1), dtype=np.float32))
        concrete = overlay.materialize()
        assert isinstance(concrete, EagerMemoryBackend)


# ===========================================================================
# 11. Drop + Materialize
# ===========================================================================

class TestDropAndMaterialize:
    """Dropped names must survive the full materialize() call."""

    def test_dropped_feature_then_readd_not_in_concrete_if_not_readded(self, bctx: BackendContext):
        feat_name = bctx.feat_names[0]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.drop_feature(feat_name)
        concrete = overlay.materialize()
        assert feat_name not in concrete.get_feature_names()

    def test_drop_then_readd_feature_in_concrete(self, bctx: BackendContext):
        feat_name = bctx.feat_names[0]
        feat_dim = bctx.feat_sizes[feat_name]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.drop_feature(feat_name)
        # Re-add with new values
        new_arr = np.full((bctx.num_rows, feat_dim), 7.0, dtype=np.float32)
        overlay.update_feature(feat_name, new_arr)
        concrete = overlay.materialize()
        assert feat_name in concrete.get_feature_names()
        np.testing.assert_array_equal(concrete.get_feature(feat_name), new_arr)

    def test_drop_nonexistent_feature_raises(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.array([0, 1], dtype=np.intp))
        with pytest.raises(KeyError):
            overlay.drop_feature("does_not_exist")

    def test_deleted_feature_raises_keyerror_on_get(self, bctx: BackendContext):
        feat_name = bctx.feat_names[0]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.drop_feature(feat_name)
        with pytest.raises(KeyError):
            overlay.get_feature(feat_name)

    def test_dropped_metadata_col_then_add_back(self, bctx: BackendContext):
        first_col = bctx.meta_cols[0]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.drop_metadata_columns(first_col)
        # Re-add with different values
        overlay.add_metadata_column(first_col, np.arange(bctx.num_rows, dtype=np.int64))
        concrete = overlay.materialize()
        assert first_col in concrete.get_metadata().columns

    def test_drop_all_metadata_cols(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.drop_metadata_columns()  # None → drop all
        concrete = overlay.materialize()
        # All columns from base should be absent
        for col in bctx.meta_cols:
            assert col not in concrete.get_metadata().columns


# ===========================================================================
# 12. Save via Overlay
# ===========================================================================

class TestSaveViaOverlay:
    """overlay.save() internally calls materialize().save() – test round-trip."""

    def test_save_creates_bundle_directory(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "test.dlb"
            table = DictTable(_backend=overlay)
            table.save(bundle)
            assert bundle.is_dir()

    def test_save_and_reload_preserves_object_count(self, bctx: BackendContext):
        indices = np.array([0, 5, 10, 15, 20], dtype=np.intp)
        overlay = OverlayBackend(bctx.backend, indices)

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "test.dlb"
            table = DictTable(_backend=overlay)
            table.save(bundle)
            reloaded = DictTable.load(bundle)
            assert len(reloaded) == len(indices)

    def test_save_and_reload_preserves_metadata(self, bctx: BackendContext):
        indices = np.array([2, 4, 6], dtype=np.intp)
        first_col = bctx.meta_cols[0]
        overlay = OverlayBackend(bctx.backend, indices)

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "test.dlb"
            table = DictTable(_backend=overlay)
            table.save(bundle)
            reloaded = DictTable.load(bundle)

            generated = _make_metadata(bctx.num_rows, *bctx.meta_cols)
            expected = generated[first_col].iloc[indices].tolist()
            assert reloaded.get_metadata()[first_col].tolist() == expected

    def test_save_and_reload_preserves_features(self, bctx: BackendContext):
        indices = np.array([1, 2, 3], dtype=np.intp)
        feat_name = bctx.feat_names[0]
        overlay = OverlayBackend(bctx.backend, indices)

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "test.dlb"
            table = DictTable(_backend=overlay)
            table.save(bundle)
            reloaded = DictTable.load(bundle)

            expected = bctx.backend.get_feature(feat_name)[indices]
            np.testing.assert_array_equal(reloaded.get_feature(feat_name), expected)

    def test_save_overlay_with_local_feature_delta(self, bctx: BackendContext):
        """Local feature mutations must survive the full save/load round-trip."""
        feat_name = bctx.feat_names[0]
        feat_dim = bctx.feat_sizes[feat_name]

        indices = np.array([0, 1, 2], dtype=np.intp)
        overlay = OverlayBackend(bctx.backend, indices)
        patch = np.full((3, feat_dim), 55.0, dtype=np.float32)
        overlay.update_feature(feat_name, patch)

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "test.dlb"
            table = DictTable(_backend=overlay)
            table.save(bundle)
            reloaded = DictTable.load(bundle)

        np.testing.assert_array_equal(reloaded.get_feature(feat_name), patch)

    def test_save_overlay_with_dropped_feature(self, bctx: BackendContext):
        feat_name = bctx.feat_names[0]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.drop_feature(feat_name)

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "test.dlb"
            table = DictTable(_backend=overlay)
            table.save(bundle)
            reloaded = DictTable.load(bundle)

        assert feat_name not in reloaded.get_feature_names()

    def test_save_does_not_mutate_base(self, bctx: BackendContext):
        """Calling save() on an overlay must leave the base backend intact."""
        feat_name = bctx.feat_names[0]
        original_feat = bctx.backend.get_feature(feat_name).copy()

        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.update_feature(feat_name, np.zeros_like(original_feat))

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "test.dlb"
            table = DictTable(_backend=overlay)
            table.save(bundle)

        np.testing.assert_array_equal(bctx.backend.get_feature(feat_name), original_feat)


# ===========================================================================
# 13. Nested Overlay Data Correctness (all three domains)
# ===========================================================================

class TestNestedOverlayDataCorrectness:
    """
    Two-level overlay chains should expose the correctly composed data for
    features, metadata, and objects without materializing into RAM.
    """

    def _make_nested(self, bctx: BackendContext):
        """Return (base, outer, inner) where inner sees rows [2, 6] of base."""
        base = bctx.backend
        outer = OverlayBackend(base, np.array([0, 2, 4, 6, 8], dtype=np.intp))
        inner = OverlayBackend(outer, np.array([1, 3], dtype=np.intp))
        # inner._index_map == [2, 6]
        return base, outer, inner

    # --- features ---

    def test_nested_feature_values_correct(self, bctx: BackendContext):
        base, _, inner = self._make_nested(bctx)
        feat_name = bctx.feat_names[0]
        expected = base.get_feature(feat_name)[[2, 6]]
        result = inner.get_feature(feat_name)
        np.testing.assert_array_equal(result, expected)

    def test_nested_local_feature_in_inner_not_in_outer(self, bctx: BackendContext):
        base, outer, inner = self._make_nested(bctx)
        inner.update_feature("inner_only", np.array([[1.0], [2.0]], dtype=np.float32))
        assert "inner_only" in inner.get_feature_names()
        assert "inner_only" not in outer.get_feature_names()
        assert "inner_only" not in base.get_feature_names()

    def test_nested_local_feature_in_outer_reflected_in_inner(self, bctx: BackendContext):
        """
        A feature added to outer at a position that inner maps to should be
        visible through inner after the inner flattens to base.

        Because inner flattens to base (not outer), outer local features are NOT
        automatically visible through inner — this tests the architectural boundary.
        """
        base, outer, inner = self._make_nested(bctx)
        # outer covers positions [0,2,4,6,8]; inner maps [1,3] of outer → [2,6] of base.
        # A feature added to outer lives only in outer._local_features; inner reads from base.
        outer.update_feature("outer_feat", np.arange(5, dtype=np.float32).reshape(5, 1))
        # inner should NOT see outer's local feature (it bypasses outer and reads base)
        assert "outer_feat" not in inner.get_feature_names()

    # --- metadata ---

    def test_nested_metadata_values_correct(self, bctx: BackendContext):
        base, _, inner = self._make_nested(bctx)
        first_col = bctx.meta_cols[0]
        generated = _make_metadata(bctx.num_rows, *bctx.meta_cols)
        expected = generated[first_col].iloc[[2, 6]].tolist()
        result = inner.get_metadata()[first_col].tolist()
        assert result == expected

    def test_nested_metadata_update_in_inner_visible(self, bctx: BackendContext):
        base, _, inner = self._make_nested(bctx)
        first_col = bctx.meta_cols[0]
        inner.update_metadata(pd.Series([777, 888], name=first_col))
        result = inner.get_metadata()[first_col].tolist()
        assert result == [777, 888]

    def test_nested_metadata_update_in_inner_not_in_base(self, bctx: BackendContext):
        base, _, inner = self._make_nested(bctx)
        first_col = bctx.meta_cols[0]
        original_vals = base.get_metadata()[first_col].tolist()
        inner.update_metadata(pd.Series([777, 888], name=first_col))
        assert base.get_metadata()[first_col].tolist() == original_vals

    # --- objects ---

    def test_nested_objects_values_correct(self, bctx: BackendContext):
        base, _, inner = self._make_nested(bctx)
        result = inner.get_objects()
        assert result == [{"id": 2}, {"id": 6}]

    def test_nested_object_update_in_inner_not_in_base(self, bctx: BackendContext):
        base, _, inner = self._make_nested(bctx)
        inner.update_objects({"id": 999}, idx=0)  # overlay pos 0 → base pos 2
        assert base.get_objects(idx=2) == {"id": 2}

    def test_nested_object_update_in_inner_visible(self, bctx: BackendContext):
        base, _, inner = self._make_nested(bctx)
        inner.update_objects({"id": 999}, idx=0)
        assert inner.get_objects(idx=0) == {"id": 999}

    # --- length ---

    def test_nested_length(self, bctx: BackendContext):
        _, _, inner = self._make_nested(bctx)
        assert len(inner) == 2

    # --- materialize through nested ---

    def test_nested_materialize_produces_correct_concrete(self, bctx: BackendContext):
        base, _, inner = self._make_nested(bctx)
        feat_name = bctx.feat_names[0]
        concrete = inner.materialize()

        assert isinstance(concrete, EagerMemoryBackend)
        assert len(concrete) == 2

        expected_feat = base.get_feature(feat_name)[[2, 6]]
        np.testing.assert_array_equal(concrete.get_feature(feat_name), expected_feat)

        assert concrete.get_objects() == [{"id": 2}, {"id": 6}]