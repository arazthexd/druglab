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
14. View Configuration                - allowlisting and read-only column slicing
15. Caching Behaviour                 - prefetch, cache invalidation, commit ignores cache
16. Detach / Attach                   - DetachedStateError, schema validation on attach

Architecture note
-----------------
Delta state is now encapsulated in isolated dataclass containers instead of flat
``self`` attributes.  The correct attribute paths are:

    overlay._object_store.delta.local          (Dict[int, Any])
    overlay._feature_store.delta.local         (Dict[str, np.ndarray])
    overlay._feature_store.delta.deleted       (Set[str])
    overlay._metadata_store.delta.local         (pd.DataFrame | None)
    overlay._metadata_store.delta.deleted_cols  (Set[str])
    overlay._feature_store.cache.data          (Dict[str, np.ndarray])
    overlay._metadata_store.cache.data          (pd.DataFrame | None)
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
from druglab.db.backend.overlay.identity import DetachedStateError, SchemaIdentity
from druglab.db.table.base import BaseTable, HistoryEntry
from druglab.db.utils import object_pkl_reader, object_pkl_writer
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
# Local fixtures – mirror conftest so the file is self-contained
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
        """OverlayBackend must not hold a copy of all base objects in its delta."""
        overlay = OverlayBackend(bctx.backend, np.array([0, 1], dtype=np.intp))
        # Using the new delta dataclass attribute
        assert len(overlay._object_store.delta.local) == 0


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

    def test_new_feature_stored_in_feat_delta_local(self, bctx: BackendContext):
        """New feature must live in the FeatureDelta.local dict, not a flat attribute."""
        base = bctx.backend
        overlay = OverlayBackend(base, np.array([0, 1, 2], dtype=np.intp))

        overlay.update_feature("my_feat", np.zeros((3, 1), dtype=np.float32))
        # New architecture: delta is encapsulated in _feat_delta dataclass
        assert "my_feat" in overlay._feature_store.delta.local

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

    def test_add_metadata_column_visible_in_overlay(self, bctx: BackendContext):
        base = bctx.backend
        overlay = OverlayBackend(base, np.arange(bctx.num_rows, dtype=np.intp))

        vals = np.arange(bctx.num_rows, dtype=np.int64) + 500
        overlay.add_metadata_column("brand_new_col", vals)

        result = overlay.get_metadata()["brand_new_col"].tolist()
        assert result == vals.tolist()

    def test_add_metadata_column_not_in_base(self, bctx: BackendContext):
        base = bctx.backend
        overlay = OverlayBackend(base, np.arange(bctx.num_rows, dtype=np.intp))

        overlay.add_metadata_column("brand_new_col", np.zeros(bctx.num_rows, dtype=np.int64))
        assert "brand_new_col" not in base.get_metadata_columns()

    def test_object_delta_stored_in_dataclass(self, bctx: BackendContext):
        """Mutations must be stored in ObjectDelta.local, not a flat attribute."""
        base = bctx.backend
        overlay = OverlayBackend(base, np.array([0, 1, 2], dtype=np.intp))

        overlay.update_objects({"id": 77}, idx=1)
        # New architecture uses _obj_delta.local dict
        assert 1 in overlay._object_store.delta.local
        assert overlay._object_store.delta.local[1] == {"id": 77}

    def test_feature_delta_stored_in_dataclass(self, bctx: BackendContext):
        """Feature mutations must be stored in FeatureDelta.local."""
        base = bctx.backend
        feat_name = bctx.feat_names[0]
        overlay = OverlayBackend(base, np.arange(bctx.num_rows, dtype=np.intp))

        new_arr = np.zeros((bctx.num_rows, bctx.feat_sizes[feat_name]), dtype=np.float32)
        overlay.update_feature(feat_name, new_arr)

        assert feat_name in overlay._feature_store.delta.local

    def test_metadata_delta_stored_in_dataclass(self, bctx: BackendContext):
        """Metadata mutations must land in MetadataDelta.local DataFrame."""
        base = bctx.backend
        first_col = bctx.meta_cols[0]
        overlay = OverlayBackend(base, np.arange(bctx.num_rows, dtype=np.intp))

        overlay.update_metadata(pd.Series([99] * bctx.num_rows, name=first_col))

        assert overlay._metadata_store.delta.local is not None
        assert first_col in overlay._metadata_store.delta.local.columns


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

    def test_commit_clears_feat_delta(self, bctx: BackendContext):
        """After commit, FeatureDelta.local must be empty."""
        base = bctx.backend
        feat_name = bctx.feat_names[0]
        overlay = OverlayBackend(base, np.arange(bctx.num_rows, dtype=np.intp))

        overlay.update_feature(feat_name, np.zeros((bctx.num_rows, bctx.feat_sizes[feat_name]), dtype=np.float32))
        overlay.commit()

        # New architecture: delta is cleared via FeatureDelta.clear()
        assert len(overlay._feature_store.delta.local) == 0

    def test_commit_flushes_metadata_to_base(self, bctx: BackendContext):
        base = bctx.backend
        first_col = bctx.meta_cols[0]
        overlay = OverlayBackend(base, np.arange(bctx.num_rows, dtype=np.intp))

        new_vals = [i + 100 for i in range(bctx.num_rows)]
        overlay.update_metadata(pd.Series(new_vals, name=first_col))

        overlay.commit()
        assert base.get_metadata()[first_col].tolist() == new_vals

    def test_commit_clears_meta_delta(self, bctx: BackendContext):
        """After commit, MetadataDelta.local must be None."""
        base = bctx.backend
        first_col = bctx.meta_cols[0]
        overlay = OverlayBackend(base, np.arange(bctx.num_rows, dtype=np.intp))

        overlay.update_metadata(pd.Series([1] * bctx.num_rows, name=first_col))
        overlay.commit()

        # New architecture: _meta_delta.local cleared by MetadataDelta.clear()
        assert overlay._metadata_store.delta.local is None

    def test_commit_flushes_objects_to_base(self, bctx: BackendContext):
        base = bctx.backend
        overlay = OverlayBackend(base, np.arange(bctx.num_rows, dtype=np.intp))

        new_obj = {"id": 999}
        overlay.update_objects(new_obj, idx=0)

        overlay.commit()
        assert base.get_objects(idx=0) == new_obj

    def test_commit_clears_obj_delta(self, bctx: BackendContext):
        """After commit, ObjectDelta.local must be empty."""
        base = bctx.backend
        overlay = OverlayBackend(base, np.arange(bctx.num_rows, dtype=np.intp))

        overlay.update_objects({"id": 999}, idx=0)
        overlay.commit()

        # New architecture: _obj_delta.local cleared by ObjectDelta.clear()
        assert len(overlay._object_store.delta.local) == 0

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

    def test_commit_does_not_clear_cache(self, bctx: BackendContext):
        """Commit flushes deltas but must NOT clear the prefetch cache."""
        base = bctx.backend
        feat_name = bctx.feat_names[0]
        overlay = OverlayBackend(base, np.arange(bctx.num_rows, dtype=np.intp))

        # Prefetch to populate cache
        overlay.prefetch(features=[feat_name])
        assert overlay._feature_store.cache.has(feat_name)

        # Now commit (no delta, but cache should survive)
        overlay.commit()
        assert overlay._feature_store.cache.has(feat_name), (
            "Cache must survive commit(); only the delta is cleared."
        )


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

        # New architecture: check via delta dataclasses
        np.testing.assert_array_equal(
            cloned._feature_store.delta.local["new_feat"],
            overlay._feature_store.delta.local["new_feat"],
        )
        assert cloned._object_store.delta.local == overlay._object_store.delta.local
        assert cloned._feature_store.delta.deleted == overlay._feature_store.delta.deleted
        assert cloned._metadata_store.delta.deleted_cols == overlay._metadata_store.delta.deleted_cols

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
        # New architecture: check via delta dataclass
        assert 1 not in overlay._object_store.delta.local

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

    def test_clone_does_not_share_cache(self, bctx: BackendContext):
        """Each clone must receive a fresh (empty) cache, not a shared reference."""
        feat_name = bctx.feat_names[0]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.prefetch(features=[feat_name])

        cloned = overlay.clone()
        # Clone should have an empty (not shared) cache per the implementation contract
        assert not cloned._feature_store.cache.has(feat_name), (
            "Clone must start with an independent, empty cache."
        )

    def test_clone_delta_is_independent_deep_copy(self, bctx: BackendContext):
        """Mutating the original's delta post-clone must not affect the clone."""
        feat_name = bctx.feat_names[0]
        feat_dim = bctx.feat_sizes[feat_name]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        arr = np.ones((bctx.num_rows, feat_dim), dtype=np.float32)
        overlay.update_feature(feat_name, arr)

        cloned = overlay.clone()

        # Mutate source delta in-place
        overlay._feature_store.delta.local[feat_name][:] = 99.0

        # Clone must be unaffected
        assert not np.all(cloned._feature_store.delta.local[feat_name] == 99.0)


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

    def test_deleted_feature_tracked_in_delta_dataclass(self, bctx: BackendContext):
        """Dropped feature name must be in FeatureDelta.deleted set."""
        feat_name = bctx.feat_names[0]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.drop_feature(feat_name)
        assert feat_name in overlay._feature_store.delta.deleted

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

    def test_deleted_meta_col_tracked_in_delta_dataclass(self, bctx: BackendContext):
        """Dropped column name must be in MetadataDelta.deleted_cols set."""
        first_col = bctx.meta_cols[0]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.drop_metadata_columns(first_col)
        assert first_col in overlay._metadata_store.delta.deleted_cols


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
            table.save(bundle, object_writer=object_pkl_writer)
            assert bundle.is_dir()

    def test_save_and_reload_preserves_object_count(self, bctx: BackendContext):
        indices = np.array([0, 5, 10, 15, 20], dtype=np.intp)
        overlay = OverlayBackend(bctx.backend, indices)

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "test.dlb"
            table = DictTable(_backend=overlay)
            table.save(bundle, object_writer=object_pkl_writer)
            reloaded = DictTable.load(bundle, object_reader=object_pkl_reader)
            assert len(reloaded) == len(indices)

    def test_save_and_reload_preserves_metadata(self, bctx: BackendContext):
        indices = np.array([2, 4, 6], dtype=np.intp)
        first_col = bctx.meta_cols[0]
        overlay = OverlayBackend(bctx.backend, indices)

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "test.dlb"
            table = DictTable(_backend=overlay)
            table.save(bundle, object_writer=object_pkl_writer)
            reloaded = DictTable.load(bundle, object_reader=object_pkl_reader)

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
            table.save(bundle, object_writer=object_pkl_writer)
            reloaded = DictTable.load(bundle, object_reader=object_pkl_reader)

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
            table.save(bundle, object_writer=object_pkl_writer)
            reloaded = DictTable.load(bundle, object_reader=object_pkl_reader)

        np.testing.assert_array_equal(reloaded.get_feature(feat_name), patch)

    def test_save_overlay_with_dropped_feature(self, bctx: BackendContext):
        feat_name = bctx.feat_names[0]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.drop_feature(feat_name)

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "test.dlb"
            table = DictTable(_backend=overlay)
            table.save(bundle, object_writer=object_pkl_writer)
            reloaded = DictTable.load(bundle, object_reader=object_pkl_reader)

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
            table.save(bundle, object_writer=object_pkl_writer)

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

    def test_nested_local_feature_in_outer_not_visible_through_inner(self, bctx: BackendContext):
        """
        A feature added to outer lives in outer's delta; inner reads from base,
        bypassing outer's delta, so inner should NOT see outer's local feature.
        """
        base, outer, inner = self._make_nested(bctx)
        outer.update_feature("outer_feat", np.arange(5, dtype=np.float32).reshape(5, 1))
        # inner bypasses outer and reads base directly
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


# ===========================================================================
# 14. View Configuration
# ===========================================================================

class TestViewConfiguration:
    """
    set_view() installs an allowlist + column slices.  Accessing names outside
    the allowlist must raise KeyError; mutating column-sliced features must raise
    RuntimeError; clear_view() restores unrestricted access.
    """

    # --- Feature allowlist ---

    def test_set_view_features_allowlist_restricts_access(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        allowed = [bctx.feat_names[0]]
        blocked = bctx.feat_names[1]
        overlay.set_view(features=allowed)

        # Allowed feature is readable
        arr = overlay.get_feature(allowed[0])
        assert arr.shape[0] == bctx.num_rows

        # Blocked feature raises KeyError
        with pytest.raises(KeyError):
            overlay.get_feature(blocked)

    def test_set_view_features_allowlist_affects_get_feature_names(self, bctx: BackendContext):
        allowed = [bctx.feat_names[0]]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.set_view(features=allowed)

        names = overlay.get_feature_names()
        assert set(names) == set(allowed)

    def test_set_view_features_allowlist_blocks_update(self, bctx: BackendContext):
        """Attempting to update a blocked feature must raise KeyError."""
        blocked = bctx.feat_names[1]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.set_view(features=[bctx.feat_names[0]])

        with pytest.raises(KeyError):
            overlay.update_feature(
                blocked,
                np.zeros((bctx.num_rows, bctx.feat_sizes[blocked]), dtype=np.float32),
            )

    # --- Metadata column allowlist ---

    def test_set_view_meta_cols_allowlist_restricts_access(self, bctx: BackendContext):
        if len(bctx.meta_cols) < 2:
            pytest.skip("Need at least 2 metadata columns")

        allowed_col = bctx.meta_cols[0]
        blocked_col = bctx.meta_cols[1]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.set_view(meta_cols=[allowed_col])

        meta = overlay.get_metadata()
        assert allowed_col in meta.columns
        assert blocked_col not in meta.columns

    def test_set_view_meta_cols_allowlist_affects_get_metadata_columns(self, bctx: BackendContext):
        if len(bctx.meta_cols) < 2:
            pytest.skip("Need at least 2 metadata columns")

        allowed_col = bctx.meta_cols[0]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.set_view(meta_cols=[allowed_col])

        cols = overlay.get_metadata_columns()
        assert cols == [allowed_col]

    def test_set_view_meta_cols_blocks_add_on_blocked_col(self, bctx: BackendContext):
        if len(bctx.meta_cols) < 2:
            pytest.skip("Need at least 2 metadata columns")

        blocked_col = bctx.meta_cols[1]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.set_view(meta_cols=[bctx.meta_cols[0]])

        with pytest.raises(KeyError):
            overlay.add_metadata_column(blocked_col, np.zeros(bctx.num_rows))

    # --- Column slicing (read-only) ---

    def test_set_view_feature_col_slice_returns_correct_columns(self, bctx: BackendContext):
        feat_name = bctx.feat_names[0]
        feat_dim = bctx.feat_sizes[feat_name]
        if feat_dim < 2:
            pytest.skip("Feature dim must be ≥ 2 to slice")

        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.set_view(feature_col_slices={feat_name: (0, 2)})

        arr = overlay.get_feature(feat_name)
        assert arr.shape == (bctx.num_rows, 2)

    def test_set_view_feature_col_slice_values_match_base(self, bctx: BackendContext):
        feat_name = bctx.feat_names[0]
        feat_dim = bctx.feat_sizes[feat_name]
        if feat_dim < 3:
            pytest.skip("Feature dim must be ≥ 3 to slice columns 1:3")

        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.set_view(feature_col_slices={feat_name: (1, 3)})

        result = overlay.get_feature(feat_name)
        expected = bctx.backend.get_feature(feat_name)[:, 1:3]
        np.testing.assert_array_equal(result, expected)

    def test_set_view_feature_col_slice_is_read_only(self, bctx: BackendContext):
        """Mutating a column-sliced feature must raise RuntimeError."""
        feat_name = bctx.feat_names[0]
        feat_dim = bctx.feat_sizes[feat_name]
        if feat_dim < 2:
            pytest.skip("Feature dim must be ≥ 2 to slice")

        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.set_view(feature_col_slices={feat_name: (0, 2)})

        with pytest.raises(RuntimeError):
            overlay.update_feature(
                feat_name,
                np.zeros((bctx.num_rows, 2), dtype=np.float32),
            )

    def test_set_view_col_slice_shape_reported_correctly(self, bctx: BackendContext):
        feat_name = bctx.feat_names[0]
        feat_dim = bctx.feat_sizes[feat_name]
        if feat_dim < 2:
            pytest.skip("Feature dim must be ≥ 2 to slice")

        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.set_view(feature_col_slices={feat_name: (0, 2)})

        shape = overlay.get_feature_shape(feat_name)
        assert shape == (bctx.num_rows, 2)

    # --- clear_view ---

    def test_clear_view_restores_full_access(self, bctx: BackendContext):
        """After clear_view(), all features and columns should be accessible."""
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.set_view(features=[bctx.feat_names[0]])

        overlay.clear_view()

        # Both features should now be accessible
        for name in bctx.feat_names:
            arr = overlay.get_feature(name)
            assert arr.shape[0] == bctx.num_rows

    def test_clear_view_restores_meta_col_access(self, bctx: BackendContext):
        if len(bctx.meta_cols) < 2:
            pytest.skip("Need at least 2 metadata columns")

        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.set_view(meta_cols=[bctx.meta_cols[0]])
        overlay.clear_view()

        cols = set(overlay.get_metadata_columns())
        assert set(bctx.meta_cols).issubset(cols)

    def test_set_view_replaces_previous_config(self, bctx: BackendContext):
        """Calling set_view() a second time must entirely replace the old config."""
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))

        # First view: allow only first feature
        overlay.set_view(features=[bctx.feat_names[0]])
        with pytest.raises(KeyError):
            overlay.get_feature(bctx.feat_names[1])

        # Second view: allow only second feature
        overlay.set_view(features=[bctx.feat_names[1]])
        arr = overlay.get_feature(bctx.feat_names[1])
        assert arr.shape[0] == bctx.num_rows

        with pytest.raises(KeyError):
            overlay.get_feature(bctx.feat_names[0])

    def test_no_view_means_all_features_visible(self, bctx: BackendContext):
        """Default (no set_view call) must allow unrestricted access."""
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        for name in bctx.feat_names:
            arr = overlay.get_feature(name)
            assert arr.shape[0] == bctx.num_rows


# ===========================================================================
# 15. Caching Behaviour
# ===========================================================================

class TestCachingBehaviour:
    """
    prefetch() fills FeatureCache / MetadataCache.
    Read order: Delta → Cache → Base.
    Mutations invalidate the relevant cache entry.
    commit() ignores the cache.
    """

    def test_prefetch_populates_feature_cache(self, bctx: BackendContext):
        feat_name = bctx.feat_names[0]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))

        assert not overlay._feature_store.cache.has(feat_name)
        overlay.prefetch(features=[feat_name])
        assert overlay._feature_store.cache.has(feat_name)

    def test_prefetch_feature_values_match_base(self, bctx: BackendContext):
        feat_name = bctx.feat_names[0]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.prefetch(features=[feat_name])

        cached = overlay._feature_store.cache.get(feat_name)
        base_arr = bctx.backend.get_feature(feat_name)
        np.testing.assert_array_equal(cached, base_arr)

    def test_prefetch_multiple_features(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.prefetch(features=list(bctx.feat_names))
        for name in bctx.feat_names:
            assert overlay._feature_store.cache.has(name)

    def test_feature_read_hits_cache_not_base(self, bctx: BackendContext):
        """
        After prefetch, get_feature must return the cached value even if
        we corrupt the cache entry directly (proving it reads cache, not base).
        """
        feat_name = bctx.feat_names[0]
        feat_dim = bctx.feat_sizes[feat_name]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.prefetch(features=[feat_name])

        # Inject a sentinel value directly into the cache
        sentinel = np.full((bctx.num_rows, feat_dim), -999.0, dtype=np.float32)
        overlay._feature_store.cache.put(feat_name, sentinel)

        result = overlay.get_feature(feat_name)
        np.testing.assert_array_equal(result, sentinel)

    def test_mutation_invalidates_feature_cache(self, bctx: BackendContext):
        """update_feature must evict the corresponding cache entry."""
        feat_name = bctx.feat_names[0]
        feat_dim = bctx.feat_sizes[feat_name]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.prefetch(features=[feat_name])
        assert overlay._feature_store.cache.has(feat_name)

        overlay.update_feature(feat_name, np.zeros((bctx.num_rows, feat_dim), dtype=np.float32))

        assert not overlay._feature_store.cache.has(feat_name), (
            "Cache entry must be evicted after update_feature()."
        )

    def test_mutation_writes_to_delta_not_cache(self, bctx: BackendContext):
        """After update_feature, data must be in the delta, not the cache."""
        feat_name = bctx.feat_names[0]
        feat_dim = bctx.feat_sizes[feat_name]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))

        new_arr = np.full((bctx.num_rows, feat_dim), 5.0, dtype=np.float32)
        overlay.update_feature(feat_name, new_arr)

        assert overlay._feature_store.delta.has(feat_name)
        assert not overlay._feature_store.cache.has(feat_name)

    def test_prefetch_noop_when_already_in_delta(self, bctx: BackendContext):
        """prefetch() must not overwrite a delta entry with base data."""
        feat_name = bctx.feat_names[0]
        feat_dim = bctx.feat_sizes[feat_name]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))

        sentinel = np.full((bctx.num_rows, feat_dim), 77.0, dtype=np.float32)
        overlay.update_feature(feat_name, sentinel)  # writes to delta

        overlay.prefetch(features=[feat_name])  # should be a no-op for this name

        # Reading should return delta (sentinel), not base data
        result = overlay.get_feature(feat_name)
        np.testing.assert_array_equal(result, sentinel)

    def test_delta_takes_priority_over_cache(self, bctx: BackendContext):
        """Delta must shadow the cache when both are populated."""
        feat_name = bctx.feat_names[0]
        feat_dim = bctx.feat_sizes[feat_name]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))

        # Populate cache
        overlay.prefetch(features=[feat_name])
        # Then write a different value to delta
        delta_arr = np.full((bctx.num_rows, feat_dim), 42.0, dtype=np.float32)
        overlay.update_feature(feat_name, delta_arr)
        # Cache was evicted; delta value should be returned
        result = overlay.get_feature(feat_name)
        np.testing.assert_array_equal(result, delta_arr)

    def test_commit_leaves_cache_intact(self, bctx: BackendContext):
        """commit() must flush the delta but leave the cache populated."""
        feat_name = bctx.feat_names[0]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))

        overlay.prefetch(features=[feat_name])
        overlay.commit()  # nothing in delta, but cache should remain

        assert overlay._feature_store.cache.has(feat_name), (
            "Prefetch cache must not be cleared by commit()."
        )

    def test_prefetch_metadata_populates_meta_cache(self, bctx: BackendContext):
        first_col = bctx.meta_cols[0]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))

        overlay.prefetch(meta_cols=[first_col])
        assert overlay._metadata_store.cache.has_col(first_col)

    def test_metadata_mutation_invalidates_meta_cache(self, bctx: BackendContext):
        first_col = bctx.meta_cols[0]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.prefetch(meta_cols=[first_col])
        assert overlay._metadata_store.cache.has_col(first_col)

        overlay.update_metadata(pd.Series([0] * bctx.num_rows, name=first_col))

        assert not overlay._metadata_store.cache.has_col(first_col), (
            "Metadata cache entry must be evicted after update_metadata()."
        )

    def test_prefetch_when_detached_raises(self, bctx: BackendContext):
        """Calling prefetch() on a detached overlay must raise DetachedStateError."""
        feat_name = bctx.feat_names[0]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.detach()

        with pytest.raises(DetachedStateError):
            overlay.prefetch(features=[feat_name])


# ===========================================================================
# 16. Detach / Attach
# ===========================================================================

class TestDetachAttach:
    """
    detach() severs the base reference.
    Reads that miss the delta + cache raise DetachedStateError.
    attach() restores the base only when schema-compatible.
    """

    def test_detach_sets_base_to_none(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.detach()
        assert overlay._base is None

    def test_is_detached_property(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        assert not overlay.is_detached

        overlay.detach()
        assert overlay.is_detached

    def test_detached_feature_read_raises_when_no_delta_no_cache(self, bctx: BackendContext):
        feat_name = bctx.feat_names[0]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.detach()

        with pytest.raises(DetachedStateError):
            overlay.get_feature(feat_name)

    def test_detached_feature_read_succeeds_from_delta(self, bctx: BackendContext):
        """Reads that hit the delta must succeed even when detached."""
        feat_name = bctx.feat_names[0]
        feat_dim = bctx.feat_sizes[feat_name]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))

        sentinel = np.full((bctx.num_rows, feat_dim), 55.0, dtype=np.float32)
        overlay.update_feature(feat_name, sentinel)
        overlay.detach()

        result = overlay.get_feature(feat_name)
        np.testing.assert_array_equal(result, sentinel)

    def test_detached_feature_read_succeeds_from_cache(self, bctx: BackendContext):
        """Reads that hit the prefetch cache must succeed even when detached."""
        feat_name = bctx.feat_names[0]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))

        overlay.prefetch(features=[feat_name])
        base_arr = bctx.backend.get_feature(feat_name).copy()
        overlay.detach()

        result = overlay.get_feature(feat_name)
        np.testing.assert_array_equal(result, base_arr)

    def test_detached_object_read_raises_for_unmodified_row(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.detach()

        with pytest.raises(DetachedStateError):
            overlay.get_objects(idx=0)

    def test_detached_object_read_from_delta_succeeds(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.update_objects({"id": 42}, idx=3)
        overlay.detach()

        result = overlay.get_objects(idx=3)
        assert result == {"id": 42}

    def test_detached_metadata_read_raises_without_cache(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.detach()

        with pytest.raises(DetachedStateError):
            overlay.get_metadata()

    def test_attach_restores_base(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.detach()
        assert overlay.is_detached

        overlay.attach(bctx.backend)
        assert not overlay.is_detached
        assert overlay._base is bctx.backend

    def test_attach_allows_read_again(self, bctx: BackendContext):
        feat_name = bctx.feat_names[0]
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.detach()
        overlay.attach(bctx.backend)

        result = overlay.get_feature(feat_name)
        assert result.shape[0] == bctx.num_rows

    def test_attach_rejects_wrong_uuid(self, bctx: BackendContext):
        """attach() must raise ValueError when UUIDs differ."""
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.detach()

        # Build a different backend with same shape but different UUID
        different_backend = EagerMemoryBackend(
            objects=_make_dict_objects(bctx.num_rows),
            metadata=_make_metadata(bctx.num_rows, *bctx.meta_cols),
            features=_make_features(bctx.num_rows, **bctx.feat_sizes),
        )

        with pytest.raises(ValueError, match="UUID mismatch"):
            overlay.attach(different_backend)

    def test_attach_rejects_mismatched_row_count(self, bctx: BackendContext):
        """attach() must raise ValueError when row counts differ."""
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))

        # Clone the backend (gets a new UUID) but we'll directly capture its identity
        # by using a backend with same UUID but fewer rows (simulate truncation)
        # We force this by patching schema_uuid on a different-sized backend.
        different_size_backend = EagerMemoryBackend(
            objects=_make_dict_objects(bctx.num_rows - 5),
            metadata=_make_metadata(bctx.num_rows - 5, *bctx.meta_cols),
            features=_make_features(bctx.num_rows - 5, **bctx.feat_sizes),
        )
        # Spoof the UUID to match so we can test the row-count check in isolation
        different_size_backend.schema_uuid = bctx.backend.schema_uuid

        overlay.detach()
        with pytest.raises(ValueError, match="Row count mismatch"):
            overlay.attach(different_size_backend)

    def test_attach_rejects_mismatched_feature_schema(self, bctx: BackendContext):
        """attach() must raise ValueError when feature schemas differ."""
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))

        backend_no_feat = EagerMemoryBackend(
            objects=_make_dict_objects(bctx.num_rows),
            metadata=_make_metadata(bctx.num_rows, *bctx.meta_cols),
            features={},  # empty features
        )
        backend_no_feat.schema_uuid = bctx.backend.schema_uuid

        overlay.detach()
        with pytest.raises(ValueError, match="Feature schema mismatch"):
            overlay.attach(backend_no_feat)

    def test_attach_rejects_mismatched_meta_schema(self, bctx: BackendContext):
        """attach() must raise ValueError when metadata schemas differ."""
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))

        backend_extra_col = EagerMemoryBackend(
            objects=_make_dict_objects(bctx.num_rows),
            metadata=_make_metadata(bctx.num_rows, *bctx.meta_cols, "surprise_col"),
            features=_make_features(bctx.num_rows, **bctx.feat_sizes),
        )
        backend_extra_col.schema_uuid = bctx.backend.schema_uuid

        overlay.detach()
        with pytest.raises(ValueError, match="Metadata schema mismatch"):
            overlay.attach(backend_extra_col)

    def test_commit_while_detached_raises(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.arange(bctx.num_rows, dtype=np.intp))
        overlay.update_feature(
            bctx.feat_names[0],
            np.zeros((bctx.num_rows, bctx.feat_sizes[bctx.feat_names[0]]), dtype=np.float32),
        )
        overlay.detach()

        with pytest.raises(DetachedStateError):
            overlay.commit()

    def test_schema_identity_capture(self, bctx: BackendContext):
        """SchemaIdentity.capture() must faithfully record backend attributes."""
        identity = SchemaIdentity.capture(bctx.backend)

        assert identity.uuid == bctx.backend.schema_uuid
        assert identity.n_rows == bctx.num_rows
        assert identity.feature_names == tuple(sorted(bctx.feat_names))
        assert identity.meta_cols == tuple(sorted(bctx.meta_cols))

    def test_schema_identity_validate_compatible_same(self, bctx: BackendContext):
        """Identical identities must validate without error."""
        a = SchemaIdentity.capture(bctx.backend)
        b = SchemaIdentity.capture(bctx.backend)
        a.validate_compatible(b)  # must not raise

    def test_schema_identity_validate_multiple_errors_reported(self, bctx: BackendContext):
        """validate_compatible must list all mismatches in the error message."""
        a = SchemaIdentity.capture(bctx.backend)
        b = SchemaIdentity(
            uuid="totally-different-uuid",
            n_rows=bctx.num_rows + 1,
            feature_names=(),
            meta_cols=(),
        )
        with pytest.raises(ValueError) as exc_info:
            a.validate_compatible(b)

        msg = str(exc_info.value)
        assert "UUID mismatch" in msg
        assert "Row count mismatch" in msg

# ===========================================================================
# 17. Append
# ===========================================================================

class TestVirtualOverlayAppend:
    def test_overlay_append_uses_virtual_rows(self, bctx: BackendContext):
        overlay = OverlayBackend(bctx.backend, np.arange(3, dtype=np.intp))
        new_objects = [{"id": 100 + i} for i in range(5)]
        new_meta = pd.DataFrame({"col1": list(range(5)), "col2": list(range(5, 10))})
        new_feat = {"feat1": np.ones((5, 4), dtype=np.float32), "feat2": np.ones((5, 8), dtype=np.float32)}
        overlay.append(new_objects, new_meta, new_feat)
        assert len(overlay) == 8
        np.testing.assert_array_equal(overlay.get_feature("feat1", idx=slice(3, 8)), np.ones((5, 4), dtype=np.float32))

# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])