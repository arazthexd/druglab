"""
tests/integration/test_overlay_table_use.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive tests for OverlayBackend.
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
from tests.shared.make_dummy_db import BackendContext, TableContext, _make_metadata, _make_dict_objects

# ===========================================================================
# 1. Zero-Copy Subsetting
# ===========================================================================

# ...

# ===========================================================================
# 2. Flattening Nested Overlays
# ===========================================================================

class TestNestedOverlayFlattening:

    def test_nested_overlay_points_to_absolute_base(self, bctx: BackendContext):
        base = bctx.backend
        outer = OverlayBackend(base, np.array([0, 1, 2, 3, 4], dtype=np.intp))
        inner = OverlayBackend(outer, np.array([1, 3], dtype=np.intp))
        
        # Flattening logic: inner._base should skip 'outer' and point to 'base'
        assert isinstance(inner._base, EagerMemoryBackend)
        assert inner._base is base

    def test_nested_overlay_not_an_overlay_of_overlay(self, bctx: BackendContext):
        base = bctx.backend
        outer = OverlayBackend(base, np.array([0, 2, 4, 6, 8], dtype=np.intp))
        inner = OverlayBackend(outer, np.array([0, 2], dtype=np.intp))
        
        # The architecture should prevent linked lists of overlays
        assert not isinstance(inner._base, OverlayBackend)

    def test_nested_index_map_composition(self, bctx: BackendContext):
        """
        outer maps [0,2,4,6,8] from base (even rows).
        inner selects [1, 3] from outer → should map to [2, 6] in base.
        """
        base = bctx.backend
        outer = OverlayBackend(base, np.array([0, 2, 4, 6, 8], dtype=np.intp))
        inner = OverlayBackend(outer, np.array([1, 3], dtype=np.intp))
        
        # Composition: base[outer[inner]] -> base[[0,2,4,6,8][1,3]] -> base[[2, 6]]
        np.testing.assert_array_equal(inner._index_map, np.array([2, 6], dtype=np.intp))

    def test_nested_overlay_data_correct(self, bctx: BackendContext):
        base = bctx.backend
        outer = OverlayBackend(base, np.array([0, 2, 4, 6, 8], dtype=np.intp))
        inner = OverlayBackend(outer, np.array([0, 2], dtype=np.intp))
        
        # inner sees rows 0 and 4 from base
        meta = inner.get_metadata()
        first_col = bctx.meta_cols[0]
        
        # Ground truth check
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
        
        # Data specifically for the subset size (3)
        new_feat = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
        
        overlay.update_feature("new_feat", new_feat)
        result = overlay.get_feature("new_feat")
        np.testing.assert_array_equal(result, new_feat)

    def test_new_feature_not_in_base(self, bctx: BackendContext):
        base = bctx.backend
        overlay = OverlayBackend(base, np.array([0, 1, 2], dtype=np.intp))
        
        overlay.update_feature("new_feat", np.ones((3, 1), dtype=np.float32))
        
        # Verify isolation: base should not see the new feature
        assert "new_feat" not in base.get_feature_names()

    def test_new_feature_raises_on_base(self, bctx: BackendContext):
        base = bctx.backend
        overlay = OverlayBackend(base, np.array([0, 1, 2], dtype=np.intp))
        
        overlay.update_feature("new_feat", np.ones((3, 1), dtype=np.float32))
        
        # Strict check: direct access on base should raise KeyError
        with pytest.raises(KeyError):
            base.get_feature("new_feat")

    def test_new_feature_stored_in_local_delta(self, bctx: BackendContext):
        base = bctx.backend
        overlay = OverlayBackend(base, np.array([0, 1, 2], dtype=np.intp))
        
        overlay.update_feature("my_feat", np.zeros((3, 1), dtype=np.float32))
        
        # Verify the architecture: check the internal delta dictionary
        assert "my_feat" in overlay._local_features

    def test_feature_visible_in_names(self, bctx: BackendContext):
        base = bctx.backend
        # Identity mapping subset
        overlay = OverlayBackend(base, np.arange(bctx.num_rows, dtype=np.intp))
        
        name = "extra_col"
        overlay.update_feature(name, np.zeros((bctx.num_rows, 1), dtype=np.float32))
        
        # Overlay should show union of base + local names
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
        
        # Base must remain identical to its state before the overlay mutation
        assert base.get_metadata()[first_col].tolist() == original_vals

    def test_partial_row_update_in_overlay(self, bctx: BackendContext):
        base = bctx.backend
        first_col = bctx.meta_cols[0]
        overlay = OverlayBackend(base, np.arange(bctx.num_rows, dtype=np.intp))
        
        target_idx = 2
        new_val = 42
        overlay.update_metadata(pd.Series([new_val], name=first_col), idx=np.array([target_idx]))
        
        assert overlay.get_metadata()[first_col].iloc[target_idx] == new_val
        # Check ground truth from generator for the base
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
        # Base should still have the original object (id: 0)
        assert base.get_objects(idx=0) == {"id": 0}

    def test_feature_update_isolated_to_overlay(self, bctx: BackendContext):
        base = bctx.backend
        feat_name = bctx.feat_names[0]
        feat_dim = bctx.feat_sizes[feat_name]
        
        # Save original state
        orig_feat_data = base.get_feature(feat_name).copy()
        
        overlay = OverlayBackend(base, np.array([0, 1, 2], dtype=np.intp))
        new_vals = np.full((3, feat_dim), -1.0, dtype=np.float32)
        
        overlay.update_feature(feat_name, new_vals)
        
        # Base unchanged
        np.testing.assert_array_equal(base.get_feature(feat_name), orig_feat_data)
        # Overlay reflects local change
        np.testing.assert_array_equal(overlay.get_feature(feat_name), new_vals)

# ===========================================================================
# 5. Commit
# ===========================================================================

class TestCommit:

    def test_commit_flushes_feature_to_base(self, bctx: BackendContext):
        base = bctx.backend
        feat_name = bctx.feat_names[0]
        feat_dim = bctx.feat_sizes[feat_name]
        
        # Identity overlay
        overlay = OverlayBackend(base, np.arange(bctx.num_rows, dtype=np.intp))
        
        new_data = np.full((bctx.num_rows, feat_dim), 7.0, dtype=np.float32)
        overlay.update_feature(feat_name, new_data)
        
        overlay.commit()
        # Verify base now holds the data previously in overlay's delta
        np.testing.assert_array_equal(base.get_feature(feat_name), new_data)

    def test_commit_clears_local_features(self, bctx: BackendContext):
        base = bctx.backend
        feat_name = bctx.feat_names[0]
        overlay = OverlayBackend(base, np.arange(bctx.num_rows, dtype=np.intp))
        
        overlay.update_feature(feat_name, np.zeros((bctx.num_rows, bctx.feat_sizes[feat_name]), dtype=np.float32))
        overlay.commit()
        
        # Local delta must be purged after flush
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
        
        # Local metadata buffer should be reset to None
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
        
        # Local object dictionary should be empty
        assert len(overlay._local_objects) == 0

    def test_commit_partial_index_flushes_correctly(self, bctx: BackendContext):
        """Overlay covers a subset of the base; only those rows should be flushed."""
        if bctx.num_rows < 6:
            pytest.skip("Test requires at least 6 rows")
            
        base = bctx.backend
        first_col = bctx.meta_cols[0]
        
        # Target indices [2, 3, 4]
        indices = np.array([2, 3, 4], dtype=np.intp)
        overlay = OverlayBackend(base, indices)
        
        update_vals = [200, 300, 400]
        overlay.update_metadata(pd.Series(update_vals, name=first_col))
        
        overlay.commit()
        
        # Ground truth for comparison
        generated_meta = _make_metadata(bctx.num_rows, *bctx.meta_cols)
        full_base_meta = base.get_metadata()[first_col].tolist()
        
        # Rows before and after the subset should remain untouched
        assert full_base_meta[:2] == generated_meta[first_col].iloc[:2].tolist()
        assert full_base_meta[2:5] == update_vals
        assert full_base_meta[5:] == generated_meta[first_col].iloc[5:].tolist()

######################## BELOW IS INCOMING  #################################

# ...