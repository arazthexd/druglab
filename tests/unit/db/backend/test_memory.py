"""
tests/test_db.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive test suite for the refactored druglab.db storage architecture.

Tests cover:
1. Memory Metadata
2. Memory Objects
3. Memory Features
"""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from druglab.db.backend import (
    BaseStorageBackend,
    EagerMemoryBackend,
    MemoryMetadataMixin,
    MemoryObjectMixin,
    MemoryFeatureMixin,
)
from druglab.db.table import BaseTable, HistoryEntry, META, OBJ, FEAT, M, O, F
from tests.shared.make_dummy_db import (
    _make_dummy_dict_memory_backend_context, 
    BackendContext, TableContext,
    _make_metadata, _make_dict_objects, _make_features
)

# ===========================================================================
# Section 1: MemoryMetadataMixin
# ===========================================================================

class TestMemoryMetadataMixin:
    """Test the metadata mixin via EagerMemoryBackend (which composes it)."""

    def _get_generated_meta(self, _context: BackendContext):
        return _make_metadata(_context.num_rows, *_context.meta_cols)

    def test_get_all_metadata(self, bctx: BackendContext):
        df = bctx.backend.get_metadata()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == bctx.num_rows
        assert list(df.columns) == list(bctx.meta_cols)

    def test_get_metadata_idx_int(self, bctx: BackendContext):
        df = bctx.backend.get_metadata(idx=bctx.num_rows-1)
        assert len(df) == 1
        first_col = bctx.meta_cols[0]
        generated_meta = self._get_generated_meta(bctx)
        assert df[first_col].iloc[0] == generated_meta[first_col].iloc[bctx.num_rows-1]

    def test_get_metadata_idx_slice(self, bctx: BackendContext):
        if bctx.num_rows < 4:
            pytest.skip("Num Rows < 4 (Skipping test...)")
            return
        
        df = bctx.backend.get_metadata(idx=slice(1, 4))
        assert len(df) == 3
        first_col = bctx.meta_cols[0]
        generated_meta = self._get_generated_meta(bctx)
        assert df[first_col].tolist() == generated_meta[first_col].iloc[1:4].to_list()

    def test_get_metadata_idx_list(self, bctx: BackendContext):
        indices = [0, 3, 5]
        df = bctx.backend.get_metadata(idx=indices)
        first_col = bctx.meta_cols[0]
        generated_meta = self._get_generated_meta(bctx)
        assert df[first_col].tolist() == generated_meta[first_col].iloc[[0, 3, 5]].to_list()

    def test_get_metadata_idx_none(self, bctx: BackendContext):
        df = bctx.backend.get_metadata(idx=None)
        assert len(df) == bctx.num_rows

    def test_get_metadata_cols_single(self, bctx: BackendContext):
        col_name = bctx.meta_cols[0]
        df = bctx.backend.get_metadata(cols=col_name)
        assert list(df.columns) == [col_name]

    def test_get_metadata_cols_list(self, bctx: BackendContext):
        if len(bctx.meta_cols) < 2:
            pytest.skip("Num Meta Cols < 4 (Skipping test...)")
            return
        
        cols = [bctx.meta_cols[0], bctx.meta_cols[1]]
        df = bctx.backend.get_metadata(cols=cols)
        assert set(df.columns) == set(cols)

    def test_get_metadata_idx_and_cols(self, bctx: BackendContext):
        first_col = bctx.meta_cols[0]
        df = bctx.backend.get_metadata(idx=slice(1, 4), cols=first_col)
        assert len(df) == 3
        assert list(df.columns) == [first_col]

    def test_update_metadata_existing(self, bctx: BackendContext):
        """Verifies in-place update of existing columns."""
        first_col = bctx.meta_cols[0]
        new_values = [i * 10 for i in range(bctx.num_rows)]
        
        bctx.backend.update_metadata(pd.Series(new_values, name=first_col))
        df = bctx.backend.get_metadata()
        assert df[first_col].tolist() == new_values
        
    def test_update_metadata_missing_raises(self, bctx: BackendContext):
        """Strict schema enforcement: update_metadata cannot create columns."""
        with pytest.raises(KeyError):
            # Attempt to update a column name that is guaranteed not to exist
            bctx.backend.update_metadata(pd.Series([1]*bctx.num_rows, name="non_existent_column"))

    def test_add_metadata_column(self, bctx: BackendContext):
        """Schema Evolution: explicitly adding a new column."""
        new_col_name = "extra_feature"
        new_values = [i + 100 for i in range(bctx.num_rows)]
        
        bctx.backend.add_metadata_column(new_col_name, new_values)
        df = bctx.backend.get_metadata()
        assert new_col_name in df.columns
        assert df[new_col_name].tolist() == new_values

# ===========================================================================
# Section 2: MemoryObjectMixin
# ===========================================================================

class TestMemoryObjectMixin:
    """Test the object mixin via EagerMemoryBackend."""

    def _get_generated_objects(self, _context: BackendContext):
        """Helper to generate the expected initial state."""
        return _make_dict_objects(_context.num_rows)

    def test_len(self, bctx: BackendContext):
        assert len(bctx.backend) == bctx.num_rows

    def test_get_single_int(self, bctx: BackendContext):
        idx = 2
        obj = bctx.backend.get_objects(idx)
        generated = self._get_generated_objects(bctx)
        assert obj == generated[idx]

    def test_get_single_negative_int(self, bctx: BackendContext):
        obj = bctx.backend.get_objects(-1)
        generated = self._get_generated_objects(bctx)
        assert obj == generated[-1]

    def test_get_slice(self, bctx: BackendContext):
        if bctx.num_rows < 4:
            pytest.skip("Num Rows < 4 (Skipping test...)")
            return
            
        objs = bctx.backend.get_objects(slice(1, 4))
        assert isinstance(objs, list)
        assert len(objs) == 3
        generated = self._get_generated_objects(bctx)
        assert objs == generated[1:4]

    def test_get_list_of_ints(self, bctx: BackendContext):
        indices = [0, 2, 4]
        # Ensure indices are valid for the fixture size
        indices = [i for i in indices if i < bctx.num_rows]
        
        objs = bctx.backend.get_objects(indices)
        assert len(objs) == len(indices)
        generated = self._get_generated_objects(bctx)
        assert objs == [generated[i] for i in indices]

    def test_update_objects_single(self, bctx: BackendContext):
        idx = 2
        new_obj = {"id": 99}
        bctx.backend.update_objects(new_obj, idx=idx)
        assert bctx.backend.get_objects(idx) == new_obj

    def test_update_objects_negative(self, bctx: BackendContext):
        new_obj = {"id": 999}
        bctx.backend.update_objects(new_obj, idx=-1)
        # Verify at the absolute index
        assert bctx.backend.get_objects(bctx.num_rows - 1) == new_obj
        
    def test_update_objects_batch(self, bctx: BackendContext):
        """Verify vector-first bulk updates."""
        if bctx.num_rows < 3:
            pytest.skip("Num Rows < 3 (Skipping test...)")
            return

        indices = [1, 2]
        updates = [{"id": 10}, {"id": 11}]
        
        bctx.backend.update_objects(updates, idx=indices)
        results = bctx.backend.get_objects(indices)
        assert results == updates

# ===========================================================================
# Section 3: MemoryFeatureMixin
# ===========================================================================

class TestMemoryFeatureMixin:
    """Test the feature mixin via EagerMemoryBackend."""

    def _get_generated_features(self, _context: BackendContext) -> dict:
        """Helper to generate the expected initial feature state."""
        return _make_features(_context.num_rows, **_context.feat_sizes)

    def test_get_feature_names(self, bctx: BackendContext):
        expected_names = set(bctx.feat_names)
        assert set(bctx.backend.get_feature_names()) == expected_names

    def test_get_feature_all(self, bctx: BackendContext):
        feat_name = bctx.feat_names[0]
        arr = bctx.backend.get_feature(feat_name)
        assert arr.shape == (bctx.num_rows, bctx.feat_sizes[feat_name])

    def test_get_feature_idx_none_returns_full(self, bctx: BackendContext):
        feat_name = bctx.feat_names[0]
        arr = bctx.backend.get_feature(feat_name, idx=None)
        assert arr.shape == (bctx.num_rows, bctx.feat_sizes[feat_name])

    def test_get_feature_idx_slice_pushdown(self, bctx: BackendContext):
        """
        STRICT QUERY PUSHDOWN TEST:
        The slice must be applied AT THE BACKEND LEVEL.
        """
        if bctx.num_rows < 6:
             pytest.skip("Test requires at least 6 rows")

        feat_name = bctx.feat_names[0]
        arr = bctx.backend.get_feature(feat_name, idx=slice(1, 4))
        
        # Only 3 rows should be returned
        expected_rows = 3
        expected_dim = bctx.feat_sizes[feat_name]
        
        assert arr.shape == (expected_rows, expected_dim), (
            f"Backend must return ONLY the sliced rows ({expected_rows}), not all {bctx.num_rows}."
        )

        # Verify the actual values using the generator
        generated = self._get_generated_features(bctx)
        expected_data = generated[feat_name][1:4]
        assert np.allclose(arr, expected_data)

    def test_get_feature_idx_int_pushdown(self, bctx: BackendContext):
        feat_name = bctx.feat_names[0]
        idx = 2
        arr = bctx.backend.get_feature(feat_name, idx=idx)
        assert arr.shape == (1, bctx.feat_sizes[feat_name])
        
        generated = self._get_generated_features(bctx)
        assert np.allclose(arr[0], generated[feat_name][idx])

    def test_get_feature_idx_list_pushdown(self, bctx: BackendContext):
        if bctx.num_rows < 6:
             pytest.skip("Test requires at least 6 rows")
             
        feat_name = bctx.feat_names[0]
        indices = [0, 5]
        arr = bctx.backend.get_feature(feat_name, idx=indices)
        assert arr.shape == (len(indices), bctx.feat_sizes[feat_name])
        
        generated = self._get_generated_features(bctx)
        assert np.allclose(arr, generated[feat_name][indices])

    def test_update_feature_new(self, bctx: BackendContext):
        """Verify adding a completely new feature array."""
        new_name = "new_vectors"
        new_dim = 16
        new_feat = np.zeros((bctx.num_rows, new_dim), dtype=np.float32)
        
        bctx.backend.update_feature(new_name, new_feat)
        assert new_name in bctx.backend.get_feature_names()
        assert bctx.backend.get_feature(new_name).shape == (bctx.num_rows, new_dim)

    def test_drop_feature(self, bctx: BackendContext):
        feat_name = bctx.feat_names[0]
        bctx.backend.drop_feature(feat_name)
        assert feat_name not in bctx.backend.get_feature_names()

    def test_drop_feature_missing_raises(self, bctx: BackendContext):
        with pytest.raises(KeyError):
            bctx.backend.drop_feature("nonexistent_feature_key")