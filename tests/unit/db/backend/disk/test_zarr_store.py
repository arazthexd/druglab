"""
tests/unit/db/backend/disk/test_zarr_store.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for ZarrFeatureStore: I/O, auto-creation, chunking, and the
-1 virtual-index safety guardrail.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

zarr = pytest.importorskip("zarr", reason="zarr not installed")

from druglab.db.backend.disk import ZarrFeatureStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_zarr(tmp_path):
    """Open a fresh Zarr group in a temp directory."""
    return tmp_path / "features.zarr", tmp_path


@pytest.fixture
def store_with_data(tmp_zarr):
    """A ZarrFeatureStore pre-populated with two features."""
    zarr_path, tmp_path = tmp_zarr
    store = ZarrFeatureStore(zarr_path)
    store.update_feature("fp", np.arange(20, dtype=np.float32).reshape(5, 4))
    store.update_feature("desc", np.ones((5, 8), dtype=np.float64))
    return store, tmp_path


# ===========================================================================
# Basic I/O
# ===========================================================================

class TestZarrFeatureStoreIO:
    def test_update_and_get_full(self, tmp_zarr):
        zarr_path, _ = tmp_zarr
        store = ZarrFeatureStore(zarr_path)
        arr = np.arange(12, dtype=np.float32).reshape(4, 3)
        store.update_feature("fp", arr)
        np.testing.assert_array_equal(store.get_feature("fp"), arr)

    def test_update_partial_index(self, store_with_data):
        store, _ = store_with_data
        idx = np.array([0, 2, 4], dtype=np.intp)
        new_vals = np.zeros((3, 4), dtype=np.float32)
        store.update_feature("fp", new_vals, idx=idx)
        result = store.get_feature("fp", idx=idx)
        np.testing.assert_array_equal(result, new_vals)

    def test_get_partial_index(self, store_with_data):
        store, _ = store_with_data
        full = store.get_feature("fp")
        idx = np.array([1, 3], dtype=np.intp)
        partial = store.get_feature("fp", idx=idx)
        np.testing.assert_array_equal(partial, full[idx])

    def test_drop_feature(self, store_with_data):
        store, _ = store_with_data
        store.drop_feature("desc")
        assert "desc" not in store.get_feature_names()
        with pytest.raises(KeyError):
            store.get_feature("desc")

    def test_get_feature_names(self, store_with_data):
        store, _ = store_with_data
        names = store.get_feature_names()
        assert set(names) == {"fp", "desc"}

    def test_get_feature_shape(self, store_with_data):
        store, _ = store_with_data
        assert store.get_feature_shape("fp") == (5, 4)
        assert store.get_feature_shape("desc") == (5, 8)

    def test_n_rows(self, store_with_data):
        store, _ = store_with_data
        assert store.n_rows() == 5

    def test_n_rows_empty(self, tmp_zarr):
        zarr_path, _ = tmp_zarr
        store = ZarrFeatureStore(zarr_path)
        assert store.n_rows() == 0

    def test_update_full_array_replaces(self, store_with_data):
        store, _ = store_with_data
        new = np.full((5, 4), 99.0, dtype=np.float32)
        store.update_feature("fp", new)
        np.testing.assert_array_equal(store.get_feature("fp"), new)

    def test_update_full_wrong_size_raises(self, store_with_data):
        store, _ = store_with_data
        with pytest.raises(ValueError):
            store.update_feature("fp", np.zeros((3, 4), dtype=np.float32))


# ===========================================================================
# Append and auto-creation
# ===========================================================================

class TestZarrFeatureStoreAppend:
    def test_append_to_existing(self, store_with_data):
        store, _ = store_with_data
        new_fp = np.full((3, 4), 7.0, dtype=np.float32)
        new_desc = np.full((3, 8), 3.0, dtype=np.float64)
        store.append({"fp": new_fp, "desc": new_desc})
        assert store.n_rows() == 8
        np.testing.assert_array_equal(store.get_feature("fp")[5:], new_fp)

    def test_append_auto_creates_array(self, tmp_zarr):
        zarr_path, _ = tmp_zarr
        store = ZarrFeatureStore(zarr_path)
        data = np.arange(6, dtype=np.float32).reshape(3, 2)
        store.append({"new_feat": data})
        assert store.n_rows() == 3
        np.testing.assert_array_equal(store.get_feature("new_feat"), data)

    def test_append_uses_chunk_rows(self, tmp_zarr):
        """Auto-created arrays use _CHUNK_ROWS along axis 0."""
        from druglab.db.backend.disk.zarr import _CHUNK_ROWS
        zarr_path, _ = tmp_zarr
        store = ZarrFeatureStore(zarr_path)
        store.append({"feat": np.zeros((5, 128), dtype=np.float32)})
        arr = zarr.open_group(str(zarr_path), mode="r")["feat"]
        assert arr.chunks[0] == _CHUNK_ROWS
        assert arr.chunks[1] == 128

    def test_append_rejects_object_dtype(self, tmp_zarr):
        zarr_path, _ = tmp_zarr
        store = ZarrFeatureStore(zarr_path)
        with pytest.raises(TypeError, match="numeric and boolean"):
            store.append({"bad": np.array(["a", "b"])})


# ===========================================================================
# Save / load round-trip
# ===========================================================================

class TestZarrFeatureStorePersistence:
    def test_temp_path_defaults_and_exists(self):
        store = ZarrFeatureStore()
        assert store._is_temp is True
        assert store.path.exists()
        assert store.path.parent == Path(tempfile.gettempdir())

    def test_save_moves_temp_store_into_bundle(self, tmp_path):
        store = ZarrFeatureStore()
        store.update_feature("fp", np.arange(12, dtype=np.float32).reshape(3, 4))
        old_path = store.path
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        store.save(bundle)
        assert not old_path.exists()
        assert store.path == bundle / "features.zarr"
        assert store.path.exists()
        np.testing.assert_array_equal(store.get_feature("fp"), np.arange(12, dtype=np.float32).reshape(3, 4))

    def test_save_load_roundtrip(self, store_with_data):
        store, tmp_path = store_with_data
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        store.save(bundle)

        loaded = ZarrFeatureStore.load(bundle)
        assert set(loaded.get_feature_names()) == {"fp", "desc"}
        np.testing.assert_array_equal(
            loaded.get_feature("fp"), store.get_feature("fp")
        )

    def test_load_missing_path_returns_empty(self, tmp_path):
        store = ZarrFeatureStore.load(tmp_path / "nonexistent_bundle")
        assert store.n_rows() == 0
        assert store.get_feature_names() == []


# ===========================================================================
# -1 virtual-index safety guardrail
# ===========================================================================

class TestVirtualIndexGuardrail:
    def test_get_feature_rejects_minus_one(self, store_with_data):
        store, _ = store_with_data
        idx = np.array([-1], dtype=np.intp)
        with pytest.raises(IndexError, match="virtual"):
            store.get_feature("fp", idx=idx)

    def test_get_feature_rejects_mixed_negative(self, store_with_data):
        store, _ = store_with_data
        idx = np.array([0, 2, -1], dtype=np.intp)
        with pytest.raises(IndexError, match="virtual"):
            store.get_feature("fp", idx=idx)

    def test_update_feature_rejects_minus_one(self, store_with_data):
        store, _ = store_with_data
        idx = np.array([-1], dtype=np.intp)
        with pytest.raises(IndexError, match="virtual"):
            store.update_feature("fp", np.zeros((1, 4), dtype=np.float32), idx=idx)

    def test_valid_positive_index_still_works(self, store_with_data):
        store, _ = store_with_data
        idx = np.array([0, 1, 4], dtype=np.intp)
        result = store.get_feature("fp", idx=idx)
        assert result.shape == (3, 4)
