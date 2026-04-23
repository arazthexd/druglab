"""
tests/unit/db/backend/test_memory.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive test suite for the refactored druglab.db in-memory storage.

Tests cover:
1. Memory Metadata — read/write API
2. Memory Objects — read/write API
3. Memory Features — read/write API
4. Cooperative save_storage_context
5. Cooperative load_storage_context (object_writer / object_reader)
6. _gather_materialized_state
7. EagerMemoryBackend.save() / load() round-trip
"""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from typing import Any, List

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
    BackendContext,
    TableContext,
    _make_metadata,
    _make_dict_objects,
    _make_features,
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


# ===========================================================================
# Section 4: save_storage_context
# ===========================================================================

class TestSaveStorageContext:
    """Unit-test each mixin's cooperative save hook in isolation."""

    def _make_backend(self, n=4):
        return EagerMemoryBackend(
            objects=[{"id": i} for i in range(n)],
            metadata=pd.DataFrame({"val": list(range(n))}),
            features={"fp": np.eye(n, dtype=np.float32)},
        )

    def test_metadata_mixin_saves_parquet(self, tmp_path):
        backend = self._make_backend()
        backend.save_storage_context(tmp_path)
        assert (tmp_path / "metadata.parquet").exists() or (
            tmp_path / "metadata.csv"
        ).exists()

    def test_feature_mixin_saves_npy(self, tmp_path):
        backend = self._make_backend()
        backend.save_storage_context(tmp_path)
        feat_dir = tmp_path / "features"
        assert feat_dir.exists()
        npy_files = list(feat_dir.glob("*.npy"))
        assert any(f.stem == "fp" for f in npy_files)

    def test_object_mixin_saves_pickle(self, tmp_path):
        backend = self._make_backend()
        backend.save_storage_context(tmp_path)
        obj_pkl = tmp_path / "objects" / "objects.pkl"
        assert obj_pkl.exists()

    def test_object_mixin_custom_writer(self, tmp_path):
        """object_writer overrides default pickle behaviour."""
        backend = self._make_backend(n=3)
        written: List[Any] = []

        def my_writer(objects, dir_path):
            written.extend(objects)
            (dir_path / "custom.bin").write_bytes(b"custom")

        backend.save_storage_context(tmp_path, object_writer=my_writer)
        assert written == [{"id": 0}, {"id": 1}, {"id": 2}]
        assert (tmp_path / "objects" / "custom.bin").exists()

    def test_mro_chain_fires_all_three_domains(self, tmp_path):
        """A single save_storage_context call should produce all three domain artefacts."""
        backend = self._make_backend()
        backend.save_storage_context(tmp_path)
        assert (tmp_path / "objects" / "objects.pkl").exists()
        assert any((tmp_path / "features").glob("*.npy"))
        assert (tmp_path / "metadata.parquet").exists() or (
            tmp_path / "metadata.csv"
        ).exists()


# ===========================================================================
# Section 5: load_storage_context
# ===========================================================================

class TestLoadStorageContext:
    """Unit-test each mixin's cooperative load hook in isolation."""

    def _make_bundle(self, tmp_path, n=4):
        backend = EagerMemoryBackend(
            objects=[{"id": i} for i in range(n)],
            metadata=pd.DataFrame({"val": list(range(n))}),
            features={"fp": np.eye(n, dtype=np.float32)},
        )
        backend.save(tmp_path)
        return tmp_path

    def test_load_roundtrip_default(self, tmp_path):
        bundle = self._make_bundle(tmp_path / "bundle")
        loaded = EagerMemoryBackend.load(bundle)
        # Objects round-trip (no serializer → raw dicts via default pickle)
        assert loaded.get_objects() == [{"id": i} for i in range(4)]

    def test_load_features_roundtrip(self, tmp_path):
        bundle = self._make_bundle(tmp_path / "bundle")
        loaded = EagerMemoryBackend.load(bundle)
        np.testing.assert_array_almost_equal(
            loaded.get_feature("fp"), np.eye(4, dtype=np.float32)
        )

    def test_load_metadata_roundtrip(self, tmp_path):
        bundle = self._make_bundle(tmp_path / "bundle")
        loaded = EagerMemoryBackend.load(bundle)
        assert loaded.get_metadata()["val"].tolist() == list(range(4))

    def test_load_with_custom_object_reader(self, tmp_path):
        """Custom object_reader completely replaces the default pickle loader."""
        bundle = self._make_bundle(tmp_path / "bundle")

        custom_objects = [{"custom": True, "row": i} for i in range(4)]

        def my_reader(dir_path: Path):
            return custom_objects

        loaded = EagerMemoryBackend.load(bundle, object_reader=my_reader)
        assert loaded.get_objects() == custom_objects

    def test_load_storage_context_classmethod_returns_dict(self, tmp_path):
        """load_storage_context should return a dict with 'objects','metadata','features'."""
        bundle = self._make_bundle(tmp_path / "bundle")
        kwargs = EagerMemoryBackend.load_storage_context(bundle)
        assert "objects" in kwargs
        assert "metadata" in kwargs
        assert "features" in kwargs

    def test_load_mmap_features(self, tmp_path):
        bundle = self._make_bundle(tmp_path / "bundle")
        loaded = EagerMemoryBackend.load(bundle, mmap_features=True)
        fp = loaded.get_feature("fp")
        assert fp.shape == (4, 4)


# ===========================================================================
# Section 6: _gather_materialized_state
# ===========================================================================

class TestGatherMaterializedState:

    def _make_backend(self, n=6):
        return EagerMemoryBackend(
            objects=[{"id": i} for i in range(n)],
            metadata=pd.DataFrame({"val": list(range(n)), "label": [f"L{i}" for i in range(n)]}),
            features={"fp": np.arange(n * 3).reshape(n, 3).astype(np.float32)},
        )

    def test_gather_full_state_no_index_map(self):
        backend = self._make_backend(4)
        result = backend._gather_materialized_state()
        assert "objects" in result
        assert "metadata" in result
        assert "features" in result
        assert len(result["objects"]) == 4
        assert result["metadata"].shape[0] == 4
        assert result["features"]["fp"].shape[0] == 4

    def test_gather_sliced_state_with_index_map(self):
        backend = self._make_backend(6)
        index_map = np.array([0, 2, 4], dtype=np.intp)
        result = backend._gather_materialized_state(index_map=index_map)
        assert len(result["objects"]) == 3
        assert result["objects"] == [{"id": 0}, {"id": 2}, {"id": 4}]
        assert result["metadata"].shape[0] == 3
        assert result["metadata"]["val"].tolist() == [0, 2, 4]
        assert result["features"]["fp"].shape == (3, 3)
        expected_fp = np.arange(6 * 3).reshape(6, 3)[index_map].astype(np.float32)
        np.testing.assert_array_equal(result["features"]["fp"], expected_fp)

    def test_gather_returns_copy_of_metadata(self):
        """Gathered metadata must not share memory with the original."""
        backend = self._make_backend(3)
        result = backend._gather_materialized_state()
        result["metadata"]["val"] = [-1, -2, -3]
        assert backend.get_metadata()["val"].tolist() != [-1, -2, -3]

    def test_gather_returns_copy_of_features(self):
        """Gathered features must not share memory with the original."""
        backend = self._make_backend(3)
        result = backend._gather_materialized_state()
        result["features"]["fp"][:] = 0
        assert not np.all(backend.get_feature("fp") == 0)


# ===========================================================================
# Section 7: EagerMemoryBackend.save() / load() round-trip
# ===========================================================================

class TestEagerMemoryRoundtrip:
    """End-to-end save/load tests using the new object_writer/object_reader API."""

    def test_save_creates_bundle_directory(self, tmp_path):
        backend = EagerMemoryBackend(objects=[1, 2, 3])
        backend.save(tmp_path / "bundle")
        assert (tmp_path / "bundle").is_dir()

    def test_save_load_objects_with_writer_reader(self, tmp_path):
        """Custom writer serialises bytes; custom reader deserialises them."""
        bundle = tmp_path / "bundle"
        objects = [{"payload": i} for i in range(5)]

        backend = EagerMemoryBackend(objects=objects)

        def writer(objs, dir_path):
            with open(dir_path / "objects.pkl", "wb") as f:
                pickle.dump(
                    {"format": "stream_v2", "count": len(objs), "serialized": True}, f
                )
                for obj in objs:
                    pickle.dump(json.dumps(obj).encode(), f)

        def reader(dir_path):
            with open(dir_path / "objects.pkl", "rb") as f:
                hdr = pickle.load(f)
                count = hdr["count"]
                return [json.loads(pickle.load(f).decode()) for _ in range(count)]

        backend.save(bundle, object_writer=writer)
        loaded = EagerMemoryBackend.load(bundle, object_reader=reader)
        assert loaded.get_objects() == objects

    def test_save_no_writer_plain_objects_round_trip(self, tmp_path):
        bundle = tmp_path / "bundle"
        objects = [{"id": i} for i in range(4)]
        backend = EagerMemoryBackend(objects=objects)
        backend.save(bundle)
        loaded = EagerMemoryBackend.load(bundle)
        assert loaded.get_objects() == objects

    def test_load_returns_eager_memory_backend(self, tmp_path):
        bundle = tmp_path / "bundle"
        EagerMemoryBackend(objects=[1, 2]).save(bundle)
        loaded = EagerMemoryBackend.load(bundle)
        assert isinstance(loaded, EagerMemoryBackend)


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])