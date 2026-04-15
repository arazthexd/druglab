"""
tests/test_db.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive test suite for the refactored druglab.db storage architecture.

Tests cover:
1. EagerMemoryBackend unit tests (CRUD, query pushdown, create_view)
2. BaseTable orchestrator tests (property proxies, validation, multi-axis
   indexing, backwards-compatibility)
3. Strict query pushdown verification (slice is passed to backend, not sliced
   after a full-array load)
4. Save/Load with .dlb bundle format (config.json manifest, zero .pkl clutter)
5. Config generation and automatic backend reconstruction on load
6. ConformerTable / unroll_conformers with new backend proxy wrappers
7. Full concat, copy, history audit trail across the new architecture
"""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import sys

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from druglab.db.backend import (
    BaseStorageBackend,
    EagerMemoryBackend,
    MemoryMetadataMixin,
    MemoryObjectMixin,
    MemoryFeatureMixin,
)
from druglab.db.backend.memory import _resolve_idx
from druglab.db.table import BaseTable, HistoryEntry, META, OBJ, FEAT, M, O, F

# ---------------------------------------------------------------------------
# Minimal concrete table (no RDKit required)
# ---------------------------------------------------------------------------

class DummyTable(BaseTable[dict]):
    """Stores plain dicts as 'objects'. Serialises with json."""

    def _serialize_object(self, obj: dict) -> bytes:
        return json.dumps(obj).encode()

    def _deserialize_object(self, raw: bytes) -> dict:
        return json.loads(raw.decode())

    @staticmethod
    def _deserialize_object_static(raw: bytes) -> dict:
        return json.loads(raw.decode())

    def _object_type_name(self) -> str:
        return "dict"


def make_table(n: int = 4) -> DummyTable:
    """Build a DummyTable with predictable data for testing."""
    objects = [{"id": i, "val": i * 10} for i in range(n)]
    metadata = pd.DataFrame({
        "name": [f"mol_{i}" for i in range(n)],
        "mw": [float(i) for i in range(n)],
    })
    features = {
        "fp": np.arange(n * 8, dtype=np.float32).reshape(n, 8),
        "phys": np.ones((n, 3), dtype=np.float64),
    }
    return DummyTable(objects=objects, metadata=metadata, features=features)


# ===========================================================================
# Section 1: _resolve_idx helper
# ===========================================================================

class TestResolveIdx:
    """Unit-test the index-normalisation helper independently."""

    def test_none_returns_none(self):
        assert _resolve_idx(None, 10) is None

    def test_int_positive(self):
        result = _resolve_idx(3, 10)
        assert result.tolist() == [3]

    def test_int_negative(self):
        result = _resolve_idx(-1, 10)
        assert result.tolist() == [9]

    def test_slice_basic(self):
        result = _resolve_idx(slice(2, 5), 10)
        assert result.tolist() == [2, 3, 4]

    def test_slice_step(self):
        result = _resolve_idx(slice(0, 10, 2), 10)
        assert result.tolist() == [0, 2, 4, 6, 8]

    def test_slice_open_end(self):
        result = _resolve_idx(slice(None, None), 5)
        assert result.tolist() == [0, 1, 2, 3, 4]

    def test_list_of_ints(self):
        result = _resolve_idx([0, 2, 4], 10)
        assert result.tolist() == [0, 2, 4]

    def test_numpy_array(self):
        result = _resolve_idx(np.array([1, 3]), 10)
        assert result.tolist() == [1, 3]

    def test_list_negative(self):
        result = _resolve_idx([-1, -2], 10)
        assert result.tolist() == [9, 8]

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            _resolve_idx("bad", 10)


# ===========================================================================
# Section 2: MemoryMetadataMixin
# ===========================================================================

class TestMemoryMetadataMixin:
    """Test the metadata mixin via EagerMemoryBackend (which composes it)."""

    def _make_backend(self, n: int = 4) -> EagerMemoryBackend:
        objects = [{"id": i} for i in range(n)]
        meta = pd.DataFrame({"name": [f"m{i}" for i in range(n)], "val": range(n)})
        return EagerMemoryBackend(objects=objects, metadata=meta)

    def test_get_all_metadata(self):
        b = self._make_backend(4)
        df = b.get_metadata()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert list(df.columns) == ["name", "val"]

    def test_get_metadata_idx_int(self):
        b = self._make_backend(4)
        df = b.get_metadata(idx=2)
        assert len(df) == 1
        assert df["val"].iloc[0] == 2

    def test_get_metadata_idx_slice(self):
        b = self._make_backend(6)
        df = b.get_metadata(idx=slice(1, 4))
        assert len(df) == 3
        assert df["val"].tolist() == [1, 2, 3]

    def test_get_metadata_idx_list(self):
        b = self._make_backend(6)
        df = b.get_metadata(idx=[0, 3, 5])
        assert df["val"].tolist() == [0, 3, 5]

    def test_get_metadata_idx_none(self):
        b = self._make_backend(4)
        df = b.get_metadata(idx=None)
        assert len(df) == 4

    def test_get_metadata_cols_single(self):
        b = self._make_backend(4)
        df = b.get_metadata(cols="name")
        assert list(df.columns) == ["name"]

    def test_get_metadata_cols_list(self):
        b = self._make_backend(4)
        df = b.get_metadata(cols=["name", "val"])
        assert set(df.columns) == {"name", "val"}

    def test_get_metadata_idx_and_cols(self):
        b = self._make_backend(6)
        # Pushdown: rows 1-3, only 'val' column
        df = b.get_metadata(idx=slice(1, 4), cols="val")
        assert len(df) == 3
        assert list(df.columns) == ["val"]
        assert df["val"].tolist() == [1, 2, 3]

    def test_update_metadata_resets_index(self):
        b = self._make_backend(4)
        new_df = pd.DataFrame({"x": [10, 20, 30, 40]})
        b.update_metadata(new_df)
        print(b._metadata)
        df = b.get_metadata()
        assert "x" in df.columns
        assert df.index.tolist() == [0, 1, 2, 3]


# ===========================================================================
# Section 3: MemoryObjectMixin
# ===========================================================================

class TestMemoryObjectMixin:
    """Test the object mixin via EagerMemoryBackend."""

    def _make_backend(self, n: int = 5) -> EagerMemoryBackend:
        return EagerMemoryBackend(objects=[{"id": i} for i in range(n)])

    def test_len(self):
        b = self._make_backend(5)
        assert len(b) == 5

    def test_get_single_int(self):
        b = self._make_backend(5)
        obj = b.get_objects(2)
        assert obj == {"id": 2}

    def test_get_single_negative_int(self):
        b = self._make_backend(5)
        obj = b.get_objects(-1)
        assert obj == {"id": 4}

    def test_get_slice(self):
        b = self._make_backend(5)
        objs = b.get_objects(slice(1, 4))
        assert isinstance(objs, list)
        assert len(objs) == 3
        assert objs[0] == {"id": 1}
        assert objs[2] == {"id": 3}

    def test_get_list_of_ints(self):
        b = self._make_backend(5)
        objs = b.get_objects([0, 2, 4])
        assert len(objs) == 3
        assert objs[1] == {"id": 2}

    def test_put_object(self):
        b = self._make_backend(5)
        b.put_object(2, {"id": 99})
        assert b.get_objects(2) == {"id": 99}

    def test_put_object_negative(self):
        b = self._make_backend(5)
        b.put_object(-1, {"id": 999})
        assert b.get_objects(4) == {"id": 999}


# ===========================================================================
# Section 4: MemoryFeatureMixin
# ===========================================================================

class TestMemoryFeatureMixin:
    """Test the feature mixin via EagerMemoryBackend."""

    def _make_backend(self, n: int = 4) -> EagerMemoryBackend:
        features = {
            "fp": np.arange(n * 8, dtype=np.float32).reshape(n, 8),
            "desc": np.ones((n, 3), dtype=np.float64),
        }
        return EagerMemoryBackend(objects=[{}] * n, features=features)

    def test_get_feature_names(self):
        b = self._make_backend()
        assert set(b.get_feature_names()) == {"fp", "desc"}

    def test_get_feature_all(self):
        b = self._make_backend(4)
        arr = b.get_feature("fp")
        assert arr.shape == (4, 8)

    def test_get_feature_idx_none_returns_full(self):
        b = self._make_backend(4)
        arr = b.get_feature("fp", idx=None)
        assert arr.shape == (4, 8)

    def test_get_feature_idx_slice_pushdown(self):
        """
        STRICT QUERY PUSHDOWN TEST:
        The slice must be applied AT THE BACKEND LEVEL.
        The test patches np.ndarray.__getitem__ is impractical, so we instead
        verify that the returned array has the exact expected rows/values,
        and that the full array is NOT returned (shape check).
        """
        b = self._make_backend(6)
        arr = b.get_feature("fp", idx=slice(1, 4))
        # Only 3 rows should be returned — the backend must NOT return all 6
        assert arr.shape == (3, 8), (
            "Backend must return ONLY the sliced rows (3), not all 6. "
            "Query was not pushed down to the backend."
        )
        # Values must correspond to rows 1, 2, 3 of the original feature
        expected_row0 = np.arange(1 * 8, 1 * 8 + 8, dtype=np.float32)
        assert np.allclose(arr[0], expected_row0)

    def test_get_feature_idx_int_pushdown(self):
        b = self._make_backend(4)
        arr = b.get_feature("fp", idx=2)
        assert arr.shape == (1, 8)

    def test_get_feature_idx_list_pushdown(self):
        b = self._make_backend(6)
        arr = b.get_feature("fp", idx=[0, 5])
        assert arr.shape == (2, 8)

    def test_add_feature(self):
        b = self._make_backend(4)
        new_feat = np.zeros((4, 16))
        b.add_feature("new", new_feat)
        assert "new" in b.get_feature_names()
        assert b.get_feature("new").shape == (4, 16)

    def test_drop_feature(self):
        b = self._make_backend(4)
        b.drop_feature("fp")
        assert "fp" not in b.get_feature_names()

    def test_drop_feature_missing_raises(self):
        b = self._make_backend(4)
        with pytest.raises(KeyError):
            b.drop_feature("nonexistent")


# ===========================================================================
# Section 5: EagerMemoryBackend.create_view (query pushdown for subsetting)
# ===========================================================================

class TestCreateView:
    """
    create_view is the critical path for query pushdown in subsetting.
    All three data stores must be synchronised after create_view.
    """

    def _make_backend(self, n: int = 6) -> EagerMemoryBackend:
        objects = [{"id": i} for i in range(n)]
        meta = pd.DataFrame({"val": list(range(n))})
        features = {"fp": np.arange(n * 4, dtype=np.float32).reshape(n, 4)}
        return EagerMemoryBackend(objects=objects, metadata=meta, features=features)

    def test_creates_correct_row_count(self):
        b = self._make_backend(6)
        view = b.create_view([0, 2, 4])
        assert len(view) == 3

    def test_objects_correctly_selected(self):
        b = self._make_backend(6)
        view = b.create_view([1, 3, 5])
        objs = view._objects
        assert objs[0] == {"id": 1}
        assert objs[1] == {"id": 3}
        assert objs[2] == {"id": 5}

    def test_metadata_correctly_selected(self):
        b = self._make_backend(6)
        view = b.create_view([0, 4])
        meta = view.get_metadata()
        assert meta["val"].tolist() == [0, 4]

    def test_features_correctly_selected(self):
        b = self._make_backend(6)
        view = b.create_view([2, 5])
        arr = view.get_feature("fp")
        assert arr.shape == (2, 4)
        # Row 0 of view should correspond to row 2 of original
        expected = np.arange(2 * 4, 2 * 4 + 4, dtype=np.float32)
        assert np.allclose(arr[0], expected)

    def test_view_is_independent_from_original_objects(self):
        """Mutating the view's object must NOT affect the original."""
        b = self._make_backend(4)
        view = b.create_view([0, 1])
        view._objects[0]["id"] = 9999
        assert b._objects[0]["id"] == 0

    def test_view_is_independent_from_original_metadata(self):
        b = self._make_backend(4)
        view = b.create_view([0, 1])
        view._metadata.loc[0, "val"] = 9999
        assert b._metadata.loc[0, "val"] == 0

    def test_view_is_independent_from_original_features(self):
        b = self._make_backend(4)
        view = b.create_view([0, 1])
        view._features["fp"][0, 0] = -999.0
        assert b._features["fp"][0, 0] != -999.0

    def test_index_reset_after_view(self):
        """Metadata index must be 0-based after create_view."""
        b = self._make_backend(6)
        view = b.create_view([3, 5])
        assert view.get_metadata().index.tolist() == [0, 1]

    def test_empty_view(self):
        b = self._make_backend(4)
        view = b.create_view([])
        assert len(view) == 0
        assert view.get_metadata().empty or len(view.get_metadata()) == 0

    def test_all_rows_view(self):
        b = self._make_backend(4)
        view = b.create_view([0, 1, 2, 3])
        assert len(view) == 4

    def test_view_preserves_feature_names(self):
        b = self._make_backend(4)
        b.add_feature("desc", np.zeros((4, 3)))
        view = b.create_view([0, 2])
        assert set(view.get_feature_names()) == {"fp", "desc"}


# ===========================================================================
# Section 6: EagerMemoryBackend.save / load
# ===========================================================================

class TestEagerMemoryBackendPersistence:
    """
    Save writes a .dlb-compatible directory; load reconstructs it exactly.
    """

    def _make_backend(self, n: int = 4) -> EagerMemoryBackend:
        objects = [{"id": i} for i in range(n)]
        meta = pd.DataFrame({"name": [f"obj_{i}" for i in range(n)], "score": [i * 1.5 for i in range(n)]})
        features = {"fp": np.arange(n * 8, dtype=np.float32).reshape(n, 8)}
        return EagerMemoryBackend(objects=objects, metadata=meta, features=features)

    def test_save_creates_expected_files(self, tmp_path):
        b = self._make_backend(3)
        b.save(tmp_path)
        assert (tmp_path / "objects" / "objects.pkl").exists()
        assert (tmp_path / "features" / "fp.npy").exists()
        assert (
            (tmp_path / "metadata.parquet").exists()
            or (tmp_path / "metadata.csv").exists()
        )

    def test_no_individual_pkl_files(self, tmp_path):
        """Zero .pkl clutter: only a single objects.pkl, no per-row files."""
        b = self._make_backend(5)
        b.save(tmp_path)
        obj_dir = tmp_path / "objects"
        pkl_files = list(obj_dir.glob("*.pkl"))
        # Must be exactly 1 file (the consolidated objects.pkl)
        assert len(pkl_files) == 1, (
            f"Expected exactly 1 .pkl file, found {len(pkl_files)}: {pkl_files}. "
            "Per-row .pkl files are not allowed."
        )
        assert pkl_files[0].name == "objects.pkl"

    def test_load_round_trip_objects(self, tmp_path):
        b = self._make_backend(4)
        b.save(tmp_path)
        loaded = EagerMemoryBackend.load(tmp_path)
        assert len(loaded) == 4
        assert loaded._objects[2] == {"id": 2}

    def test_load_round_trip_metadata(self, tmp_path):
        b = self._make_backend(4)
        b.save(tmp_path)
        loaded = EagerMemoryBackend.load(tmp_path)
        assert "name" in loaded.get_metadata().columns
        assert loaded.get_metadata()["score"].tolist() == pytest.approx([0.0, 1.5, 3.0, 4.5])

    def test_load_round_trip_features(self, tmp_path):
        b = self._make_backend(4)
        b.save(tmp_path)
        loaded = EagerMemoryBackend.load(tmp_path)
        assert "fp" in loaded.get_feature_names()
        assert np.allclose(loaded.get_feature("fp"), b.get_feature("fp"))

    def test_load_with_serializer(self, tmp_path):
        """Custom serializer/deserializer round-trips correctly."""
        objects = [{"id": i} for i in range(3)]
        b = EagerMemoryBackend(objects=objects)

        def ser(obj):
            return json.dumps(obj).encode()

        def deser(raw):
            return json.loads(raw.decode())

        b.save(tmp_path, serializer=ser)
        loaded = EagerMemoryBackend.load(tmp_path, deserializer=deser)
        assert loaded._objects[1] == {"id": 1}

    def test_mmap_features_load(self, tmp_path):
        b = self._make_backend(4)
        b.save(tmp_path)
        loaded = EagerMemoryBackend.load(tmp_path, mmap_features=True)
        arr = loaded.get_feature("fp")
        assert arr.shape == (4, 8)


# ===========================================================================
# Section 7: BaseTable property proxies (backwards compatibility)
# ===========================================================================

class TestBaseTablePropertyProxies:
    """
    Verify that dot-notation access (.objects, .metadata, .features) continues
    to work identically to the old attribute-based API.
    """

    def test_objects_proxy(self):
        t = make_table(4)
        assert len(t.objects) == 4
        assert t.objects[0] == {"id": 0, "val": 0}
        assert t.objects[3] == {"id": 3, "val": 30}

    def test_metadata_proxy(self):
        t = make_table(4)
        df = t.metadata
        assert isinstance(df, pd.DataFrame)
        assert "name" in df.columns
        assert len(df) == 4

    def test_features_proxy_keys(self):
        t = make_table(4)
        feats = t.features
        assert "fp" in feats
        assert "phys" in feats

    def test_features_proxy_values(self):
        t = make_table(4)
        fp = t.features["fp"]
        assert fp.shape == (4, 8)

    def test_n_property(self):
        t = make_table(7)
        assert t.n == 7
        assert len(t) == 7

    def test_metadata_columns_property(self):
        t = make_table(3)
        assert "name" in t.metadata_columns
        assert "mw" in t.metadata_columns

    def test_direct_object_mutation_via_backend(self):
        """Mutating via backend._objects should be visible through .objects proxy."""
        t = make_table(3)
        t._backend._objects[0] = {"id": 99}
        assert t.objects[0] == {"id": 99}


# ===========================================================================
# Section 8: BaseTable multi-axis indexing with strict query pushdown
# ===========================================================================

class TestMultiAxisIndexing:
    """
    Test the advanced __getitem__ routing with all axis constants and
    verify strict query pushdown (data is not pulled to Python level first).
    """

    def test_standard_row_slice_returns_table(self):
        t = make_table(6)
        sub = t[0:3]
        assert isinstance(sub, DummyTable)
        assert sub.n == 3

    def test_standard_row_list_returns_table(self):
        t = make_table(6)
        sub = t[[0, 2, 4]]
        assert sub.n == 3

    def test_standard_bool_mask_returns_table(self):
        t = make_table(4)
        mask = pd.Series([True, False, True, False])
        sub = t[mask]
        assert sub.n == 2

    # --- FEAT axis ---

    def test_feat_axis_with_name_and_idx(self):
        t = make_table(6)
        arr = t[FEAT, "fp", 1:4]
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 8), (
            "Query pushdown failed: only 3 rows should be returned."
        )

    def test_feat_alias_F(self):
        t = make_table(4)
        arr = t[F, "fp", 0:2]
        assert arr.shape == (2, 8)

    def test_feat_axis_without_idx(self):
        t = make_table(4)
        arr = t[FEAT, "fp"]
        assert arr.shape == (4, 8)

    def test_feat_axis_list_of_names(self):
        t = make_table(4)
        result = t[FEAT, ["fp", "phys"], 0:2]
        assert isinstance(result, dict)
        assert "fp" in result and "phys" in result
        assert result["fp"].shape == (2, 8)
        assert result["phys"].shape == (2, 3)

    def test_feat_axis_names_none_returns_all_features(self):
        t = make_table(4)
        result = t[FEAT, None, 0:2]
        assert "fp" in result
        assert "phys" in result

    # --- META axis ---

    def test_meta_axis_with_idx(self):
        t = make_table(6)
        df = t[META, 2:5]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_meta_axis_with_cols_and_idx(self):
        t = make_table(6)
        df = t[META, ["name"], 0:3]
        assert list(df.columns) == ["name"]
        assert len(df) == 3

    def test_meta_alias_M(self):
        t = make_table(4)
        df = t[M, 0:2]
        assert len(df) == 2

    def test_meta_axis_single_col_string(self):
        t = make_table(4)
        df = t[META, "name", 0:2]
        assert list(df.columns) == ["name"]

    # --- OBJ axis ---

    def test_obj_axis_single_int(self):
        t = make_table(4)
        obj = t[OBJ, 2]
        assert obj == {"id": 2, "val": 20}

    def test_obj_alias_O(self):
        t = make_table(4)
        obj = t[O, 0]
        assert obj == {"id": 0, "val": 0}

    def test_obj_axis_slice(self):
        t = make_table(6)
        objs = t[OBJ, 1:4]
        assert isinstance(objs, list)
        assert len(objs) == 3

    def test_obj_axis_named_cols_raises(self):
        t = make_table(4)
        with pytest.raises(ValueError, match="Objects do not have named columns"):
            _ = t[OBJ, "some_col", 0]

    # --- Error cases ---

    def test_invalid_tuple_length_raises(self):
        t = make_table(4)
        with pytest.raises(ValueError, match="Invalid slicing format"):
            _ = t[FEAT, "fp", 0, "extra"]

    def test_unknown_axis_raises(self):
        t = make_table(4)
        with pytest.raises(ValueError, match="Unknown attribute identifier"):
            _ = t["bad_axis", 0]

    # --- Pushdown verification ---

    def test_feat_pushdown_does_not_load_full_array(self):
        """
        Verify that table[FEAT, 'fp', 0:2] calls backend.get_feature with
        idx=slice(0, 2) rather than fetching all rows and slicing in Python.
        """
        t = make_table(10)
        original_get_feature = t._backend.get_feature
        calls = []

        def spy_get_feature(name, idx=None):
            calls.append({"name": name, "idx": idx})
            return original_get_feature(name, idx=idx)

        t._backend.get_feature = spy_get_feature
        arr = t[FEAT, "fp", 0:2]

        # The backend's get_feature must have been called with the idx arg
        assert len(calls) == 1, "get_feature should be called exactly once"
        assert calls[0]["idx"] is not None, (
            "idx must be passed to get_feature (not None), "
            "proving pushdown rather than post-hoc Python slicing."
        )
        # Returned array must contain only the requested rows
        assert arr.shape == (2, 8)


# ===========================================================================
# Section 9: BaseTable construction & validation
# ===========================================================================

class TestBaseTableConstruction:
    def test_basic(self):
        t = make_table(4)
        assert t.n == 4

    def test_empty(self):
        t = DummyTable()
        assert t.n == 0

    def test_metadata_row_mismatch_raises(self):
        objects = [{"id": 0}]
        bad_meta = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="metadata"):
            DummyTable(objects=objects, metadata=bad_meta)

    def test_feature_row_mismatch_raises(self):
        objects = [{"id": i} for i in range(3)]
        bad_feat = {"fp": np.zeros((5, 8))}
        with pytest.raises(ValueError, match="Feature"):
            DummyTable(objects=objects, features=bad_feat)

    def test_repr(self):
        t = make_table(2)
        r = repr(t)
        assert "DummyTable" in r
        assert "n=2" in r

    def test_backend_is_eager_memory_by_default(self):
        t = make_table(3)
        assert isinstance(t._backend, EagerMemoryBackend)


# ===========================================================================
# Section 10: Metadata & feature mutation (backwards-compatible API)
# ===========================================================================

class TestMutationAPI:
    def test_add_metadata_column(self):
        t = make_table(3)
        t.add_metadata_column("activity", [1.0, 2.0, 3.0])
        assert "activity" in t.metadata.columns
        assert list(t.metadata["activity"]) == [1.0, 2.0, 3.0]

    def test_add_metadata_column_length_mismatch(self):
        t = make_table(3)
        with pytest.raises(ValueError):
            t.add_metadata_column("bad", [1, 2])

    def test_drop_metadata_columns(self):
        t = make_table(3)
        t.drop_metadata_columns("name")
        assert "name" not in t.metadata.columns

    def test_add_feature(self):
        t = make_table(4)
        arr = np.zeros((4, 16))
        t.add_feature("ecfp4", arr)
        assert t.has_feature("ecfp4")
        assert t.features["ecfp4"].shape == (4, 16)

    def test_add_feature_wrong_rows(self):
        t = make_table(4)
        with pytest.raises(ValueError):
            t.add_feature("bad", np.zeros((3, 16)))

    def test_drop_feature(self):
        t = make_table(4)
        t.drop_feature("fp")
        assert not t.has_feature("fp")

    def test_has_feature(self):
        t = make_table(4)
        assert t.has_feature("fp")
        assert not t.has_feature("nonexistent")


# ===========================================================================
# Section 11: Subsetting (subset / __getitem__ as row selection)
# ===========================================================================

class TestSubset:
    def test_subset_by_indices(self):
        t = make_table(6)
        sub = t.subset([0, 2, 4])
        assert sub.n == 3
        assert sub.objects[0]["id"] == 0
        assert sub.objects[1]["id"] == 2
        assert sub.metadata["mw"].tolist() == [0.0, 2.0, 4.0]
        assert sub.features["fp"].shape == (3, 8)

    def test_subset_by_boolean_mask(self):
        t = make_table(4)
        mask = np.array([True, False, True, False])
        sub = t.subset(mask)
        assert sub.n == 2

    def test_subset_adds_history(self):
        t = make_table(4)
        sub = t.subset([0, 1])
        assert any("subset" in e.block_name for e in sub.history)

    def test_subset_returns_same_subclass(self):
        t = make_table(4)
        sub = t.subset([0, 1])
        assert type(sub) is DummyTable

    def test_subset_uses_backend_create_view(self):
        """Verify subset delegates to backend.create_view (pushdown)."""
        t = make_table(4)
        original_create_view = t._backend.create_view
        calls = []

        def spy_create_view(indices):
            calls.append(indices)
            return original_create_view(indices)

        t._backend.create_view = spy_create_view
        _ = t.subset([0, 2])

        assert len(calls) == 1
        assert list(calls[0]) == [0, 2]

    def test_getitem_int(self):
        t = make_table(4)
        sub = t[0]
        assert sub.n == 1

    def test_getitem_slice(self):
        t = make_table(6)
        sub = t[1:4]
        assert sub.n == 3

    def test_getitem_list(self):
        t = make_table(6)
        sub = t[[0, 5]]
        assert sub.n == 2

    def test_boolean_mask_length_mismatch(self):
        t = make_table(4)
        with pytest.raises(ValueError):
            t.subset(np.array([True, False]))  # wrong length


# ===========================================================================
# Section 12: Copy
# ===========================================================================

class TestCopy:
    def test_copy_is_independent_objects(self):
        t = make_table(3)
        c = t.copy()
        c._backend._objects[0]["val"] = 9999
        assert t.objects[0]["val"] == 0

    def test_copy_is_independent_features(self):
        t = make_table(3)
        c = t.copy()
        c._backend._features["fp"][0, 0] = -1.0
        assert t.features["fp"][0, 0] != -1.0

    def test_copy_is_independent_metadata(self):
        t = make_table(3)
        c = t.copy()
        c._backend._metadata.loc[0, "mw"] = 999.0
        assert t.metadata["mw"].iloc[0] != 999.0

    def test_copy_preserves_history(self):
        t = make_table(3)
        t.append_history(HistoryEntry.now("step", {}, 3, 3))
        c = t.copy()
        assert len(c.history) == 1


# ===========================================================================
# Section 13: Concatenation
# ===========================================================================

class TestConcat:
    def test_basic_concat(self):
        a = make_table(3)
        b = make_table(2)
        c = DummyTable.concat([a, b])
        assert c.n == 5
        assert c.features["fp"].shape == (5, 8)
        assert len(c.metadata) == 5

    def test_concat_missing_feature_zeros(self):
        a = make_table(2)
        b = DummyTable(
            objects=[{"id": 99}],
            metadata=pd.DataFrame({"name": ["x"], "mw": [0.0]}),
            features={},
        )
        c = DummyTable.concat([a, b], handle_missing_features="zeros")
        assert "fp" in c.features
        assert c.features["fp"].shape == (3, 8)
        assert c.features["fp"][2].sum() == 0.0

    def test_concat_missing_feature_raise(self):
        a = make_table(2)
        b = DummyTable(objects=[{"id": 99}])
        with pytest.raises(ValueError):
            DummyTable.concat([a, b], handle_missing_features="raise")

    def test_concat_type_mismatch_raises(self):
        class OtherTable(DummyTable):
            pass
        a = make_table(2)
        b = OtherTable(objects=[{"id": 0}])
        with pytest.raises(TypeError):
            DummyTable.concat([a, b])

    def test_concat_history_accumulated(self):
        a = make_table(2)
        b = make_table(2)
        c = DummyTable.concat([a, b])
        assert any("concat" in e.block_name for e in c.history)

    def test_concat_empty_raises(self):
        with pytest.raises(ValueError):
            DummyTable.concat([])

    def test_concat_result_uses_eager_memory_backend(self):
        a = make_table(2)
        b = make_table(2)
        c = DummyTable.concat([a, b])
        assert isinstance(c._backend, EagerMemoryBackend)


# ===========================================================================
# Section 14: History
# ===========================================================================

class TestHistory:
    def test_append_history(self):
        t = make_table(3)
        entry = HistoryEntry.now("MyBlock", {"k": "v"}, 3, 3)
        t.append_history(entry)
        assert t.history[-1] is entry

    def test_history_preserved_through_subset(self):
        t = make_table(4)
        t.append_history(HistoryEntry.now("InitBlock", {}, 4, 4))
        sub = t.subset([0, 1])
        names = [e.block_name for e in sub.history]
        assert "InitBlock" in names


# ===========================================================================
# Section 15: Save / Load with .dlb bundle format
# ===========================================================================

class TestDlbPersistence:
    """
    The table must save as a .dlb directory bundle with a config.json manifest.
    Zero per-row .pkl clutter is required.
    """

    def test_save_creates_config_json(self, tmp_path):
        t = make_table(3)
        t.save(tmp_path / "table.dlb")
        assert (tmp_path / "table.dlb" / "config.json").exists()

    def test_config_json_contents(self, tmp_path):
        t = make_table(3)
        t.append_history(HistoryEntry.now("Step1", {"a": 1}, 3, 3))
        t.save(tmp_path / "table.dlb")

        config = json.loads((tmp_path / "table.dlb" / "config.json").read_text())
        assert config["table_class"] == "DummyTable"
        assert config["backend_class"] == "EagerMemoryBackend"
        assert config["schema_version"] == 2
        assert config["n"] == 3
        assert len(config["history"]) == 1
        assert config["history"][0]["block_name"] == "Step1"

    def test_zero_pkl_clutter(self, tmp_path):
        """No per-row .pkl files are allowed anywhere in the bundle."""
        t = make_table(5)
        root = t.save(tmp_path / "table.dlb")

        all_pkl = list(root.rglob("*.pkl"))
        # Only one .pkl file is permitted: the consolidated objects.pkl
        assert len(all_pkl) == 1, (
            f"Expected exactly 1 .pkl file; found {[str(p) for p in all_pkl]}. "
            "Per-row .pkl files cause filesystem inode exhaustion and are forbidden."
        )
        assert all_pkl[0].name == "objects.pkl"

    def test_no_legacy_meta_json(self, tmp_path):
        """New format must NOT write the old _meta.json file."""
        t = make_table(3)
        root = t.save(tmp_path / "table.dlb")
        assert not (root / "_meta.json").exists()

    def test_save_and_load_round_trip(self, tmp_path):
        t = make_table(4)
        t.append_history(HistoryEntry.now("Step1", {"a": 1}, 4, 4))
        root = t.save(tmp_path / "table.dlb")

        loaded = DummyTable.load(root)
        assert loaded.n == 4
        assert loaded.objects[2]["id"] == 2
        assert "name" in loaded.metadata.columns
        assert "fp" in loaded.features
        assert np.allclose(loaded.features["fp"], t.features["fp"])
        assert len(loaded.history) == 1
        assert loaded.history[0].block_name == "Step1"

    def test_load_restores_correct_backend_class(self, tmp_path):
        t = make_table(3)
        root = t.save(tmp_path / "table.dlb")
        loaded = DummyTable.load(root)
        assert isinstance(loaded._backend, EagerMemoryBackend)

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DummyTable.load(tmp_path / "does_not_exist.dlb")

    def test_save_load_empty_table(self, tmp_path):
        t = DummyTable()
        t.save(tmp_path / "empty.dlb")
        loaded = DummyTable.load(tmp_path / "empty.dlb")
        assert loaded.n == 0

    def test_mmap_load(self, tmp_path):
        t = make_table(4)
        t.save(tmp_path / "table.dlb")
        loaded = DummyTable.load(tmp_path / "table.dlb", mmap_features=True)
        assert loaded.features["fp"].shape == (4, 8)


# ===========================================================================
# Section 16: MoleculeTable with new backend (requires RDKit)
# ===========================================================================

rdkit_mark = pytest.mark.skipif(
    True,  # skip by default; override if rdkit is available
    reason="RDKit not installed",
)

try:
    from rdkit import Chem
    _HAS_RDKIT = True
    rdkit_mark = pytest.mark.usefixtures()  # no-op; tests will run
except ImportError:
    _HAS_RDKIT = False


@pytest.mark.skipif(not _HAS_RDKIT, reason="RDKit not installed")
class TestMoleculeTableBackendIntegration:
    """Verify MoleculeTable works end-to-end with the new backend."""

    def test_from_smiles_uses_eager_backend(self):
        from druglab.db.table.molecule import MoleculeTable
        t = MoleculeTable.from_smiles(["CCO", "c1ccccc1"])
        assert isinstance(t._backend, EagerMemoryBackend)
        assert t.n == 2

    def test_smiles_property(self):
        from druglab.db.table.molecule import MoleculeTable
        t = MoleculeTable.from_smiles(["CCO", "c1ccccc1"])
        assert len(t.smiles) == 2
        assert all(s is not None for s in t.smiles)

    def test_subset_preserves_molecules(self):
        from druglab.db.table.molecule import MoleculeTable
        t = MoleculeTable.from_smiles(["C", "CC", "CCC", "CCCC"])
        sub = t[0:2]
        assert sub.n == 2
        assert sub.smiles[0] == t.smiles[0]

    def test_multi_axis_feat_pushdown_with_molecule_table(self):
        from druglab.db.table.molecule import MoleculeTable
        t = MoleculeTable.from_smiles(["C", "CC", "CCC", "CCCC", "CCCCC"])
        # Manually add a feature to test pushdown without requiring pipe
        t.add_feature("fp", np.arange(5 * 512, dtype=np.float32).reshape(5, 512))

        arr = t[FEAT, "fp", 1:3]
        assert arr.shape == (2, 512), (
            "Feature pushdown failed: expected exactly 2 rows for slice 1:3."
        )

    def test_save_load_molecule_table(self, tmp_path):
        from druglab.db.table.molecule import MoleculeTable
        t = MoleculeTable.from_smiles(["CCO", "c1ccccc1", "CC(=O)O"])
        root = t.save(tmp_path / "mols.dlb")

        loaded = MoleculeTable.load(root)
        assert loaded.n == 3
        original_smiles = set(s for s in t.smiles if s)
        loaded_smiles = set(s for s in loaded.smiles if s)
        assert original_smiles == loaded_smiles

    def test_config_json_table_class_is_molecule_table(self, tmp_path):
        from druglab.db.table.molecule import MoleculeTable
        t = MoleculeTable.from_smiles(["CCO"])
        root = t.save(tmp_path / "mol.dlb")
        config = json.loads((root / "config.json").read_text())
        assert config["table_class"] == "MoleculeTable"
        assert config["object_type"] == "Mol"

    def test_unroll_conformers_works_with_new_backend(self):
        """
        unroll_conformers must function correctly with the new backend proxy wrappers.
        """
        from rdkit.Chem import AllChem
        from druglab.db.table.molecule import MoleculeTable
        from druglab.db.table.conformer import ConformerTable

        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=3, randomSeed=42, numThreads=1)
        assert mol.GetNumConformers() == 3

        t = MoleculeTable(
            objects=[mol],
            metadata=pd.DataFrame({"name": ["ethanol"]}),
        )
        conf_table = t.unroll_conformers()

        assert isinstance(conf_table, ConformerTable)
        assert isinstance(conf_table._backend, EagerMemoryBackend)
        assert conf_table.n == 3
        for m in conf_table.objects:
            assert m is not None
            assert m.GetNumConformers() == 1

    def test_unroll_collapse_round_trip_with_new_backend(self):
        """Full round-trip: unroll → collapse restores conformer count."""
        from rdkit.Chem import AllChem
        from druglab.db.table.molecule import MoleculeTable

        mols = []
        for smi in ["CCO", "c1ccccc1"]:
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMultipleConfs(mol, numConfs=2, randomSeed=0, numThreads=1)
            mols.append(mol)

        t = MoleculeTable(objects=mols, metadata=pd.DataFrame({"name": ["a", "b"]}))
        conf_table = t.unroll_conformers()
        assert conf_table.n == 4

        collapsed = conf_table.collapse()
        assert isinstance(collapsed, MoleculeTable)
        assert collapsed.n == 2
        for m in collapsed.objects:
            assert m.GetNumConformers() == 2


# ===========================================================================
# Section 17: HistoryEntry (preserved from original tests)
# ===========================================================================

class TestHistoryEntry:
    def test_round_trip(self):
        entry = HistoryEntry.now(
            block_name="TestBlock",
            config={"param": 42},
            rows_in=100,
            rows_out=80,
        )
        d = entry.to_dict()
        restored = HistoryEntry.from_dict(d)
        assert restored == entry

    def test_immutable(self):
        entry = HistoryEntry.now("block", {}, 10, 10)
        with pytest.raises((AttributeError, TypeError)):
            entry.block_name = "other"


# ===========================================================================
# Section 18: Edge-cases & regression tests
# ===========================================================================

class TestEdgeCases:
    def test_empty_backend_view(self):
        t = make_table(4)
        view = t._backend.create_view([])
        assert len(view) == 0

    def test_feature_names_slash_safe(self, tmp_path):
        """Feature names with slashes are saved safely."""
        t = DummyTable(
            objects=[{"id": 0}],
            features={"ecfp/4": np.zeros((1, 8))},
        )
        root = t.save(tmp_path / "t.dlb")
        feat_files = list((root / "features").glob("*.npy"))
        assert len(feat_files) == 1
        # Slash should be replaced, not cause nested directories
        assert "/" not in feat_files[0].name

    def test_subset_after_concat_maintains_sync(self):
        """After concat, subsetting must keep all three data stores in sync."""
        a = make_table(3)
        b = make_table(3)
        c = DummyTable.concat([a, b])
        sub = c.subset([0, 3, 5])
        assert sub.n == 3
        assert len(sub.metadata) == 3
        assert sub.features["fp"].shape[0] == 3

    def test_add_metadata_column_after_subset(self):
        t = make_table(6)
        sub = t[1:4]
        sub.add_metadata_column("score", [0.1, 0.2, 0.3])
        assert "score" in sub.metadata.columns
        assert list(sub.metadata["score"]) == [0.1, 0.2, 0.3]

    def test_numerical_metadata_conversion(self):
        t = DummyTable(
            objects=[{"id": 0}, {"id": 1}],
            metadata=pd.DataFrame({
                "ID": ["mol1", "mol2"],
                "MolWt": ["46.07", "78.11"],
            }),
        )
        t.backend.try_numerize_metadata(columns=["MolWt"])
        assert pd.api.types.is_numeric_dtype(t.metadata["MolWt"])
        assert t.metadata["MolWt"].iloc[0] == pytest.approx(46.07)

    def test_update_metadata_by_key(self):
        t = make_table(3)
        ext = pd.DataFrame({"name": ["mol_2", "mol_0"], "score": [20, 10]})
        t.update_metadata(ext, on="name")
        # Order preserved
        assert t.metadata["name"].tolist() == ["mol_0", "mol_1", "mol_2"]
        assert t.metadata["score"].iloc[0] == 10
        assert t.metadata["score"].iloc[2] == 20


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])