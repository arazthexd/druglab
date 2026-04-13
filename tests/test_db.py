"""
tests/test_db.py
~~~~~~~~~~~~~~~~
Tests for druglab.db.  Written to run without RDKit by using a
lightweight ``DummyTable`` subclass that stores plain dicts.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from druglab.db.base import BaseTable, HistoryEntry

# ---------------------------------------------------------------------------
# Minimal concrete subclass (no RDKit required)
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
    objects = [{"id": i, "val": i * 10} for i in range(n)]
    metadata = pd.DataFrame({"name": [f"mol_{i}" for i in range(n)], "mw": [float(i) for i in range(n)]})
    features = {
        "fp": np.arange(n * 8, dtype=np.float32).reshape(n, 8),
        "phys": np.ones((n, 3), dtype=np.float64),
    }
    return DummyTable(objects=objects, metadata=metadata, features=features)


# ---------------------------------------------------------------------------
# HistoryEntry
# ---------------------------------------------------------------------------

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
            entry.block_name = "other"  # frozen dataclass


# ---------------------------------------------------------------------------
# Construction & invariant enforcement
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_basic(self):
        t = make_table(4)
        assert t.n == 4
        assert len(t) == 4

    def test_empty(self):
        t = DummyTable()
        assert t.n == 0
        assert t.metadata.empty or len(t.metadata) == 0

    def test_metadata_row_mismatch_raises(self):
        objects = [{"id": 0}]
        bad_meta = pd.DataFrame({"x": [1, 2, 3]})  # 3 rows, 1 object
        with pytest.raises(ValueError, match="metadata"):
            DummyTable(objects=objects, metadata=bad_meta)

    def test_feature_row_mismatch_raises(self):
        objects = [{"id": i} for i in range(3)]
        bad_feat = {"fp": np.zeros((5, 8))}  # 5 rows, 3 objects
        with pytest.raises(ValueError, match="Feature"):
            DummyTable(objects=objects, features=bad_feat)

    def test_repr_contains_class_name(self):
        t = make_table(2)
        r = repr(t)
        assert "DummyTable" in r
        assert "n=2" in r


# ---------------------------------------------------------------------------
# Metadata & feature helpers
# ---------------------------------------------------------------------------

class TestMutators:
    def test_add_metadata_column(self):
        t = make_table(3)
        t.add_metadata_column("activity", [1.0, 2.0, 3.0])
        assert "activity" in t.metadata.columns
        assert list(t.metadata["activity"]) == [1.0, 2.0, 3.0]

    def test_add_metadata_column_length_mismatch(self):
        t = make_table(3)
        with pytest.raises(ValueError):
            t.add_metadata_column("bad", [1, 2])

    def test_drop_metadata_column(self):
        t = make_table(3)
        t.drop_metadata_columns("name")
        assert "name" not in t.metadata.columns

    def test_add_feature(self):
        t = make_table(4)
        new_feat = np.zeros((4, 16))
        t.add_feature("ecfp4", new_feat)
        assert "ecfp4" in t.features
        assert t.features["ecfp4"].shape == (4, 16)

    def test_add_feature_wrong_rows(self):
        t = make_table(4)
        with pytest.raises(ValueError):
            t.add_feature("bad", np.zeros((3, 16)))

    def test_drop_feature(self):
        t = make_table(4)
        t.drop_feature("fp")
        assert "fp" not in t.features

    def test_has_feature(self):
        t = make_table(4)
        assert t.has_feature("fp")
        assert not t.has_feature("nonexistent")


# ---------------------------------------------------------------------------
# Subsetting / slicing
# ---------------------------------------------------------------------------

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

    def test_subset_adds_history(self):
        t = make_table(4)
        sub = t.subset([0, 1])
        assert any("subset" in e.block_name for e in sub.history)

    def test_subset_returns_same_subclass(self):
        t = make_table(4)
        sub = t.subset([0, 1])
        assert type(sub) is DummyTable


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------

class TestCopy:
    def test_copy_is_independent(self):
        t = make_table(3)
        c = t.copy()
        c._objects[0]["val"] = 9999
        assert t._objects[0]["val"] == 0  # original unchanged

    def test_copy_features_independent(self):
        t = make_table(3)
        c = t.copy()
        c._features["fp"][0, 0] = -1.0
        assert t._features["fp"][0, 0] != -1.0


# ---------------------------------------------------------------------------
# Concatenation
# ---------------------------------------------------------------------------

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
        assert c.features["fp"][2].sum() == 0.0  # filled with zeros

    def test_concat_missing_feature_raise(self):
        a = make_table(2)
        b = DummyTable(
            objects=[{"id": 99}],
            metadata=pd.DataFrame({"name": ["x"], "mw": [0.0]}),
            features={},
        )
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


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

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
        block_names = [e.block_name for e in sub.history]
        assert "InitBlock" in block_names


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        t = make_table(4)
        t.append_history(HistoryEntry.now("Step1", {"a": 1}, 4, 4))
        saved_path = t.save(tmp_path / "mytable")

        loaded = DummyTable.load(saved_path)
        assert loaded.n == 4
        assert loaded.objects[2]["id"] == 2
        assert "name" in loaded.metadata.columns
        assert "fp" in loaded.features
        assert loaded.features["fp"].shape == (4, 8)
        assert np.allclose(loaded.features["fp"], t.features["fp"])
        assert len(loaded.history) == 1
        assert loaded.history[0].block_name == "Step1"

    def test_save_creates_expected_files(self, tmp_path: Path):
        t = make_table(2)
        root = t.save(tmp_path / "t")
        assert (root / "_meta.json").exists()
        assert (root / "metadata.parquet").exists() or (root / "metadata.csv").exists()
        assert (root / "history.json").exists()
        assert (root / "objects" / "0000000.pkl").exists()
        assert (root / "features" / "fp.npy").exists()

    def test_load_nonexistent_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            DummyTable.load(tmp_path / "does_not_exist")

    def test_save_load_empty_table(self, tmp_path: Path):
        t = DummyTable()
        t.save(tmp_path / "empty")
        loaded = DummyTable.load(tmp_path / "empty")
        assert loaded.n == 0

    def test_mmap_load(self, tmp_path: Path):
        t = make_table(4)
        t.save(tmp_path / "mmapped")
        loaded = DummyTable.load(tmp_path / "mmapped", mmap_features=True)
        assert loaded.features["fp"].shape == (4, 8)
        assert np.allclose(loaded.features["fp"], t.features["fp"])


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])