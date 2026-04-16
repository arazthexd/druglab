import pytest
import numpy as np
import pandas as pd
import json

import logging
import pickle
import pytest

from druglab.db import BaseTable
from druglab.pipe import DictCache, BaseBlock

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

def test_dict_cache_memory_bound():
    cache = DictCache(max_size=100)
    for i in range(200):
        cache.set(f"key_{i}", "data")
    
    # Currently fails: len is 200. Should be bounded to 100.
    assert len(cache._store) <= 100
    
    cache.clear()
    assert len(cache._store) == 0

def test_block_cannot_return_none():
    class BadBlock(BaseBlock):
        def _process(self, table):
            return None

    with pytest.raises(ValueError):
        BadBlock().run(make_table(4))

def test_eager_backend_avoids_pickle_dumps(monkeypatch, tmp_path):
    import pickle
    from druglab.db.backend.memory import EagerMemoryBackend

    backend = EagerMemoryBackend(objects=[1, 2, 3])
    
    # Poison dumps/loads to ensure they raise an error if called
    def mock_dumps(*args, **kwargs):
        raise RuntimeError("OOM Risk: pickle.dumps should not be called!")
    
    monkeypatch.setattr(pickle, "dumps", mock_dumps)
    monkeypatch.setattr(pickle, "loads", mock_dumps)
    
    bundle_path = tmp_path / "test_bundle.dlb"
    
    # Create the parent bundle directory just like the Orchestrator would
    bundle_path.mkdir(parents=True, exist_ok=True)
    
    # Should stream safely via pickle.dump
    backend.save(bundle_path)
    loaded = EagerMemoryBackend.load(bundle_path)
    
    assert loaded.get_objects() == [1, 2, 3]

def test_eager_backend_serializer_mode_streams_without_list_payload(monkeypatch, tmp_path):
    from druglab.db.backend.memory import EagerMemoryBackend

    dumped_types = []
    original_dump = pickle.dump

    def spy_dump(obj, fp, *args, **kwargs):
        dumped_types.append(type(obj))
        return original_dump(obj, fp, *args, **kwargs)

    monkeypatch.setattr(pickle, "dump", spy_dump)

    backend = EagerMemoryBackend(objects=[1, 2, 3, 4])
    bundle_path = tmp_path / "stream_bundle.dlb"
    bundle_path.mkdir(parents=True, exist_ok=True)

    backend.save(bundle_path, serializer=lambda x: str(x).encode())
    loaded = EagerMemoryBackend.load(bundle_path, deserializer=lambda b: int(b.decode()))

    # The streaming format writes a header dict + one serialized item per object.
    assert dumped_types[0] is dict
    assert dumped_types.count(list) == 0
    assert loaded.get_objects() == [1, 2, 3, 4]

def test_subset_with_bool_mask_wrong_length():
    """subset() should raise, not silently truncate, on wrong-length bool mask."""
    table = make_table(4)
    bad_mask = np.array([True, False, True])  # length 3, not 4
    with pytest.raises(ValueError, match="Boolean mask length"):
        table.subset(bad_mask)


def test_concat_empty_list_raises():
    """concat([]) should raise ValueError, not AttributeError."""
    with pytest.raises(ValueError):
        DummyTable.concat([])


def test_concat_mixed_types_raises():
    """concat() with mixed table types should raise TypeError."""
    class AnotherTable(DummyTable):
        pass
    t1 = make_table(2)
    t2 = AnotherTable(objects=[{"id": 99}])
    with pytest.raises(TypeError):
        DummyTable.concat([t1, t2])


def test_update_feature_wrong_length_raises():
    """update_feature with wrong-length array should raise, not silently corrupt."""
    table = make_table(4)
    bad_array = np.zeros((3, 8), dtype=np.float32)  # 3 rows, not 4
    with pytest.raises(ValueError):
        table.update_feature("fp", bad_array)


def test_resolve_idx_out_of_bounds():
    """_resolve_idx should raise IndexError on out-of-bounds positive index."""
    from druglab.db.backend.memory import _resolve_idx
    with pytest.raises(IndexError):
        _resolve_idx([0, 5], n=4)


def test_resolve_idx_negative_out_of_bounds():
    """_resolve_idx should raise IndexError on deeply negative index."""
    from druglab.db.backend.memory import _resolve_idx
    with pytest.raises(IndexError):
        _resolve_idx(-10, n=4)


def test_set_metadata_wrong_length_raises():
    """set_metadata with wrong row count should raise ValueError."""
    table = make_table(4)
    bad_df = pd.DataFrame({"a": [1, 2, 3]})  # 3 rows, not 4
    with pytest.raises(ValueError):
        table.metadata = bad_df


def test_set_objects_wrong_length_raises():
    """set_objects with wrong count should raise ValueError."""
    table = make_table(4)
    with pytest.raises(ValueError):
        table.objects = [{"id": 0}, {"id": 1}]  # only 2


def test_validate_catches_dimension_mismatch():
    """Backend validate() should catch when features have wrong row count."""
    from druglab.db.backend.memory import EagerMemoryBackend
    import numpy as np
    backend = EagerMemoryBackend(
        objects=[1, 2, 3, 4],
        features={"fp": np.zeros((3, 8))}  # 3 rows, but 4 objects
    )
    with pytest.raises(ValueError, match="Dimension Mismatch"):
        backend.validate()


def test_feature_name_with_slash_raises():
    """Feature names with path separators should be rejected."""
    table = make_table(4)
    with pytest.raises(ValueError):
        table.update_feature("ecfp/4", np.zeros((4, 8)))


def test_dict_cache_lru_eviction_order():
    """DictCache should evict the OLDEST entry when max_size is exceeded."""
    cache = DictCache(max_size=3)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    cache.set("d", 4)  # should evict "a"
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("d") == 4


def test_save_load_roundtrip(tmp_path):
    """save/load roundtrip should preserve objects, metadata, features, history."""
    table = make_table(4)
    bundle = tmp_path / "test.dlb"
    table.save(bundle)
    loaded = DummyTable.load(bundle)
    assert loaded.n == 4
    assert list(loaded.metadata.columns) == ["name", "mw"]
    assert set(loaded.feature_names) == {"fp", "phys"}
    assert loaded.get_objects(0) == {"id": 0, "val": 0}


def test_save_no_overwrite_raises(tmp_path):
    """save() should raise FileExistsError without overwrite=True."""
    table = make_table(2)
    bundle = tmp_path / "test.dlb"
    table.save(bundle)
    with pytest.raises(FileExistsError):
        table.save(bundle)


def test_save_overwrite(tmp_path):
    """save(overwrite=True) should atomically replace the existing bundle."""
    table = make_table(4)
    bundle = tmp_path / "test.dlb"
    table.save(bundle)
    table2 = make_table(2)
    table2.save(bundle, overwrite=True)
    loaded = DummyTable.load(bundle)
    assert loaded.n == 2