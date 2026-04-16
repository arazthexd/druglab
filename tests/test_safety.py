import pytest
import numpy as np
import pandas as pd
import json
from rdkit import Chem

from druglab.pipe.cache import DictCache
from druglab.io.readers import SDFFormatReader
from druglab.pipe.pipeline import Pipeline
from druglab.pipe.base import BaseBlock
from druglab.db import BaseTable
from druglab.pipe.blocks.prepare import MoleculeDesalter

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
