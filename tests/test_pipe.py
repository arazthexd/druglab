"""
tests/test_pipe.py
~~~~~~~~~~~~~~~~~~
Tests for druglab.pipe orchestration, caching, and archetypes.
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from druglab.db.base import BaseTable, HistoryEntry
from druglab.pipe import (
    Pipeline, FunctionFeaturizer, FunctionFilter, 
    FunctionPreparation, DictCache, MemoryIOBlock
)

# ---------------------------------------------------------------------------
# Dummy Table for isolated testing
# ---------------------------------------------------------------------------

class DummyTable(BaseTable[dict]):
    """Stores plain dicts as 'objects'."""
    def _serialize_object(self, obj: dict) -> bytes: return json.dumps(obj).encode()
    def _deserialize_object(self, raw: bytes) -> dict: return json.loads(raw.decode())
    @staticmethod
    def _deserialize_object_static(raw: bytes) -> dict: return json.loads(raw.decode())
    def _object_type_name(self) -> str: return "dict"

def make_table(n: int = 10) -> DummyTable:
    objects = [{"id": i, "val": i * 10} for i in range(n)]
    metadata = pd.DataFrame({"name": [f"mol_{i}" for i in range(n)]})
    features = {}
    return DummyTable(objects=objects, metadata=metadata, features=features)

# ---------------------------------------------------------------------------
# Module-level functions (Picklable for Multiprocessing tests)
# ---------------------------------------------------------------------------

def feat_func(item): return [item["val"], item["val"] * 2]
def filter_func(item): return item["val"] % 20 == 0  # Keep evens
def prep_func(item): 
    item["val"] += 1 
    return item

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestArchetypes:
    def test_function_featurizer(self):
        t = make_table(4)
        block = FunctionFeaturizer(func=feat_func, name="DoubleVal")
        out = block.run(t)
        
        assert "DoubleVal" in out.features
        assert out.features["DoubleVal"].shape == (4, 2)
        assert out.features["DoubleVal"][1].tolist() == [10, 20]
        assert len(out.history) == 1

    def test_function_filter(self):
        t = make_table(4) # vals: 0, 10, 20, 30
        block = FunctionFilter(func=filter_func)
        out = block.run(t)
        
        assert len(out) == 2
        assert out.objects[0]["val"] == 0
        assert out.objects[1]["val"] == 20

    def test_function_preparation(self):
        t = make_table(3) # vals: 0, 10, 20
        block = FunctionPreparation(func=prep_func)
        out = block.run(t)
        
        assert out.objects[0]["val"] == 1
        assert out.objects[1]["val"] == 11
        # Check copy mechanism (original should be untouched)
        assert t.objects[0]["val"] == 0

class TestPipelineAndBatching:
    def test_standard_pipeline(self):
        t = make_table(10)
        pipe = Pipeline([
            FunctionPreparation(func=prep_func),      # adds 1
            FunctionFilter(func=lambda x: x["id"] < 5), # keeps first 5
            FunctionFeaturizer(func=feat_func, name="F1")
        ])
        
        out = pipe.run(t)
        assert len(out) == 5
        assert out.objects[4]["val"] == 41 # 4 * 10 + 1
        assert "F1" in out.features
        
        # History is 4 because BaseTable.subset() adds its own audit entry
        # inside the FunctionFilter execution.
        assert len(out.history) == 4
        assert any("subset" in h.block_name for h in out.history)

    def test_io_batch_pipeline(self):
        """Tests Scenario A: IOBlock feeding batches forward."""
        t = make_table(10)
        
        pipe = Pipeline([
            MemoryIOBlock(table=t, batch_size=3),
            FunctionFilter(func=lambda x: x["id"] % 2 == 0) # Keeps 0, 2, 4, 6, 8
        ])
        
        # We start with None, pipeline relies on the IO block to generate the data
        out = pipe.run(None)
        
        assert len(out) == 5
        assert out.objects[-1]["id"] == 8
        
        # History check: should show the concat operation from recombining batches
        assert any("concat" in h.block_name for h in out.history)

    def test_mid_pipeline_batching(self):
        """Tests Scenario B: Standard block declaring batch_size forces chunking."""
        t = make_table(10)
        
        # Featurizer declares batch_size=4. Pipeline should break table into 3 chunks.
        pipe = Pipeline([
            FunctionFeaturizer(func=feat_func, name="F1", batch_size=4),
        ])
        
        out = pipe.run(t)
        assert len(out) == 10
        assert "F1" in out.features
        # Ensure it was chunked and recombined
        assert any("concat" in h.block_name for h in out.history)

class TestMechanics:
    def test_caching(self):
        cache = DictCache()
        t = make_table(2)
        
        # Pass 1: populate cache
        block = FunctionFeaturizer(func=feat_func, name="F", use_cache=True, cache_backend=cache)
        out1 = block.run(t)
        
        # Modify the function slightly for pass 2
        def bad_func(item): return [999, 999]
        block2 = FunctionFeaturizer(func=bad_func, name="F", use_cache=True, cache_backend=cache)
        
        # Pass 2: Should hit cache and ignore bad_func
        out2 = block2.run(t)
        assert out2.features["F"][0].tolist() == [0, 0] # From pass 1, not [999, 999]

    def test_multiprocessing(self):
        t = make_table(10)
        # using n_workers=2
        block = FunctionPreparation(func=prep_func, n_workers=2)
        out = block.run(t)
        assert len(out) == 10
        assert out.objects[1]["val"] == 11

if __name__ == "__main__":
    pytest.main([__file__, "-v"])