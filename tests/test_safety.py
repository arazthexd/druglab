import pytest
import numpy as np
from rdkit import Chem

from druglab.pipe.cache import DictCache
from druglab.io.readers import SDFFormatReader
from druglab.pipe.pipeline import Pipeline
from druglab.pipe.base import BaseBlock
from druglab.pipe.blocks.prepare import MoleculeDesalter

def test_dict_cache_memory_bound():
    cache = DictCache(max_size=100)
    for i in range(200):
        cache.set(f"key_{i}", "data")
    
    # Currently fails: len is 200. Should be bounded to 100.
    assert len(cache._store) <= 100
    
    cache.clear()
    assert len(cache._store) == 0

