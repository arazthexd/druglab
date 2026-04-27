from __future__ import annotations

import numpy as np
import pandas as pd

from druglab.db.backend import CompositeStorageBackend
from druglab.db.backend.memory import (
    MemoryFeatureStore,
    MemoryMetadataStore,
    MemoryObjectStore,
)


def _make_composite(n: int = 4) -> CompositeStorageBackend:
    obj = MemoryObjectStore([{"id": i} for i in range(n)])
    meta = MemoryMetadataStore(pd.DataFrame({"m": list(range(n))}))
    feat = MemoryFeatureStore({"f": np.arange(n * 2, dtype=np.float32).reshape(n, 2)})
    return CompositeStorageBackend(obj, meta, feat)


class TestCompositeStorageBackend:
    def test_injection_with_memory_stores(self):
        backend = _make_composite(5)
        assert len(backend) == 5
        assert backend.get_feature("f").shape == (5, 2)

    def test_manifest_round_trip(self, tmp_path):
        backend = _make_composite(3)
        backend.save(tmp_path)

        manifest = tmp_path / "composite_manifest.json"
        assert manifest.exists()

        loaded = CompositeStorageBackend.load(tmp_path)
        assert isinstance(loaded, CompositeStorageBackend)
        assert loaded.get_objects() == backend.get_objects()
        pd.testing.assert_frame_equal(loaded.get_metadata(), backend.get_metadata())
        np.testing.assert_array_equal(loaded.get_feature("f"), backend.get_feature("f"))