"""Tests for BaseStorageBackend as a strict abstract contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from druglab.db.backend.base import BaseStorageBackend
from druglab.db.backend import EagerMemoryBackend


class TestBaseStorageBackendABC:
    def test_base_backend_is_abstract(self):
        with pytest.raises(TypeError):
            BaseStorageBackend()

    def test_eager_backend_has_schema_uuid(self):
        backend = EagerMemoryBackend(
            objects=[{"id": 0}],
            metadata=pd.DataFrame({"m": [1]}),
            features={"f": np.ones((1, 1), dtype=np.float32)},
        )
        assert isinstance(backend.schema_uuid, str)
        assert len(backend.schema_uuid) > 0


class TestBaseBackendSharedBehaviors:
    def test_clone_uses_materialized_state_and_new_uuid(self):
        backend = EagerMemoryBackend(
            objects=[{"id": 0}, {"id": 1}],
            metadata=pd.DataFrame({"m": [1, 2]}),
            features={"f": np.ones((2, 2), dtype=np.float32)},
        )
        cloned = backend.clone()

        assert isinstance(cloned, EagerMemoryBackend)
        assert cloned.schema_uuid != backend.schema_uuid
        assert cloned.get_objects() == backend.get_objects()
        pd.testing.assert_frame_equal(cloned.get_metadata(), backend.get_metadata())
        np.testing.assert_array_equal(cloned.get_feature("f"), backend.get_feature("f"))

    def test_save_and_load_round_trip(self, tmp_path: Path):
        backend = EagerMemoryBackend(
            objects=[{"id": 0}, {"id": 1}],
            metadata=pd.DataFrame({"m": [1, 2]}),
            features={"f": np.arange(4, dtype=np.float32).reshape(2, 2)},
        )
        backend.save(tmp_path)

        loaded = EagerMemoryBackend.load(tmp_path)
        assert loaded.get_objects() == backend.get_objects()
        pd.testing.assert_frame_equal(loaded.get_metadata(), backend.get_metadata())
        np.testing.assert_array_equal(loaded.get_feature("f"), backend.get_feature("f"))


class TestEagerBackendValidation:
    def test_validate_raises_on_dimension_mismatch(self):
        with pytest.raises(ValueError):
            EagerMemoryBackend(
                objects=[{"id": 0}, {"id": 1}],
                metadata=pd.DataFrame({"m": [1]}),
                features={"f": np.ones((2, 1), dtype=np.float32)},
            )
