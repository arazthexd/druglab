"""
tests/unit/db/backend/test_base.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive tests for druglab.db.backend.base — the core mixins and
interfaces for storage backends.

Covers:
1. _LifecycleBase — terminal kwargs absorption and cooperative MRO.
2. BaseFeatureMixin — default get/update implementations and validation.
3. BaseMetadataMixin — set_metadata bounds checking and column delegation.
4. BaseObjectMixin — set_objects length validation.
5. BaseStorageBackend — __init__ lifecycle sequence and validate() consistency.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pytest

from druglab.db.backend.base.mixins._lifecycle import _LifecycleBase
from druglab.db.backend.base.mixins.feature import BaseFeatureMixin
from druglab.db.backend.base.mixins.metadata import BaseMetadataMixin
from druglab.db.backend.base.mixins.objects import BaseObjectMixin
from druglab.db.backend.base import BaseStorageBackend

# ===========================================================================
# Dummy Implementations for ABCs
# ===========================================================================

class DummyLifecycleA(_LifecycleBase):
    """Dummy lifecycle mixin to track initialization order."""
    def initialize_storage_context(self, **kwargs: Any) -> None:
        self.init_a_called = True
        self.init_a_kwargs = kwargs.copy()
        super().initialize_storage_context(**kwargs)

class DummyLifecycleB(_LifecycleBase):
    """Dummy lifecycle mixin to test kwarg consumption."""
    def initialize_storage_context(self, **kwargs: Any) -> None:
        self.init_b_called = True
        # consume a kwarg and pass the rest
        kwargs.pop("b_arg", None)
        super().initialize_storage_context(**kwargs)


class DummyFeatureMixin(BaseFeatureMixin):
    """Dummy feature mixin for testing default feature delegations."""
    def __init__(self):
        self.features = {}
        self.called_update = []

    def get_feature(self, name: str, idx: Optional[Any] = None) -> np.ndarray:
        return self.features[name]

    def update_feature(self, name: str, array: np.ndarray, idx: Optional[Any] = None, na: Any = None, **kwargs) -> None:
        self.called_update.append((name, array))

    def drop_feature(self, name: str) -> None:
        del self.features[name]

    def get_feature_names(self) -> List[str]:
        return list(self.features.keys())

    def get_feature_shape(self, name: str) -> tuple:
        return self.features[name].shape


class DummyMetadataMixin(BaseMetadataMixin):
    """Dummy metadata mixin for testing metadata schema operations."""
    def __init__(self, n_rows=5):
        self.n_rows = n_rows
        self.added_columns = {}
        self.dropped_all = False

    def get_metadata(self, idx: Optional[Any] = None, cols: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        return pd.DataFrame()

    def add_metadata_column(self, name: str, value: Any, idx: Optional[Any] = None, na: Any = None, **kwargs) -> None:
        self.added_columns[name] = value

    def update_metadata(self, values: Any, idx: Optional[Any] = None, **kwargs) -> None:
        pass

    def drop_metadata_columns(self, cols: Optional[Union[str, List[str]]] = None) -> None:
        if cols is None:
            self.dropped_all = True

    def _n_metadata_rows(self) -> int:
        return self.n_rows


class DummyObjectMixin(BaseObjectMixin):
    """Dummy object mixin for testing object schema operations."""
    def __init__(self, n_objs=5):
        self.n_objs = n_objs
        self.objects = []

    def get_objects(self, idx: Optional[Any] = None) -> Union[Any, List[Any]]:
        return self.objects

    def update_objects(self, objs: Union[Any, List[Any]], idx: Optional[Any] = None, **kwargs) -> None:
        self.objects = objs

    def _n_objects(self) -> int:
        return self.n_objs


class DummyFullBackend(BaseStorageBackend):
    """Dummy backend assembling all mixins for testing lifecycle hooks."""
    def __init__(self, expected_len=5, meta_len=5, feat_len=5, obj_len=5, **kwargs):
        self._expected_len = expected_len
        self._meta_len = meta_len
        self._feat_len = feat_len
        self._obj_len = obj_len
        self.hooks_fired = []
        super().__init__(**kwargs)

    def __len__(self):
        return self._expected_len

    def _n_metadata_rows(self) -> int:
        return self._meta_len

    def _n_feature_rows(self) -> int:
        return self._feat_len

    def _n_objects(self) -> int:
        return self._obj_len

    # Provide minimal ABC implementations to allow instantiation
    def get_feature(self, *args, **kwargs): pass
    def update_feature(self, *args, **kwargs): pass
    def drop_feature(self, *args, **kwargs): pass
    def get_feature_names(self): return []
    def get_feature_shape(self, *args, **kwargs): return (self._feat_len,)
    def get_metadata(self, *args, **kwargs): return pd.DataFrame()
    def add_metadata_column(self, *args, **kwargs): pass
    def update_metadata(self, *args, **kwargs): pass
    def drop_metadata_columns(self, *args, **kwargs): pass
    def get_objects(self, *args, **kwargs): return []
    def update_objects(self, *args, **kwargs): pass

    # Override hooks to track order
    def initialize_storage_context(self, **kwargs):
        self.hooks_fired.append("init")
        super().initialize_storage_context(**kwargs)

    def bind_capabilities(self):
        self.hooks_fired.append("bind")
        super().bind_capabilities()

    def post_initialize_validate(self):
        self.hooks_fired.append("validate")
        super().post_initialize_validate()

# ===========================================================================
# Section 1: _LifecycleBase
# ===========================================================================

class TestLifecycleBase:
    def test_terminal_node_absorbs_kwargs(self):
        """Ensure the base terminal node absorbs unknown kwargs silently."""
        # A direct instantiation should absorb any kwargs without TypeError
        node = _LifecycleBase()
        node.initialize_storage_context(random_arg=123, another="test")
        node.bind_capabilities()
        node.post_initialize_validate()
        # If no exceptions were raised, the terminal absorption worked.
        assert True

    def test_cooperative_mro_chain(self):
        """Verify cooperative super() chains propagate through the MRO correctly."""
        class Combined(DummyLifecycleB, DummyLifecycleA): # Important ordering
            pass
        
        instance = Combined()
        instance.initialize_storage_context(b_arg=99, shared_arg="hello")
        
        # Verify both mixins fired
        assert instance.init_a_called
        assert instance.init_b_called
        
        # Verify kwargs were passed and popped correctly
        # DummyLifecycleB is next in MRO, it pops 'b_arg', so DummyLifecycleA shouldn't see it.
        assert "b_arg" not in instance.init_a_kwargs
        assert instance.init_a_kwargs["shared_arg"] == "hello"

# ===========================================================================
# Section 2: BaseFeatureMixin
# ===========================================================================

class TestBaseFeatureMixin:
    def test_get_features_delegates_to_get_feature(self):
        """Check that get_features delegates to get_feature properly."""
        backend = DummyFeatureMixin()
        backend.features = {
            "feat1": np.array([1, 2, 3]),
            "feat2": np.array([4, 5, 6])
        }
        
        # Test all features
        res = backend.get_features()
        assert list(res.keys()) == ["feat1", "feat2"]
        assert np.array_equal(res["feat1"], backend.features["feat1"])

        # Test subset
        res_subset = backend.get_features(["feat2"])
        assert list(res_subset.keys()) == ["feat2"]

    def test_update_features_delegates(self):
        """Check that update_features delegates to update_feature."""
        backend = DummyFeatureMixin()
        arrays = {
            "f1": np.array([0, 0]),
            "f2": np.array([1, 1])
        }
        backend.update_features(arrays)
        
        assert len(backend.called_update) == 2
        names = [call[0] for call in backend.called_update]
        assert "f1" in names and "f2" in names

    def test_n_feature_rows_empty(self):
        """Ensure row count falls back to backend length when features are empty."""
        backend = DummyFeatureMixin()
        # Mock __len__ on the dummy object to test the fallback
        backend.__class__.__len__ = lambda self: 10
        assert backend._n_feature_rows() == 10

    def test_n_feature_rows_populated(self):
        """Verify row count returns the shape of the first feature array."""
        backend = DummyFeatureMixin()
        backend.features["f1"] = np.zeros((7, 3))
        assert backend._n_feature_rows() == 7

    def test_validate_features_success(self):
        """Test that validation passes when all feature arrays have identical lengths."""
        backend = DummyFeatureMixin()
        backend.features["f1"] = np.zeros((5, 3))
        backend.features["f2"] = np.zeros((5, 2))
        # Should not raise
        backend._validate_features()

    def test_validate_features_mismatch_raises(self):
        """Ensure validation raises ValueError on feature length mismatch."""
        backend = DummyFeatureMixin()
        backend.features["f1"] = np.zeros((5, 3))
        backend.features["f2"] = np.zeros((4, 2))
        
        with pytest.raises(ValueError, match="Feature 'f2' has 4 rows, expected 5"):
            backend._validate_features()

# ===========================================================================
# Section 3: BaseMetadataMixin
# ===========================================================================

class TestBaseMetadataMixin:
    def test_add_metadata_columns_delegates(self):
        """Check that add_metadata_columns delegates to add_metadata_column."""
        backend = DummyMetadataMixin()
        backend.add_metadata_columns({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })
        
        assert "col1" in backend.added_columns
        assert "col2" in backend.added_columns
        assert backend.added_columns["col1"] == [1, 2, 3]

    def test_set_metadata_success(self):
        """Verify set_metadata replaces all existing metadata columns."""
        backend = DummyMetadataMixin(n_rows=3)
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        
        backend.set_metadata(df)
        
        assert backend.dropped_all is True
        assert "A" in backend.added_columns
        assert np.array_equal(backend.added_columns["A"], [1, 2, 3])

    def test_set_metadata_length_mismatch(self):
        """Ensure set_metadata raises ValueError if dataframe length mismatches backend."""
        backend = DummyMetadataMixin(n_rows=5) # backend expects 5
        df = pd.DataFrame({"A": [1, 2, 3]})    # df only has 3
        
        with pytest.raises(ValueError, match="metadata has 3 rows but backend has 5 objects"):
            backend.set_metadata(df)

# ===========================================================================
# Section 4: BaseObjectMixin
# ===========================================================================

class TestBaseObjectMixin:
    def test_set_objects_success(self):
        """Test that set_objects correctly delegates to update_objects."""
        backend = DummyObjectMixin(n_objs=3)
        new_objs = ["obj1", "obj2", "obj3"]
        
        backend.set_objects(new_objs)
        assert backend.objects == new_objs

    def test_set_objects_length_mismatch(self):
        """Ensure set_objects raises ValueError when lengths mismatch."""
        backend = DummyObjectMixin(n_objs=5)
        new_objs = ["obj1", "obj2"]
        
        with pytest.raises(ValueError, match="new objects has 2 items but backend has 5"):
            backend.set_objects(new_objs)

# ===========================================================================
# Section 5: BaseStorageBackend
# ===========================================================================

class TestBaseStorageBackend:
    def test_init_lifecycle_orchestration(self):
        """Verify __init__ fires the three lifecycle hooks in order."""
        # Verify that __init__ fires the three hooks in the correct order
        # and safely absorbs extra kwargs.
        backend = DummyFullBackend(extra_arg="should_be_ignored")
        
        assert backend.hooks_fired == ["init", "bind", "validate"]

    def test_validate_success(self):
        """Ensure validate passes when all mixin dimensions match."""
        # All dimensions match (default is 5)
        backend = DummyFullBackend(expected_len=5, meta_len=5, feat_len=5, obj_len=5)
        # Should not raise
        backend.validate()

    def test_validate_mismatch_raises(self):
        """Ensure validate raises ValueError when dimensions diverge across mixins."""
        # Cause a mismatch in the object count
        backend = DummyFullBackend(expected_len=5, meta_len=5, feat_len=5, obj_len=4)
        
        expected_msg = (
            "Backend Dimension Mismatch!\n"
            "Global Length: 5\n"
            "Metadata Rows: 5\n"
            "Feature Rows:  5\n"
            "Object Count:  4"
        )
        
        with pytest.raises(ValueError, match=expected_msg):
            backend.validate()

# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
