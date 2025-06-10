import numpy as np
import pandas as pd
import h5py
import tempfile
from pathlib import Path
from unittest.mock import Mock

from druglab.storage.base import StorageMetadata, StorageFeatures, BaseStorage


class TestStorageMetadata:
    """Test StorageMetadata class."""
    
    def test_init_and_access(self):
        meta = StorageMetadata(key1="value1", key2=42)
        assert meta["key1"] == "value1"
        assert meta["key2"] == 42
        assert "key1" in meta
        assert "key3" not in meta
    
    def test_dict_operations(self):
        meta = StorageMetadata(a=1, b=2)
        meta["c"] = 3
        assert meta["c"] == 3
        
        del meta["a"]
        assert "a" not in meta
        
        assert meta.get("b") == 2
        assert meta.get("missing", "default") == "default"
    
    def test_update_and_copy(self):
        meta1 = StorageMetadata(a=1, b=2)
        meta2 = StorageMetadata(b=3, c=4)
        
        meta1.update(meta2)
        assert meta1["b"] == 3
        assert meta1["c"] == 4
        
        meta3 = meta1.copy()
        meta3["d"] = 5
        assert "d" not in meta1
    
    def test_compatibility(self):
        meta1 = StorageMetadata(type="mol", version=1)
        meta2 = StorageMetadata(type="mol", version=2)
        meta3 = StorageMetadata(type="rxn", version=1)
        
        assert meta1.is_compatible(meta2, required_keys=[])
        assert meta1.is_compatible(meta2, required_keys=["type"])
        assert not meta1.is_compatible(meta2, required_keys=["version"])
        assert not meta1.is_compatible(meta3, required_keys=["type"])


class TestStorageFeatures:
    """Test StorageFeatures class."""
    
    def test_add_and_get_features(self):
        features = StorageFeatures()
        data = np.random.rand(10, 5)
        
        features.add_features("test", data)
        
        assert "test" in features
        assert len(features) == 1
        np.testing.assert_array_equal(features.get_features("test"), data)
    
    def test_add_features_with_metadata(self):
        features = StorageFeatures()
        data = np.random.rand(5, 3)
        
        mock_featurizer = Mock()
        mock_featurizer.fnames = ["feat1", "feat2", "feat3"]
        mock_featurizer.__class__.__name__ = "MockFeaturizer"
        
        features.add_features(
            "test", data, 
            featurizer=mock_featurizer,
            metadata={"custom": "value"}
        )
        
        metadata = features.get_metadata("test")
        assert metadata["feature_names"] == ["feat1", "feat2", "feat3"]
        assert metadata["featurizer_class"] == "MockFeaturizer"
        assert metadata["custom"] == "value"
    
    def test_remove_features(self):
        features = StorageFeatures()
        data = np.random.rand(5, 3)
        features.add_features("test", data)
        
        assert features.remove_features("test")
        assert "test" not in features
        assert not features.remove_features("nonexistent")
    
    def test_subset(self):
        features = StorageFeatures()
        data1 = np.random.rand(10, 3)
        data2 = np.random.rand(10, 4)
        
        features.add_features("feat1", data1)
        features.add_features("feat2", data2)
        
        indices = [0, 2, 4]
        subset = features.subset(indices)
        
        np.testing.assert_array_equal(subset.get_features("feat1"), 
                                      data1[indices])
        np.testing.assert_array_equal(subset.get_features("feat2"), 
                                      data2[indices])
    
    def test_extend(self):
        features1 = StorageFeatures()
        features2 = StorageFeatures()
        
        data1 = np.random.rand(5, 3)
        data2 = np.random.rand(3, 3)
        
        features1.add_features("test", data1)
        features2.add_features("test", data2)
        
        features1.extend(features2)
        
        result = features1.get_features("test")
        expected = np.vstack([data1, data2])
        np.testing.assert_array_equal(result, expected)
    
    def test_concatenate_all(self):
        features = StorageFeatures()
        data1 = np.random.rand(5, 2)
        data2 = np.random.rand(5, 3)
        
        features.add_features("a_feat", data1)
        features.add_features("b_feat", data2)
        
        concat_data, names = features.concatenate_all()
        
        assert concat_data.shape == (5, 5)
        assert len(names) == 5
        np.testing.assert_array_equal(concat_data[:, :2], data1)
        np.testing.assert_array_equal(concat_data[:, 2:], data2)
    
    def test_to_dataframe(self):
        features = StorageFeatures()
        data = np.array([[1, 2], [3, 4], [5, 6]])
        
        mock_featurizer = Mock()
        mock_featurizer.fnames = ["col1", "col2"]
        mock_featurizer.__class__.__name__ = "TestFeaturizer"
        
        features.add_features("test", data, featurizer=mock_featurizer)
        
        df = features.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["col1", "col2"]
        np.testing.assert_array_equal(df.values, data)
    
    def test_to_dataframe_with_metadata(self):
        features = StorageFeatures()
        data = np.random.rand(3, 2)
        features.add_features("test", data)
        
        df = features.to_dataframe(include_metadata=True)
        assert hasattr(df, 'attrs')
        assert 'feature_metadata' in df.attrs


# Concrete implementation for testing BaseStorage
class ExampleStorage(BaseStorage):
    @property
    def required_object_keys(self):
        return ['items']
    
    @property
    def save_dtypes(self):
        return {'items': h5py.string_dtype()}
    
    def __init__(self, items=None, **kwargs):
        objects = {'items': items or []}
        super().__init__(objects=objects, **kwargs)
    
    def get_save_ready_objects(self):
        return {'items': [str(item) for item in self._objects['items']]}
    
    def get_load_ready_objects(self, db, indices=None):
        if indices is None:
            data = db['items'][:]
        else:
            data = db['items'][indices]
        
        items = [item.decode() if isinstance(item, bytes) else str(item) 
                 for item in data]
        return {'items': items}


class TestBaseStorage:
    """Test BaseStorage functionality using TestStorage."""
    
    def test_init_empty(self):
        storage = ExampleStorage()
        assert len(storage) == 0
        assert storage.num_features == 0
    
    def test_init_with_data(self):
        items = ['a', 'b', 'c']
        storage = ExampleStorage(items)
        assert len(storage) == 3
        assert storage._objects['items'] == items
    
    def test_properties(self):
        storage = ExampleStorage(['a', 'b'])
        
        # Test objects property returns copy
        objects = storage.objects
        objects['items'].append('c')
        assert len(storage) == 2  # Original unchanged
        
        # Test features property
        assert isinstance(storage.features, StorageFeatures)
        assert isinstance(storage.metadata, StorageMetadata)
    
    def test_sample(self):
        storage = ExampleStorage(['a', 'b', 'c', 'd'])
        
        # Sample single item
        result = storage.sample(1)
        assert isinstance(result, dict)
        assert 'items' in result
        assert result['items'] in ['a', 'b', 'c', 'd']
        
        # Sample multiple items
        result = storage.sample(2)
        assert isinstance(result, dict)
    
    def test_indexing(self):
        storage = ExampleStorage(['a', 'b', 'c'])
        
        # Single index
        assert storage[0] == 'a'
        assert storage[1] == 'b'
        
        # Multiple indices
        result = storage[[0, 2]]
        assert result == ['a', 'c']
        
        # Setting items
        storage[0] = 'x'
        assert storage[0] == 'x'
    
    def test_extend(self):
        storage1 = ExampleStorage(['a', 'b'])
        storage2 = ExampleStorage(['c', 'd'])
        
        storage1.extend(storage2)
        assert len(storage1) == 4
        assert storage1._objects['items'] == ['a', 'b', 'c', 'd']
    
    def test_extend_empty(self):
        storage1 = ExampleStorage()
        storage2 = ExampleStorage(['a', 'b'])
        
        storage1.extend(storage2)
        assert len(storage1) == 2
        assert storage1._objects['items'] == ['a', 'b']
    
    def test_subset(self):
        storage = ExampleStorage(['a', 'b', 'c', 'd'])
        
        # Add some features
        features = np.random.rand(4, 3)
        storage.features.add_features('test', features)
        
        subset = storage.subset([0, 2])
        
        assert len(subset) == 2
        assert subset._objects['items'] == ['a', 'c']
        np.testing.assert_array_equal(
            subset.features.get_features('test'), 
            features[[0, 2]]
        )
    
    def test_delete_item(self):
        storage = ExampleStorage(['a', 'b', 'c'])
        
        # Add features
        features = np.random.rand(3, 2)
        storage.features.add_features('test', features)
        
        del storage[1]
        
        assert len(storage) == 2
        assert storage._objects['items'] == ['a', 'c']
        
        # Check features were also deleted
        remaining_features = storage.features.get_features('test')
        expected_features = np.array([features[0], features[2]])
        np.testing.assert_array_equal(remaining_features, expected_features)
    
    def test_iteration(self):
        storage = ExampleStorage(['a', 'b', 'c'])
        
        items = list(storage)
        assert len(items) == 3
        assert items[0] == {'items': 'a'}
        assert items[1] == {'items': 'b'}
        assert items[2] == {'items': 'c'}
    
    def test_contains(self):
        storage = ExampleStorage(['a', 'b', 'c'])
        assert 'a' in storage
        assert 'd' not in storage
    
    def test_save_load_cycle(self):
        # Create storage with data
        storage = ExampleStorage(['item1', 'item2', 'item3'])
        
        # Add features
        features = np.random.rand(3, 4)
        storage.features.add_features('test_feat', features)
        
        # Add metadata
        storage.metadata['version'] = 1
        storage.metadata['created_by'] = 'test'
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save
            storage.write(temp_path)
            
            # Load
            loaded_storage = ExampleStorage.load(temp_path)
            
            # Verify objects
            assert len(loaded_storage) == 3
            assert loaded_storage._objects['items'] == ['item1', 
                                                        'item2', 
                                                        'item3']
            
            # Verify features
            loaded_features = loaded_storage.features.get_features('test_feat')
            np.testing.assert_array_almost_equal(loaded_features, features)
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_save_load_with_indices(self):
        storage = ExampleStorage(['a', 'b', 'c', 'd'])
        features = np.random.rand(4, 2)
        storage.features.add_features('test', features)
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_path = f.name
        
        try:
            storage.write(temp_path)
            
            # Load subset
            loaded_storage = ExampleStorage.load(temp_path, indices=[0, 2])
            
            assert len(loaded_storage) == 2
            assert loaded_storage._objects['items'] == ['a', 'c']
            
            loaded_features = loaded_storage.features.get_features('test')
            np.testing.assert_array_equal(loaded_features, features[[0, 2]])
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_repr(self):
        storage = ExampleStorage(['a', 'b'])
        storage.features.add_features('test', np.random.rand(2, 3))
        
        repr_str = repr(storage)
        assert storage.__class__.__name__ in repr_str
        assert '2 objects' in repr_str
        assert '1 feature sets' in repr_str