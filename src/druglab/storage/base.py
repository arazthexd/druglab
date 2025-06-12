from __future__ import annotations
from typing import List, Any, Tuple, Type, Dict, Optional, Union
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import random

import h5py
import numpy as np
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
from sklearn.neighbors import NearestNeighbors

try:
    from ..featurize import BaseFeaturizer # for the sake of typing
except ImportError:
    pass

logger = logging.getLogger(__name__)


class StorageMetadata:
    """Generic metadata container that works as a dictionary."""
    
    def __init__(self, **kwargs):
        self._data = kwargs
    
    def __getitem__(self, key: str) -> Any:
        return self._data[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value
    
    def __delitem__(self, key: str) -> None:
        del self._data[key]
    
    def __contains__(self, key: str) -> bool:
        return key in self._data
    
    def __iter__(self):
        return iter(self._data)
    
    def keys(self):
        return self._data.keys()
    
    def values(self):
        return self._data.values()
    
    def items(self):
        return self._data.items()
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)
    
    def update(self, other: Union[dict, StorageMetadata]) -> None:
        if isinstance(other, StorageMetadata):
            self._data.update(other._data)
        else:
            self._data.update(other)
    
    def copy(self) -> StorageMetadata:
        return StorageMetadata(**self._data.copy())
    
    def to_dict(self) -> Dict[str, Any]:
        return self._data.copy()
    
    def is_compatible(self, other: StorageMetadata, 
                     required_keys: Optional[List[str]] = None) -> bool:
        """Check if two metadata objects are compatible."""
        if required_keys is None:
            required_keys = []
        
        for key in required_keys:
            if self.get(key) != other.get(key):
                return False
        return True

class StorageFeatures:
    """Container for managing multiple feature sets with different dtypes and metadata."""
    
    def __init__(self):
        self._features: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, StorageMetadata] = {}
        self._featurizers: Dict[str, Optional[BaseFeaturizer]] = {}
    
    def add_features(self, 
                     key: str, 
                     features: np.ndarray,
                     dtype: Optional[Type[np.dtype]] = None,
                     featurizer: Optional[BaseFeaturizer] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a feature set with given key."""
        if dtype is not None:
            features = features.astype(dtype)
        
        self._features[key] = features
        self._featurizers[key] = featurizer
        
        # Create metadata for this feature set
        feat_metadata = StorageMetadata()
        if featurizer is not None:
            feat_metadata['feature_names'] = featurizer.fnames.copy()
            feat_metadata['featurizer_class'] = featurizer.__class__.__name__
        else:
            feat_metadata['feature_names'] = [
                f"{key}_feat_{i}" for i in range(features.shape[1])
            ]
        
        if metadata:
            if any(key in feat_metadata for key in metadata):
                logger.debug("When adding features to storage feats, some "
                             "metadata keys were identical. They will be "
                             "overwritten.")
            feat_metadata.update(metadata)
        
        self._metadata[key] = feat_metadata

    def remove_features(self, key: str) -> bool:
        """Remove a feature set."""
        if key in self._features:
            del self._features[key]
            del self._metadata[key]
            del self._featurizers[key]
            return True
        return False
    
    def get_features(self, key: str) -> Optional[np.ndarray]:
        """Get features by key."""
        return self._features.get(key)
    
    def get_all_features(self) -> Dict[str, np.ndarray]:
        """Get all feature sets."""
        return self._features.copy()
    
    def get_metadata(self, key: str) -> Optional[StorageMetadata]:
        """Get metadata for a feature set."""
        return self._metadata.get(key)
    
    def get_featurizer(self, key: str) -> Optional[BaseFeaturizer]:
        """Get featurizer for a feature set."""
        return self._featurizers.get(key)
    
    def keys(self):
        """Get all feature keys."""
        return self._features.keys()
    
    def subset(self, indices: Union[List[int], np.ndarray]) -> StorageFeatures:
        """Create a subset of features."""
        subset_features = StorageFeatures()
        
        for key in self._features.keys():
            subset_features.add_features(
                key=key,
                features=self._features[key][indices],
                featurizer=self._featurizers[key],
                metadata=self._metadata[key].to_dict()
            )
        
        return subset_features
    
    def extend(self, other: StorageFeatures) -> None:
        """Extend with another StorageFeatures object."""
        if set(other.keys()) != set(self.keys()):
            logger.warning("Some feature keys are incompatible when extending "
                           "StorageFeatures instance.")
        
        for key in other.keys():
            assert key in self
            assert self._metadata[key].is_compatible(other._metadata[key],
                                                     required_keys=['feature_names'])
            self._features[key] = np.vstack(
                (self._features[key], other._features[key]),
                dtype=self._features[key].dtype,
            )
    
    def concatenate_all(self, dtype: Optional[Type[np.dtype]] = None) \
        -> Tuple[np.ndarray, List[str]]:
        """Concatenate all features into a single array."""
        if not self._features:
            return np.empty((0, 0)), []
        
        all_features = []
        all_names = []
        
        for key in sorted(self._features.keys()):
            all_features.append(self._features[key])
            all_names.extend([name for name
                              in self._metadata[key]['feature_names']])
        
        return np.hstack(all_features, dtype=dtype), all_names
    
    def to_dataframe(self, 
                     feature_keys: Optional[Union[str, List[str]]] = None,
                     include_metadata: bool = False) -> pd.DataFrame:
        """Convert features to a pandas DataFrame.
        
        Args:
            feature_keys: Specific feature keys to include. If None, includes all.
            include_metadata: If True, adds metadata as DataFrame attributes.
            
        Returns:
            DataFrame with features as columns.
        """
        if not PANDAS_AVAILABLE:
            raise ModuleNotFoundError("In order to use `to_dataframe`, pandas "
                                      "needs to be installed.")
        if feature_keys is None:
            feature_keys = list(self._features.keys())
        elif isinstance(feature_keys, str):
            feature_keys = [feature_keys]
        
        if not feature_keys:
            return pd.DataFrame()
        
        # Collect features and column names
        feature_arrays = []
        column_names = []
        metadata_dict = {}
        
        for key in feature_keys:
            if key not in self._features:
                logger.warning(f"Feature key '{key}' not found, skipping")
                continue
            
            features = self._features[key]
            metadata = self._metadata[key]
            
            feature_arrays.append(features)
            
            # Generate column names
            if 'feature_names' in metadata:
                names = [f"{name}" for name in metadata['feature_names']]
            else:
                names = [f"{key}_{i}" for i in range(features.shape[1])]
            
            column_names.extend(names)
            
            # Collect metadata if requested
            if include_metadata:
                metadata_dict[key] = metadata.to_dict()
        
        # Create DataFrame
        if feature_arrays:
            combined_features = np.hstack(feature_arrays)
            df = pd.DataFrame(combined_features, columns=column_names)
            
            # Add metadata as attributes if requested
            if include_metadata:
                df.attrs['feature_metadata'] = metadata_dict
            
            return df
        else:
            return pd.DataFrame()
    
    def __dataframe__(self, 
                      nan_as_null: bool = True, 
                      allow_copy: bool = True) -> pd.DataFrame:
        """Implement the dataframe interchange protocol."""
        return self.to_dataframe()
    
    def __len__(self) -> int:
        return len(self._features)
    
    def __contains__(self, key: str) -> bool:
        return key in self._features
    
    def __iter__(self):
        return iter(self._features)

class BaseStorage(ABC):
    """Abstract class for storing objects."""
    
    def __init__(self, 
                 objects: Optional[Dict[str, List[Any]]] = None,
                 features: Optional[StorageFeatures] = None,
                 metadata: Optional[StorageMetadata] = None):
        
        if not objects:
            objects = {key: list() for key in self.required_object_keys}
        self._objects: Dict[str, List[Any]] = objects
        self._features = features or StorageFeatures()
        self._metadata = metadata or StorageMetadata()
        self._knn: Optional[NearestNeighbors] = None

        # TODO: Check object types as well using OBJECT_DTYPES
        
        # Initialize default object containers if not provided
        provided_required_keys = set(self._objects.keys()).intersection(
            self.required_object_keys)
        if len(provided_required_keys) == 0:
            for key in self.required_object_keys:
                self._objects[key] = []
            return
        
        ref_key = list(provided_required_keys)[0]
        if len(self._objects) < len(self.required_object_keys):
            for key in self.required_object_keys:
                if key not in self._objects:
                    self._objects[key] = [None] * len(self._objects[ref_key])
        
    # Properties
    @property
    def objects(self) -> Dict[str, List[Any]]:
        """Get objects (read-only)."""
        return {k: v.copy() for k, v in self._objects.items()}
    
    @property
    @abstractmethod
    def required_object_keys(self) -> List[str]:
        pass
    
    @property
    def features(self) -> StorageFeatures:
        """Get features container."""
        return self._features
    
    @property
    def metadata(self) -> StorageMetadata:
        """Get metadata."""
        return self._metadata
    
    @property
    def num_features(self) -> int:
        """Get total number of feature sets."""
        return len(self._features)
    
    @property
    @abstractmethod
    def save_dtypes(self) -> Dict[str, Type[np.dtype]]:
        pass
    
    # Core operations
    def sample(self, 
               n: int = 1, 
               object_keys: Optional[str | List[str]] = None) \
                -> Union[Any, List[Any]]:
        """Sample random objects."""
        if object_keys is None:
            object_keys = self.required_object_keys.copy()
        if isinstance(object_keys, str):
            object_keys = [object_keys]
        
        indices = random.choices(range(len(self)), k=n)
        if n == 1:
            return {k: v[indices[0]] 
                    for k, v in self._objects.items()}
        return {k: v[i] 
                for k, v in self._objects.items() 
                for i in indices}
    
    def extend(self, other: BaseStorage) -> None:
        """Extend this storage with another storage."""
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot extend {type(self)} with {type(other)}")
        
        if len(self) == 0:
            # If empty, just copy everything
            self._objects = {k: v.copy() for k, v in other._objects.items()}
            self._features = StorageFeatures()
            for key in other._features.keys():
                self._features.add_features(
                    key=key,
                    features=other._features.get_features(key),
                    featurizer=other._features.get_featurizer(key),
                    metadata=other._features.get_metadata(key).to_dict()
                )
            self._metadata = other._metadata.copy()
            return
        
        # Extend objects
        for key in self._objects.keys():
            if key in other._objects:
                self._objects[key].extend(other._objects[key])
            else:
                self._objects[key].extend([None] * len(other))
        
        # Extend features
        self._features.extend(other._features)
    
    def subset(self, indices: Union[List[int], np.ndarray]) -> BaseStorage:
        """Create a subset of this storage."""
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        
        subset_objects = {}
        for key, objects_list in self._objects.items():
            subset_objects[key] = [objects_list[i] for i in indices]
        
        subset_features = self._features.subset(indices)
        
        return self.__class__(
            *[subset_objects[key] for key in self.required_object_keys],
            features=subset_features,
            metadata=self._metadata.copy(),
        )
    
    # Save/Load methods
    def _append_to_objdb(self, db: h5py.File, key: str, data: List[Any]):
        if key not in db:
            db.create_dataset(
                name=key,
                data=data,
                dtype=self.save_dtypes[key],
                maxshape=(None,), # TODO: max shape as input
                compression="gzip",
                compression_opts=4,
                chunks=True
            ) # TODO: Have a `set_save_config` method to get these options
        else:
            db[key].resize(db[key].shape[0] + len(data), axis=0)
            db[key][-len(data):] = data

    def _append_to_featdb(self, db: h5py.File, key: str, data: np.ndarray):
        if key not in db:
            db.create_dataset(
                name=key,
                data=data,
                maxshape=(None, *data.shape[1:]), # TODO: max shape as input
                compression="gzip",
                compression_opts=4,
                chunks=True
            ) # TODO: Have a `set_save_config` method to get these options
        else:
            db[key].resize(db[key].shape[0] + data.shape[0], axis=0)
            db[key][-data.shape[0]:] = data

    def save_objects_to_file(self, db: h5py.File) -> None:
        """Get the data that should be saved for a list of objects."""
        for key, data in self.get_save_ready_objects().items():
            self._append_to_objdb(db, key, data)

            # Save object metadata directly to dataset
            if key in self._metadata:
                obj_metadata = self._metadata[key]
                if isinstance(obj_metadata, StorageMetadata):
                    for attr_key, value in obj_metadata.items():
                        db[key].attrs[attr_key] = value
    
    @abstractmethod
    def get_save_ready_objects(self) -> Dict[str, List[Any]]:
        pass

    def save_features_to_file(self, db: h5py.File) -> None:
        # Save features as a group
        if 'features' not in db:
            features_group = db.create_group('features')
        else:
            features_group = db['features']
        
        for feat_key in self._features.keys():
            features = self._features.get_features(feat_key)
            metadata = self._features.get_metadata(feat_key)

            self._append_to_featdb(features_group, feat_key, features)
            if isinstance(metadata, StorageMetadata):
                for key, val in metadata.items():
                    features_group[feat_key].attrs[key] = val

    def load_objects_from_file(self, 
                               db: h5py.File, 
                               indices: Optional[List[int] | np.ndarray | slice] = None,
                               append: bool = False) -> None:
        """Load objects from saved data."""
        if indices is None:
            indices = list(range(db[self.required_object_keys[0]].shape[0]))

        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        elif isinstance(indices, slice):
            start, stop, step = \
                indices.indices(db[self.required_object_keys[0]].shape[0])
            indices = [i for i in range(start, stop, step)]
        
        for key, objs in self.get_load_ready_objects(db, indices).items():
            if append:
                self._objects[key].extend(objs)
            else:
                self._objects[key] = objs
            
        for key in db.attrs:
            if key not in self._objects: # TODO: It should not be in the save keys not loaded ones
                if key not in self._metadata.keys():
                    self._metadata[key] = db.attrs[key]
    
    @abstractmethod
    def get_load_ready_objects(self,
                               db: h5py.File, 
                               indices: List[int] = None) \
                                -> Dict[str, List[Any]]:
        pass

    def load_features_from_file(self,
                                db: h5py.File,
                                indices: List[int] | np.ndarray | slice = None,
                                append: bool = False) -> None:
        if 'features' not in db:
            return
        
        features = StorageFeatures()
        
        feat_grp = db['features']
        for key in feat_grp:
            if indices is None:
                indices = list(range(feat_grp[key].shape[0]))

            features.add_features(key, 
                                  feat_grp[key][indices],
                                  metadata=None or dict(feat_grp[key].attrs))
        
        if append:
            self.features.extend(features)
        else:
            for key in features.keys():
                if key not in self.features:
                    self.features.add_features(
                        key,
                        features.get_features(key),
                        metadata=features.get_metadata(key)
                    )

    def write(self, 
              path: Union[str, Path], 
              mode: str = 'w') -> None:
        """Write storage to file. Mode can be 'w' (write) or 'a' (append)."""
        
        path = Path(path)

        with h5py.File(path, mode) as f:
            f.attrs['class_name'] = self.__class__.__name__

            # Save objects
            self.save_objects_to_file(f)

            # Save features
            self.save_features_to_file(f)

            # Save metadata
            for key in self._metadata:
                if key not in f.keys():
                    f.attrs[key] = self._metadata[key]
    
    @classmethod
    def load(cls, 
             path: Union[str, Path], 
             indices: Optional[Union[List[int], np.ndarray]] = None) \
                -> BaseStorage:
        """Load storage from file, optionally loading specified indices."""                
        path = Path(path)
        with h5py.File(path, "r") as f:
            storage = cls()
            storage.load_objects_from_file(f, indices)
            storage.load_features_from_file(f, indices, append=False)
            for k, v in f.attrs.items():
                storage._metadata[k] = v # TODO: check?
        return storage
        
    # Magic methods TODO: Needs extensive review
    def __len__(self) -> int:
        # Return length of first object list
        if self._objects:
            return len(list(self._objects.values())[0])
        return 0
    
    def __getitem__(self, 
                    idx: Union[int, List[int], np.ndarray]) \
                        -> Union[Any, Dict[str, Any]]:
        if isinstance(idx, int):
            if len(self._objects) == 1:
                return list(self._objects.values())[0][idx]
            else:
                return {key: obj_list[idx] 
                        for key, obj_list in self._objects.items()}
        elif isinstance(idx, (list, np.ndarray)):
            if len(self._objects) == 1:
                obj_list = list(self._objects.values())[0]
                return [obj_list[i] for i in idx]
            else:
                return {key: [obj_list[i] for i in idx] 
                        for key, obj_list in self._objects.items()}
    
    def __setitem__(self, idx: int, obj: Union[Any, Dict[str, Any]]) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in self._objects:
                    self._objects[key][idx] = value
        else:
            # Set in first object list
            self._objects[self.required_object_keys[0]][idx] = obj
    
    def __delitem__(self, idx: int) -> None:
        for obj_list in self._objects.values():
            if idx < len(obj_list):
                del obj_list[idx]
        
        # Remove from all feature sets
        for feat_key in self._features.keys():
            features = self._features.get_features(feat_key)
            if features is not None and idx < len(features):
                updated_features = np.delete(features, idx, axis=0)
                self._features.add_features(
                    key=feat_key,
                    features=updated_features,
                    featurizer=self._features.get_featurizer(feat_key),
                    metadata=self._features.get_metadata(feat_key).to_dict()
                )
    
    def __iter__(self):
        for i in range(len(self)):
            yield {
                key: objs[i]
                for key, objs in self._objects.items()
            }
    
    def __contains__(self, obj: Any) -> bool:
        return any(obj in obj_list for obj_list in self._objects.values())
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({len(self)} objects, "
                f"{self.num_features} feature sets, "
                f"{len(self.objects)} object types)")