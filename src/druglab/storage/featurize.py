from __future__ import annotations
from typing import List, Any, Dict, Optional, Type
from abc import ABC, abstractmethod
import logging
from functools import partial
import numpy as np
from tqdm import tqdm
import mpire
import time

from ..featurize import BaseFeaturizer
from .base import BaseStorage
from .utils import _dict_to_list

logger = logging.getLogger(__name__)

class StorageFeaturizer(ABC):
    """Abstract base class for featurizing BaseStorage objects.
    
    This class provides a framework for extracting features from storage objects
    and adding them to the storage's feature container. Supports both 
    single-threaded and multi-threaded processing.
    """
    
    def __init__(self, 
                 feature_key: str,
                 n_processes: int = 1, 
                 dtype: Optional[Type[np.dtype]] = None):
        """Initialize the featurizer.
        
        Args:
            feature_key: Key to store the computed features under.
            n_processes: Number of processes to use for parallel processing.
                If 1, uses single-threaded processing.
            dtype: Data type for the computed features.
        """
        self.feature_key = feature_key
        self.n_processes = max(1, n_processes)
        self.dtype = dtype or np.float32
        self._fnames: Optional[List[str]] = None

        self._name = None
    
    @property
    def fnames(self) -> List[str]:
        """Feature names for the computed features."""
        if self._fnames is None:
            raise ValueError("Feature names not set. Call set_feature_names() first.")
        return self._fnames
    
    def set_feature_names(self, names: List[str]) -> None:
        """Set the feature names."""
        self._fnames = names.copy()
    
    def extract_context_data(self, storage: BaseStorage) -> Any:
        """Extract context data from storage that will be used for featurization.
        
        This method should extract any global information needed for computing
        features from individual objects (e.g., vocabularies, normalization 
        parameters, model weights).
        
        Args:
            storage: The storage object to extract data from.
            
        Returns:
            Any data structure that will be passed to compute_features.
        """
        return None
    
    @abstractmethod
    def compute_features(self, 
                         object_dict: Dict[str, Any],
                         context_data: Any) -> Optional[np.ndarray]:
        """Compute features for a single object dict using extracted context.
        
        This is the main method that subclasses should implement to define
        their featurization logic.
        
        Args:
            object_dict: Single object dictionary to compute features for.
            context_data: Data extracted from extract_context_data.
            
        Returns:
            Feature vector as numpy array, or None if featurization fails.
        """
        pass
    
    def compute_features_wrapper(self,
                                 object_dict: Dict[str, Any],
                                 context_data: Any) -> Optional[np.ndarray]:
        """Wrapper around compute_features with error handling."""
        try:
            features = self.compute_features(object_dict, context_data)
            if features is not None:
                features = features.astype(self.dtype)
            return features
        except Exception as e:
            logger.error(f"Failed to compute features for object: {e}")
            return None
    
    def validate_features(self, 
                          features_list: List[Optional[np.ndarray]],
                          remove_fails: bool = False) -> np.ndarray:
        """Validate and process the computed features.
        
        Args:
            features_list: List of computed feature arrays (some may be None).
            remove_fails: If True, removes failed computations and returns
                         indices of successful ones.
            
        Returns:
            Stacked feature array, with failed computations either removed
            or replaced with nans.
        """
        # Count successful computations
        successful_features = [f for f in features_list if f is not None]
        
        if not successful_features:
            raise ValueError("No features were successfully computed")
        
        # Get feature dimension from first successful computation
        feature_dim = successful_features[0].shape[-1]
        
        if remove_fails:
            # Return only successful features
            return np.vstack(successful_features)
        else:
            # Replace None with nan feats
            processed_features = []
            for features in features_list:
                if features is not None:
                    processed_features.append(features)
                else:
                    processed_features.append(np.full(feature_dim, 
                                                      np.nan, 
                                                      dtype=self.dtype))
            
            return np.vstack(processed_features)
    
    def get_success_indices(self, 
                            features_list: List[Optional[np.ndarray]]) \
                                -> List[int]:
        """Get indices of successfully computed features."""
        return [i for i, f in enumerate(features_list) if f is not None]
    
    def apply_featurization(self, 
                            storage: BaseStorage,
                            features_array: np.ndarray,
                            context_data: Any,
                            success_indices: Optional[List[int]] = None) -> None:
        """Apply the computed features to the storage object.
        
        This method can be overridden to perform additional operations
        after feature computation is complete.
        
        Args:
            storage: The storage object being featurized.
            features_array: The computed features array.
            context_data: The context data from extract_context_data.
            success_indices: Indices of objects with successful feature 
                computation.
        """
        # Create metadata for the features
        metadata = {
            'n_processes': self.n_processes,
        }
        
        if success_indices is not None:
            metadata['success_indices'] = success_indices
            metadata['success_rate'] = len(success_indices) / len(storage)
        
        # Add features to storage
        storage.features.add_features(
            key=self.feature_key,
            features=features_array,
            dtype=self.dtype,
            featurizer=self,
            metadata=metadata
        )
    
    def featurize(self, 
                 storage: BaseStorage,
                 remove_fails: bool = False,
                 overwrite: bool = False) -> BaseStorage:
        """Main method to featurize a storage object.
        
        Args:
            storage: The storage object to featurize.
            remove_fails: If True, removes objects that fail featurization.
            overwrite: If True, overwrites existing features with the same key.
            
        Returns:
            The storage object with added features.
        """
        if len(storage) == 0:
            logger.warning("Storage is empty, nothing to featurize.")
            return storage
        
        # Check if features already exist
        if self.feature_key in storage.features and not overwrite:
            logger.warning(f"Features '{self.feature_key}' already exist. "
                          f"Use overwrite=True to replace them.")
            return storage
        
        # Extract context data
        context_data = self.extract_context_data(storage)
        
        # Convert objects to list format
        objects_list = _dict_to_list(storage.objects)
        
        # Compute features
        print(f"Computing features for {storage.__class__.__name__} "
              f"using {self._name or self.__class__.__name__}")
        time.sleep(0.1)
        
        if self.n_processes == 1:
            # Single-threaded processing
            features_list = [self.compute_features_wrapper(obj, context_data) 
                             for obj in tqdm(objects_list)]
        else:
            # Multi-threaded processing
            with mpire.WorkerPool(processes=self.n_processes) as pool:
                process_func = partial(self.compute_features_wrapper, 
                                       context_data=context_data)
                features_list = pool.map(process_func, objects_list)
        
        # Validate features
        success_indices = self.get_success_indices(features_list) \
            if remove_fails else None
        features_array = self.validate_features(features_list, remove_fails)
        
        # If removing fails, also subset the storage
        if remove_fails and success_indices:
            if len(success_indices) < len(storage):
                logger.info(f"Removing {len(storage) - len(success_indices)} "
                            f"objects that failed featurization")
                storage = storage.subset(success_indices)
        
        # Apply featurization to storage
        self.apply_featurization(storage, features_array, 
                                 context_data, success_indices)
        
        return storage
    
    def __call__(self, storage: BaseStorage, **kwargs) -> BaseStorage:
        """Make the featurizer callable."""
        return self.featurize(storage, **kwargs)

class BasicStorageFeaturizerWrapper(StorageFeaturizer):
    def __init__(self,  
                 featurizer: BaseFeaturizer,
                 input_keys: List[str],
                 feature_key: Optional[str] = None,
                 n_processes = 1):
        super().__init__(feature_key or featurizer.name, 
                         n_processes, 
                         featurizer.dtype)
        self.featurizer = featurizer
        self.input_keys = input_keys

        self._name = self.featurizer.name
    
    def compute_features(self, object_dict, context_data):
        return self.featurizer.featurize(
            *[object_dict[key] for key in self.input_keys]
        )
    
    @property
    def fnames(self):
        return self.featurizer.fnames

class CompositeFeaturizer(StorageFeaturizer): # TODO: Extensive review
    """Featurizer that combines multiple featurizers.
    
    This featurizer runs multiple sub-featurizers and concatenates their 
        outputs.
    """
    
    def __init__(self, 
                 featurizers: List[StorageFeaturizer],
                 feature_key: str,
                 n_processes: int = 1,
                 dtype: Optional[Type[np.dtype]] = None):
        """Initialize the composite featurizer.
        
        Args:
            featurizers: List of featurizers to combine.
            feature_key: Key to store the combined features under.
            n_processes: Number of processes (applied to each sub-featurizer).
            dtype: Data type for the combined features.
        """
        super().__init__(feature_key, n_processes, None, dtype)
        self.featurizers = featurizers
        
        # Set n_processes for all sub-featurizers
        for featurizer in self.featurizers:
            featurizer.n_processes = n_processes
    
    def extract_context_data(self, storage: BaseStorage) -> Dict[str, Any]:
        """Extract context data from all sub-featurizers."""
        return {
            f"featurizer_{i}": featurizer.extract_context_data(storage)
            for i, featurizer in enumerate(self.featurizers)
        }
    
    def compute_features(self, 
                         object_dict: Dict[str, Any],
                         context_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Compute features using all sub-featurizers and concatenate."""
        feature_parts = []
        
        for i, featurizer in enumerate(self.featurizers):
            sub_context = context_data[f"featurizer_{i}"]
            features = featurizer.compute_features(object_dict, sub_context)
            
            if features is None:
                return None  # If any sub-featurizer fails, the whole thing fails
            
            feature_parts.append(features)
        
        return np.concatenate(feature_parts)
    
    def apply_featurization(self, 
                            storage: BaseStorage,
                            features_array: np.ndarray,
                            context_data: Any,
                            success_indices: Optional[List[int]] = None) \
                                -> None:
        """Apply featurization and set combined feature names."""
        # Set combined feature names
        combined_names = []
        for i, featurizer in enumerate(self.featurizers):
            if hasattr(featurizer, 'fnames'):
                combined_names.extend([f"{featurizer.feature_key}_{name}" 
                                     for name in featurizer.fnames])
            else:
                # Generate default names if featurizer doesn't have fnames
                n_features = len(featurizer.compute_features(
                    _dict_to_list(storage.objects)[0], 
                    context_data[f"featurizer_{i}"]
                ))
                combined_names.extend([f"{featurizer.feature_key}_{i}" 
                                      for i in range(n_features)])
        
        self.set_feature_names(combined_names)
        
        # Call parent method
        super().apply_featurization(storage, features_array, 
                                    context_data, success_indices)