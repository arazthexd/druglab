from __future__ import annotations
from typing import List, Any, Dict, Optional, Callable, Union
from abc import ABC, abstractmethod
import logging
from functools import partial
import time
import numpy as np
from tqdm import tqdm
import mpire

from .base import BaseStorage
from .utils import _dict_to_list

logger = logging.getLogger(__name__)


class BaseStorageFilter(ABC):
    """Abstract base class for filtering BaseStorage objects.
    
    This class provides a framework for extracting context data from storage,
    filtering objects based on that data, and returning a subset of the storage.
    Supports both single-threaded and multi-threaded processing.
    """
    
    def __init__(self, 
                 n_processes: int = 1):
        """Initialize the storage filter.
        
        Args:
            n_processes: Number of processes to use for parallel processing.
                If 1, uses single-threaded processing.
        """
        self.n_processes = max(1, n_processes)
        self._name = None
    
    def extract_context_data(self, storage: BaseStorage) -> Any:
        """Extract context data from storage that will be used for filtering.
        
        This method should extract any global information needed for filtering
        individual objects (e.g., statistics, thresholds, lookup tables).
        
        Args:
            storage: The storage object to extract data from.
            
        Returns:
            Any data structure that will be passed to should_keep.
        """
        return None
    
    @abstractmethod
    def should_keep(self, 
                    object_dict: Dict[str, Any],
                    idx: int, 
                    context_data: Any) -> bool:
        """Determine whether to keep a single object dict.
        
        This is the main method that subclasses should implement to define
        their filtering logic.
        
        Args:
            object_dict: Single object dictionary to evaluate.
            idx: Index of the object currently checking.
            context_data: Data extracted from extract_context_data.
            
        Returns:
            True if the object should be kept, False otherwise.
        """
        pass
    
    def should_keep_wrapper(self,
                            object_dict: Dict[str, Any],
                            idx: int,
                            context_data: Any) -> bool:
        """Wrapper around should_keep with error handling."""

        try:
            return self.should_keep(object_dict, idx, context_data)
        except Exception as e:
            logger.error(f"Failed to evaluate object for filtering: {e}")
            return False  # Default to rejecting objects that cause errors
    
    def apply_filter_results(self, 
                             storage: BaseStorage,
                             keep_flags: List[bool],
                             context_data: Any) -> List[int]:
        """Apply additional operations after filtering is complete.
        
        This method can be overridden to perform additional operations
        after the main filtering is complete (e.g., logging statistics,
        updating metadata).
        
        Args:
            storage: The original storage object.
            keep_flags: Boolean flags indicating which objects to keep.
            context_data: The context data from extract_context_data.
            
        Returns:
            The passing indices
        """
        # Get indices of objects to keep
        keep_indices = [i for i, keep in enumerate(keep_flags) if keep]
        
        # Log filtering statistics
        n_original = len(storage)
        n_kept = len(keep_indices)
        n_removed = n_original - n_kept
        
        logger.info(f"Filtering completed: kept {n_kept}/{n_original} objects "
                   f"({n_kept/n_original*100:.1f}%), removed {n_removed}")
        
        # Return subset
        if n_kept == 0:
            logger.warning("All objects were filtered out!")
            # Return empty storage of the same type
            return keep_indices
        elif n_kept == n_original:
            logger.info("No objects were filtered out")
            return keep_indices
        else:
            return keep_indices
    
    def filter(self, 
               storage: BaseStorage,
               return_storage: bool = False) -> List[int] | BaseStorage:
        """Main method to filter a storage object.
        
        Args:
            storage: The storage object to filter.
            
        Returns:
            Indices that should be kept OR a new storage
        """
        if len(storage) == 0:
            logger.warning("Storage is empty, nothing to filter.")
            return storage
        
        # Extract context data
        context_data = self.extract_context_data(storage)
        
        # Convert objects to list format
        objects_list = _dict_to_list(storage.objects)
        
        # Filter objects
        print(f"Filtering {storage.__class__.__name__} "
              f"using {self._name or self.__class__.__name__}")
        time.sleep(0.1)
        
        if self.n_processes == 1:
            # Single-threaded processing
            keep_flags = [self.should_keep_wrapper(obj, i, context_data) 
                          for i, obj in enumerate(tqdm(objects_list))]
        else:
            # Multi-threaded processing
            with mpire.WorkerPool(n_jobs=self.n_processes,
                                  use_dill=True) as pool:
                process_func = partial(self.should_keep_wrapper, 
                                       context_data=context_data)
                objects_list = [(o, i) for i, o in enumerate(objects_list)]
                keep_flags = pool.map(lambda o, i: process_func(o, i), 
                                      objects_list,
                                      progress_bar=True)
        
        # Apply filter results and return filtered storage
        indices = self.apply_filter_results(storage, keep_flags, context_data)
        if return_storage:
            return storage.subset(indices)
        else:
            return indices
    
    def __call__(self, storage: BaseStorage) -> BaseStorage:
        """Make the filter callable."""
        return self.filter(storage)


class CustomFuncFilter(BaseStorageFilter):
    """Filter that uses a custom function to determine which objects to keep.
    
    This filter applies a user-provided function to each object dictionary
    to determine whether it should be kept or filtered out.
    """
    
    def __init__(self,
                 filter_func: Callable[[Any], bool],
                 input_keys: List[str],
                 n_processes: int = 1,
                 name: Optional[str] = None):
        """Initialize the custom function filter.
        
        Args:
            filter_func: Function that takes an object dictionary and returns
                        True if the object should be kept, False otherwise.
            n_processes: Number of processes to use for parallel processing.
            name: Optional name for the filter (used in logging).
        """
        super().__init__(n_processes)
        self.filter_func = filter_func
        self.input_keys = input_keys
        self._name = name or "CustomFuncFilter"
    
    def should_keep(self, 
                    object_dict: Dict[str, Any],
                    idx: int,
                    context_data: Any) -> bool:
        """Apply the custom function to determine if object should be kept."""
        return self.filter_func(*[object_dict[key] for key in self.input_keys])
    

class FeatureBasedFilter(BaseStorageFilter):
    """Filter objects based on their computed features.
    
    This filter applies conditions to feature vectors to determine which
    objects to keep.
    """
    
    def __init__(self,
                 feature_key: str,
                 condition_func: Callable[[np.ndarray], bool],
                 name: str = None,
                 n_processes: int = 1):
        """Initialize the feature-based filter.
        
        Args:
            feature_key: Key of the features to use for filtering.
            condition_func: Function that takes a feature vector (np.ndarray)
                           and returns True if the object should be kept.
            n_processes: Number of processes to use.
        """
        super().__init__(n_processes)
        self.feature_key = feature_key
        self.condition_func = condition_func
        self._name = name or f"FeatureFilter({feature_key})"
    
    def extract_context_data(self, storage: BaseStorage) -> np.ndarray:
        """Extract the feature array from storage."""
        if self.feature_key not in storage.features:
            raise ValueError(f"Feature key '{self.feature_key}' "
                             "not found in storage")
        
        return storage.features.get_features(self.feature_key)
    
    def should_keep(self, 
                    object_dict: Dict[str, Any],
                    idx: int,
                    context_data: np.ndarray) -> bool:
        """Apply the condition to the object's features."""
        if idx >= context_data.shape[0]:
            logger.warning(f"Object index {idx} "
                           "out of bounds for features")
            return False
        
        feature_vector = context_data[idx]
        return self.condition_func(feature_vector)

class CompositeFilter(BaseStorageFilter): # TODO: Extensive testing...
    """Filter that combines multiple sub-filters using logical operations.
    
    This filter runs multiple sub-filters and combines their results using
    AND, OR, or custom logic.
    """
    
    def __init__(self,
                 filters: List[BaseStorageFilter],
                 operation: str = "and",
                 n_processes: int = 1):
        """Initialize the composite filter.
        
        Args:
            filters: List of filters to combine.
            operation: How to combine filter results. Options:
                      - "and": Keep objects that pass ALL filters
                      - "or": Keep objects that pass ANY filter
                      - "majority": Keep objects that pass majority of filters
            n_processes: Number of processes (applied to each sub-filter).
        """
        super().__init__(n_processes)
        self.filters = filters
        self.operation = operation.lower()
        
        if self.operation not in ["and", "or", "majority"]:
            raise ValueError(f"Unknown operation: {operation}. "
                           f"Must be 'and', 'or', or 'majority'")
        
        # Set n_processes for all sub-filters
        for filter_obj in self.filters:
            filter_obj.n_processes = n_processes
        
        self._name = f"CompositeFilter({operation})"
    
    def extract_context_data(self, storage: BaseStorage) -> Dict[str, Any]:
        """Extract context data from all sub-filters."""
        return {
            f"filter_{i}": filter_obj.extract_context_data(storage)
            for i, filter_obj in enumerate(self.filters)
        }
    
    def should_keep(self, 
                    object_dict: Dict[str, Any],
                    context_data: Dict[str, Any]) -> bool:
        """Apply all sub-filters and combine results."""
        results = []
        
        for i, filter_obj in enumerate(self.filters):
            sub_context = context_data[f"filter_{i}"]
            result = filter_obj.should_keep(object_dict, sub_context)
            results.append(result)
        
        # Apply logical operation
        if self.operation == "and":
            return all(results)
        elif self.operation == "or":
            return any(results)
        elif self.operation == "majority":
            return sum(results) > len(results) / 2
        else:
            raise ValueError(f"Unknown operation: {self.operation}")


class KeyBasedFilter(BaseStorageFilter): # TODO: Extensive Testing
    """Filter objects based on values in specific keys of the object dictionary.
    
    This is a convenience filter for common filtering operations based on
    object attributes.
    """
    
    def __init__(self,
                 key: str,
                 condition: Union[Callable[[Any], bool], Any, List[Any]],
                 condition_type: str = "function",
                 n_processes: int = 1,
                 chunk_size: Optional[int] = None):
        """Initialize the key-based filter.
        
        Args:
            key: Key in the object dictionary to check.
            condition: The condition to apply. Can be:
                      - A function that takes the value and returns bool
                      - A single value for equality check
                      - A list of values for membership check
            condition_type: Type of condition. Options:
                           - "function": condition is a callable
                           - "equals": keep if value equals condition
                           - "not_equals": keep if value doesn't equal condition
                           - "in": keep if value is in condition list
                           - "not_in": keep if value is not in condition list
                           - "greater": keep if value > condition
                           - "less": keep if value < condition
                           - "greater_equal": keep if value >= condition
                           - "less_equal": keep if value <= condition
            n_processes: Number of processes to use.
            chunk_size: Size of chunks for batch processing.
        """
        super().__init__(n_processes, chunk_size)
        self.key = key
        self.condition = condition
        self.condition_type = condition_type.lower()
        
        valid_types = ["function", "equals", "not_equals", "in", "not_in", 
                      "greater", "less", "greater_equal", "less_equal"]
        if self.condition_type not in valid_types:
            raise ValueError(f"Unknown condition_type: {condition_type}. "
                           f"Must be one of {valid_types}")
        
        self._name = f"KeyBasedFilter({key}_{condition_type})"
    
    def should_keep(self, 
                    object_dict: Dict[str, Any],
                    context_data: Any) -> bool:
        """Apply the key-based condition."""
        if self.key not in object_dict:
            logger.warning(f"Key '{self.key}' not found in object, filtering out")
            return False
        
        value = object_dict[self.key]
        
        if self.condition_type == "function":
            return self.condition(value)
        elif self.condition_type == "equals":
            return value == self.condition
        elif self.condition_type == "not_equals":
            return value != self.condition
        elif self.condition_type == "in":
            return value in self.condition
        elif self.condition_type == "not_in":
            return value not in self.condition
        elif self.condition_type == "greater":
            return value > self.condition
        elif self.condition_type == "less":
            return value < self.condition
        elif self.condition_type == "greater_equal":
            return value >= self.condition
        elif self.condition_type == "less_equal":
            return value <= self.condition
        else:
            raise ValueError(f"Unknown condition_type: {self.condition_type}")

