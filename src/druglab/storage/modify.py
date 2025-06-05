from __future__ import annotations
from typing import List, Any, Dict, Optional
from abc import ABC
import logging
from functools import partial
from tqdm import tqdm
import mpire

from .base import BaseStorage
from .utils import _dict_to_list, _list_to_dict

logger = logging.getLogger(__name__)


class BaseStorageModifier(ABC):
    """Abstract base class for modifying BaseStorage objects.
    
    This class provides a framework for extracting data from storage,
    modifying objects based on that data, and applying changes back to storage.
    Supports both single-threaded and multi-threaded processing.
    """
    
    def __init__(self, n_processes: int = 1, chunk_size: Optional[int] = None):
        """Initialize the storage modifier.
        
        Args:
            n_processes: Number of processes to use for parallel processing.
                        If 1, uses single-threaded processing.
            chunk_size: Size of chunks for batch processing. If None, processes
                       all objects at once (or uses reasonable default for multiprocessing).
        """
        self.n_processes = max(1, n_processes)
        self.chunk_size = chunk_size
    
    def extract_context_data(self, storage: BaseStorage) -> Any:
        """Extract context data from storage that will be used for modification.
        
        This method should extract any global information needed for modifying
        individual objects (e.g., statistics, lookup tables, model parameters).
        
        Args:
            storage: The storage object to extract data from.
            
        Returns:
            Any data structure that will be passed to modify_objects_batch.
        """
        return None    
    
    def modify_objects(self, 
                       object_dict: Dict[str, Any],
                       context_data: Any) -> List[Dict[str, Any] | None]:
        """Modify a single object dict using the extracted context data.
        
        This is the main method that subclasses should implement to define
        their modification logic.
        
        Args:
            object_dict: single object dictionary to modify.
            context_data: Data extracted from extract_context_data.
            
        Returns:
            Modified objects dictionary. None if the modification fails.
        """
        return object_dict
    
    def modify_objects_wrapper(self,
                               object_dict: Dict[str, Any],
                               context_data: Any) \
                                -> List[Dict[str, Any] | None]:
        try:
            return self.modify_objects(object_dict, context_data)
        except Exception as e:
            logger.error(f"Failed to modify object: {e}")
            return None
    
    def apply_modifications(self, 
                            storage: BaseStorage,
                            modified_objects: List[Dict[str, Any]],
                            context_data: Any,
                            remove_fails: bool = False) -> None:
        """Apply any additional changes after object modification.
        
        This method can be overridden to perform additional operations
        after the main object modification is complete (e.g., updating
        features, metadata, or performing validation).
        
        Args:
            storage: The storage object being modified.
            modified_objects: The modified objects.
            context_data: The context data from extract_context_data.
        """
        success_ids = [i for i, objd in enumerate(modified_objects) 
                       if objd is None]
        if remove_fails:
            modified_objects = [mobj 
                                for mobj in modified_objects 
                                if mobj is not None]
        else:
            orig_objs = _dict_to_list(storage.objects)
            modified_objects = [
                mobj if mobj is not None else orig_objs[i]
                for i, mobj in enumerate(modified_objects)
            ]
        
        storage._objects = _list_to_dict(modified_objects)
        storage._features = storage.features.subset(success_ids)
    
    def modify(self, 
               storage: BaseStorage, 
               in_place: bool = True,
               remove_fails: bool = False) -> BaseStorage:
        """Main method to modify a storage object.
        
        Args:
            storage: The storage object to modify.
            in_place: If True, modifies the storage object in place.
                If False, creates a copy before modification.
            remove_fails: If True, it will remove any object that fails on the
                modifications
            
        Returns:
            The modified storage object.
        """
        if not in_place:
            # Create a copy of the storage
            storage = storage.subset(list(range(len(storage))))
        
        if len(storage) == 0:
            logger.warning("Storage is empty, nothing to modify.")
            return storage
        
        # Extract context data
        context_data = self.extract_context_data(storage)
        
        # Convert objects to list format
        objects_list = _dict_to_list(storage.objects)
        
        # Process objects
        print(f"Modifying {storage.__class__.__name__} "
              f"using {self.__class__.__name__}")
        if self.n_processes == 1:
            # Single-threaded processing
            modified_list = [self.modify_objects_wrapper(obj, context_data) 
                            for obj in tqdm(objects_list)]
        else:
            # Multi-threaded processing
            with mpire.WorkerPool(processes=self.n_processes) as pool:
                process_func = partial(self.modify_objects_wrapper, 
                                       context_data=context_data)
                modified_list = pool.map(process_func, objects_list)
        
        # Validate that we have the same number of objects
        if len(modified_list) != len(objects_list):
            raise ValueError(
                f"Number of modified objects ({len(modified_list)}) "
                f"doesn't match original ({len(objects_list)})"
            )
        
        # Apply modifications to storage
        self.apply_modifications(storage, modified_list, context_data)
        
        return storage
    
    def __call__(self, storage: BaseStorage, **kwargs) -> BaseStorage:
        """Make the modifier callable."""
        return self.modify(storage, **kwargs)
