from typing import List, Dict, Any

import dill
import logging

logger = logging.getLogger(__name__)

def _dict_to_list(objects_dict: Dict[str, List[Any]]) \
    -> List[Dict[str, Any]]:
    """Convert dictionary of lists to list of dictionaries.
    
    Transforms Dict[str, List[Any]] -> List[Dict[str, Any]]
    """
    if not objects_dict:
        return []
    
    # Get length from first non-empty list
    length = next((len(obj_list) 
                   for obj_list in objects_dict.values() if obj_list), 0)
    
    # Convert to list of dictionaries
    return [{key: obj_list[i] if i < len(obj_list) else None
            for key, obj_list in objects_dict.items()}
            for i in range(length)]

def _list_to_dict(objects_list: List[Dict[str, Any]],
                  required_keys: List[str] = None) \
                    -> Dict[str, List[Any]]:
    """Convert list of dictionaries to dictionary of lists
    
    Transforms List[Dict[str, Any]] -> Dict[str, List[Any]]
    """
    if not objects_list:
        return {key: [] for key in required_keys}
    
    # Use list comprehension to build the dictionary quickly
    objects_dict = {key: [obj_dict.get(key) for obj_dict in objects_list]
                    for key in (required_keys or objects_list[0].keys())}
    
    return objects_dict

def serialize_objects(objects: List[Any]) -> List[bytes]:
    """Serialize objects using dill.
    
    Args:
        objects: List of objects
        
    Returns:
        List of serialized objects as bytes
    """
    serialized = []
    for obj in objects:
        try:
            serialized.append(dill.dumps(obj, 0))
        except Exception as e:
            logger.warning(f"Failed to serialize object: {e}")
            serialized.append(b"")
    return serialized

def deserialize_objects(serialized_objs: List[bytes]) -> List[Any]:
    """Deserialize objects from bytes.
    
    Args:
        serialized_objs: List of serialized objects as bytes
        
    Returns:
        List of objects
    """
    objs = []
    for serialized in serialized_objs:
        if not serialized:
            objs.append(None)
            continue
        try:
            obj = dill.loads(serialized)
            objs.append(obj)
        except Exception as e:
            logger.warning(f"Failed to deserialize route: {e}")
            objs.append(None)
    return objs