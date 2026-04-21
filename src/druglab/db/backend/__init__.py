"""
druglab.db.backend
~~~~~~~~~~~~~~~~~~~
Storage backend architecture for DrugLab tables.

This module provides the Strategy/Repository Pattern for separating
data representation from storage mechanisms.

Classes
-------
BaseStorageBackend
    Abstract unified interface for all storage backends.

MemoryMetadataMixin
    In-memory metadata management (Pandas DataFrame).

MemoryObjectMixin
    In-memory object management (Python list).

MemoryFeatureMixin
    In-memory feature management (dict of NumPy arrays).

EagerMemoryBackend
    Fully in-memory concrete backend combining all memory mixins.
"""

from druglab.db.backend.base import (
    BaseObjectMixin,
    BaseMetadataMixin,
    BaseFeatureMixin,
    BaseStorageBackend,
    INDEX_LIKE,
    RowSelection,
    normalize_row_index,
    coerce_bool_mask,
    validate_take_index,
)
from druglab.db.backend.memory import (
    MemoryMetadataMixin,
    MemoryObjectMixin,
    MemoryFeatureMixin,
    EagerMemoryBackend,
    _resolve_idx,
)

__all__ = [
    "BaseObjectMixin",
    "BaseMetadataMixin",
    "BaseFeatureMixin",
    "BaseStorageBackend",
    "INDEX_LIKE",
    "RowSelection",
    "normalize_row_index",
    "coerce_bool_mask",
    "validate_take_index",
    "MemoryMetadataMixin",
    "MemoryObjectMixin",
    "MemoryFeatureMixin",
    "EagerMemoryBackend",
    "_resolve_idx",
]