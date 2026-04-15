"""
druglab.db.backends
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

from druglab.db.backend.base import BaseStorageBackend, INDEX_LIKE
from druglab.db.backend.memory import (
    MemoryMetadataMixin,
    MemoryObjectMixin,
    MemoryFeatureMixin,
    EagerMemoryBackend,
)

__all__ = [
    "BaseStorageBackend",
    "INDEX_LIKE",
    "MemoryMetadataMixin",
    "MemoryObjectMixin",
    "MemoryFeatureMixin",
    "EagerMemoryBackend",
]