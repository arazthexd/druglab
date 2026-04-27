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

BaseObjectStore
    Abstract interface for object handling in storage backends.

BaseFeatureStore
    Abstract interface for feature handling in storage backends.

BaseMetadataStore
    Abstract interface for metadata handling in storage backends.

CompositeStorageBackend
    Composite pattern for combining multiple domain stores.

MemoryMetadataStore
    In-memory metadata store.

MemoryObjectStore
    In-memory object store.

MemoryFeatureStore
    In-memory feature store.

EagerMemoryBackend
    Eager in-memory storage backend (Memory Metadata, Objects, and Features).

OverlayBackend
    Overlay backend mainly is used for creating views of other backends
"""

from .base import *
from .composite import *
from .overlay import *
from .memory import *

__all__ = [
    "BaseStorageBackend",
    "BaseObjectStore",
    "BaseMetadataStore",
    "BaseFeatureStore",
    "CompositeStorageBackend",
    "MemoryMetadataStore",
    "MemoryObjectStore",
    "MemoryFeatureStore",
    "EagerMemoryBackend",
    "OverlayBackend",
]