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

OverlayBackend
    Zero-copy proxy backend with Copy-on-Write semantics.
"""

from .base import *
from .memory import *
from .overlay import *

__all__ = [
    # Base
    "BaseStorageBackend",
    "BaseMetadataMixin",
    "BaseObjectMixin",
    "BaseFeatureMixin",

    # Memory
    "MemoryMetadataMixin",
    "MemoryObjectMixin",
    "MemoryFeatureMixin",
    "EagerMemoryBackend",

    # Overlay
    "OverlayBackend",
]