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
"""

from druglab.db.backends.base import BaseStorageBackend, INDEX_LIKE

__all__ = [
    "BaseStorageBackend",
    "INDEX_LIKE",
]