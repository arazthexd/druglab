"""
druglab.pipe.cache
~~~~~~~~~~~~~~~~~~
Caching interfaces for pipeline blocks.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseCache(ABC):
    """Abstract interface for block caching."""

    @abstractmethod
    def get(self, key: str) -> Any:
        """Retrieve an item from cache, return None if not found."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Store an item in the cache."""
        pass


class DictCache(BaseCache):
    """A simple in-memory dictionary cache."""

    def __init__(self):
        self._store = {}

    def get(self, key: str) -> Any:
        return self._store.get(key)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

# Global memory cache (can be overridden by specific blocks)
default_cache = DictCache()