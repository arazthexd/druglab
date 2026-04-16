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

    def __init__(self, max_size: int = 1000):
        self._store = {}
        self._order = []
        self.max_size = max_size

    def set(self, key, value):
        if key not in self._store:
            self._order.append(key)

        self._store[key] = value

        if len(self._order) > self.max_size:
            oldest = self._order.pop(0)
            del self._store[oldest]

    def get(self, key: str) -> Any:
        return self._store.get(key)
    
    def clear(self) -> None:
        self._store.clear()
        self._order.clear()

# Global memory cache (can be overridden by specific blocks)
default_cache = DictCache()