"""
druglab.pipe.cache
~~~~~~~~~~~~~~~~~~
Caching interfaces for pipeline blocks.
"""

from abc import ABC, abstractmethod
from typing import Any
from collections import OrderedDict


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
        if max_size < 1:
            raise ValueError("max_size must be at least 1.")
        self._store = OrderedDict()
        self.max_size = max_size

    def set(self, key, value):
        if key in self._store:
            self._store.move_to_end(key)

        self._store[key] = value

        if len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def get(self, key: str) -> Any:
        return self._store.get(key)
    
    def clear(self) -> None:
        self._store.clear()

# Global memory cache (can be overridden by specific blocks)
default_cache = DictCache()