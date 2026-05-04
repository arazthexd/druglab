from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, List, TypeVar, Generic, Optional
from typing_extensions import Self

import numpy as np

from ..types import IdxLike

EngineIOT = TypeVar("EngineIOT")

class BaseEngine(ABC, Generic[EngineIOT]):
    """Every engine must know how to subset itself."""
    
    @abstractmethod
    def materialize(self, namespace: str, what: str, *args, **kwargs) -> EngineIOT:
        pass

    @abstractmethod
    def spawn_view(self, namespace: str, rows: IdxLike) -> Self:
        """Must return a proxy of this engine restricted to the specific rows."""
        pass

    @abstractmethod
    def write(self, namespace: str, what: str, data: EngineIOT, **kwargs) -> None:
        """Writes or appends data to the underlying storage."""
        pass

    @abstractmethod
    def export(self, target: Optional[str] = None, namespaces: Optional[List[str]] = None) -> Self:
        """
        Materializes the current state (or view) into a brand new, independent root engine.
        Should re-align row indices to guarantee synchronization across all engines.
        """
        pass