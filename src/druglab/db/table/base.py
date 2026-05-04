from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Generic, List, Optional, Tuple, TypeVar, Union, Iterable, Any, FrozenSet, Sequence
from typing_extensions import Self


import numpy as np
import pandas as pd

from ..db import BaseDB
from ..types import IdxLike
from .history import History

ItemT = TypeVar("ItemT") 

class BaseTable(ABC, Generic[ItemT]):
    """Abstract base class for all DrugLab table types."""

    DEFAULT_W2E: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self, 
        db: BaseDB, 
        name: str, 
        _w2e: Optional[Dict[str, str]] = None,
        _history: Optional[History] = None
    ) -> None:
        self._db = db
        self._name = name
        
        # Ensure history carries over when spawning views
        self._history = _history if _history is not None else History()

        if _w2e is not None:
            self._w2e = _w2e.copy()
        else:
            self._w2e = self.DEFAULT_W2E.copy()

        for engname in set(self._w2e.values()):
            self._db.request_engine(engname) # Make sure engines exist

    # ------------------------------------------------------------------
    # The Write Path (Abstract)
    # ------------------------------------------------------------------

    @abstractmethod
    def extend(
        self, 
        items: Iterable[ItemT],
        **kwargs
    ) -> None:
        """
        Translates domain objects (e.g., RDKit Mols) into Engine-friendly 
        formats (DataFrames/Arrays) and writes them to the underlying storage.
        """
        pass

    def _write_to_engine(self, what: str, data: Any, **kwargs) -> None:
        """
        Helper method for subclasses to quickly route data to the correct engine 
        during the extend() process.
        """
        engine = self._db.request_engine(self._w2e[what])
        engine.write(namespace=self._name, what=what, data=data, **kwargs)
        
    # ------------------------------------------------------------------
    # The Read Path (Materialization & Views)
    # ------------------------------------------------------------------

    def materialize(self, what: str, *args, **kwargs) -> Any:
        engine = self._db.request_engine(self._w2e[what])
        return engine.materialize(self._name, what, *args, **kwargs)

    def view(self, rows: Optional[IdxLike] = None, **kwargs) -> Self:
        """
        Spawns a new, read-only Table connected to a restricted Database proxy.
        Accepts rows and any additional filtering kwargs (columns, features, etc.).
        """

        # We tell the DB to spawn a restricted proxy, passing down ALL filters.
        restricted_db = self._db.spawn_restricted_view(
            namespace=self._name, 
            rows=rows,
            **kwargs
        )

        # We wrap a new instance of the exact same table class around this new DB
        return self.__class__(
            db=restricted_db, 
            name=self._name, 
            _w2e=self._w2e,
            _history=self._history
        )
    
    # ------------------------------------------------------------------
    # Python Dunder Methods
    # ------------------------------------------------------------------

    @abstractmethod
    def __len__(self) -> int:
        """
        Every table should know its length. Subclasses should implement this 
        by querying their primary engine (usually the metadata table).
        """
        pass