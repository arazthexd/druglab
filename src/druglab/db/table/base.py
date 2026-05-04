from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Generic, Iterable, List, Optional, TypeVar
from typing_extensions import Self

import numpy as np
import pandas as pd

from ..db import BaseDB
from ..db.proxy import RestrictedDBProxy
from ..types import ColsLike, IdxLike
from .history import History, HistoryEntry

ItemT = TypeVar("ItemT")


class BaseTable(ABC, Generic[ItemT]):
    """Abstract base class for all DrugLab table types.

    This class provides a unified interface for writing to and reading from 
    multi-engine storage systems. It implements a template method pattern for 
    data extension to ensure automatic history tracking and supports restricted 
    views via database proxies.

    Attributes:
        DEFAULT_W2E (ClassVar[Dict[str, str]]): A mapping of data components 
            (e.g., "metadata") to their storage engine names (e.g., "duckdb"). 
            Subclasses should define this at the class level.

    Note:
        Subclasses must implement the private `_extend_impl` method instead of 
        overriding `extend` to maintain history logging integrity.
    """

    DEFAULT_W2E: ClassVar[Dict[str, str]] = {}

    def __init__(
        self,
        db: BaseDB,
        name: str,
        _w2e: Optional[Dict[str, str]] = None,
        _history: Optional[History] = None,
    ) -> None:
        """Initializes a BaseTable instance.

        Args:
            db: The database backend or proxy to use.
            name: The unique namespace/name for this table in the database.
            _w2e: Internal mapping for components to engines. If None, 
                defaults to a copy of `DEFAULT_W2E`.
            _history: Shared history log. If None, a new History is created.
        """
        self._db = db
        self._name = name
        self._history = _history if _history is not None else History()

        # Work from an instance copy to prevent class-level mutation
        self._w2e: Dict[str, str] = (
            _w2e.copy() if _w2e is not None else self.DEFAULT_W2E.copy()
        )

        # Fail fast if required engines are not available
        for eng_name in set(self._w2e.values()):
            self._db.request_engine(eng_name)

    def extend(self, items: Iterable[Any], **kwargs) -> None:
        """Adds items to the table and records the operation in the history.

        This is a template method. It calculates the row delta and appends a 
        `HistoryEntry` automatically after calling the subclass implementation.

        Args:
            items: An iterable of domain objects to be added.
            **kwargs: Additional configuration passed to the engine.

        Important:
            Do not override this method. Override `_extend_impl` instead.
        """
        rows_before = len(self)
        self._extend_impl(items, **kwargs)
        rows_after = len(self)

        self._history.append(
            HistoryEntry.now(
                operation=f"{type(self).__name__}.extend",
                config=kwargs,
                rows_in=rows_before,
                rows_out=rows_after,
            )
        )

    @abstractmethod
    def _extend_impl(self, items: Iterable[Any], **kwargs) -> None:
        """Subclass implementation for writing data to engines.

        Args:
            items: Domain objects to process and store.
            **kwargs: Implementation-specific write arguments.
        """
        pass

    def _write_to_engine(self, what: str, data: Any, **kwargs) -> None:
        """Routes data to the specific engine mapped to a component.

        Args:
            what: The component name (key in w2e mapping).
            data: The data to be written.
            **kwargs: Arguments passed to the engine's write method.
        """
        engine = self._db.request_engine(self._w2e[what])
        engine.write(namespace=self._name, what=what, data=data, **kwargs)

    def materialize(
        self,
        what: str,
        rows: Optional[IdxLike] = None,
        cols: Optional[ColsLike] = None,
        *args,
        **kwargs,
    ) -> Any:
        """Reads data from the engine, applying row and column filters.

        The method resolves columns based on the following priority:
        1. Explicit `cols` argument provided here.
        2. Columns defined in the table's `RestrictedDBProxy` (if applicable).
        3. All columns (default).

        Args:
            what: The component to read (e.g., 'fingerprints').
            rows: Row indices or masks to retrieve.
            cols: Specific columns or features to retrieve.
            *args: Positional arguments for the engine.
            **kwargs: Keyword arguments for the engine.

        Returns:
            Any: The materialized data (e.g., pd.DataFrame or np.ndarray).
        """
        engine = self._db.request_engine(self._w2e[what])

        effective_cols = cols
        if effective_cols is None and isinstance(self._db, RestrictedDBProxy):
            effective_cols = self._db.cols

        return engine.materialize(
            self._name, what, rows, effective_cols, *args, **kwargs
        )

    def view(
        self,
        rows: Optional[IdxLike] = None,
        cols: Optional[ColsLike] = None,
        **kwargs,
    ) -> Self:
        """Creates a restricted, read-only view of the current table.

        Args:
            rows: Row mask (slice, list of indices, or boolean array).
            cols: Column selection to be applied to all subsequent 
                materialization calls in the returned view.
            **kwargs: Additional constraints for the proxy.

        Returns:
            Self: A new instance of the table linked to a restricted proxy.
        """
        restricted_db = self._db.spawn_restricted_view(
            namespace=self._name,
            rows=rows,
            cols=cols,
            **kwargs,
        )

        return self.__class__(
            db=restricted_db,
            name=self._name,
            _w2e=self._w2e,
            _history=self._history,
        )

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of rows visible in the current view."""
        pass