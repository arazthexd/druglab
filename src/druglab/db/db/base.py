from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from typing_extensions import Self

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..table import BaseTable
    from ..engine import BaseEngine
    from ..types import ColsLike, IdxLike


class BaseDB(ABC):
    """
    Abstract contract and common utility for all DrugLab database managers.

    Subclasses define routing logic (``request_engine``) and how restricted
    views are spawned (``spawn_restricted_view``).
    """

    def __init__(self):
        self._tables: Dict[str, "BaseTable"] = {}
        self._engines: Dict[str, "BaseEngine"] = {}

    # ------------------------------------------------------------------
    # Concrete utilities
    # ------------------------------------------------------------------

    def get_table(self, name: str) -> "BaseTable":
        if name not in self._tables:
            raise KeyError(f"Table '{name}' is not registered in this database.")
        return self._tables[name]

    def list_tables(self) -> List[str]:
        return list(self._tables.keys())

    # ------------------------------------------------------------------
    # Abstract contracts
    # ------------------------------------------------------------------

    @abstractmethod
    def request_engine(self, name: str) -> "BaseEngine":
        """
        Factory / router method.  Returns the active engine connection,
        creating it lazily if necessary.

        For proxy databases this should return a *view* of the parent engine,
        restricted to the proxy's row mask and column selection.
        """
        pass

    @abstractmethod
    def spawn_restricted_view(
        self,
        namespace: str,
        rows: Optional["IdxLike"],
        cols: Optional["ColsLike"] = None,
        **kwargs,
    ) -> Self:
        """
        Creates a restricted clone of the database manager.

        Parameters
        ----------
        namespace:
            The table namespace to restrict.
        rows:
            Row index mask applied to *all* engines for this namespace.
        cols:
            Optional column selection passed through to ``materialize`` calls.
            Stored on the proxy so that the table's read path can forward it
            without the caller having to remember it every time.
        """
        pass