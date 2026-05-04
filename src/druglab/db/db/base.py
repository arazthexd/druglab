from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
from typing_extensions import Self

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..table import BaseTable
    from ..engine import BaseEngine
    from ..types import IdxLike


class BaseDB(ABC):
    """
    Abstract contract and common utility class for all DrugLab database managers.
    Handles table registries while forcing subclasses to define routing logic.
    """

    def __init__(self):
        # Common state for any DB (Root or Proxy)
        self._tables: Dict[str, 'BaseTable'] = {}
        self._engines: Dict[str, 'BaseEngine'] = {}

    # ------------------------------------------------------------------
    # Concrete Utilities (Inherited by all subclasses)
    # ------------------------------------------------------------------

    def get_table(self, name: str) -> 'BaseTable':
        """Retrieves a registered table."""
        if name not in self._tables:
            raise KeyError(f"Table '{name}' is not registered in this database.")
        return self._tables[name]

    def list_tables(self) -> List[str]:
        """Returns a list of all registered table namespaces."""
        return list(self._tables.keys())

    # ------------------------------------------------------------------
    # Abstract Contracts (Must be implemented by Root and Proxy)
    # ------------------------------------------------------------------

    @abstractmethod
    def request_engine(self, name: str) -> 'BaseEngine':
        """
        Factory/Router method. 
        Ensures the requested engine exists and returns the active connection.
        """
        pass

    @abstractmethod
    def spawn_restricted_view(self, namespace: str, rows: IdxLike, **kwargs) -> Self:
        """
        Creates a restricted clone of the database manager, applying the given 
        row mask to the specific namespace across all underlying engines.
        """
        pass