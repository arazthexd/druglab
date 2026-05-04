from typing import Type, Any
from typing_extensions import Self

from .base import BaseDB

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..types import IdxLike

class RestrictedDBProxy(BaseDB):
    """
    A lightweight wrapper around a parent database. 
    Intercepts engine requests and forces them to be read-only, restricted views.
    """
    def __init__(self, parent_db: BaseDB, namespace: str, rows: IdxLike, **kwargs):
        super().__init__()
        self._parent = parent_db
        self._namespace = namespace
        self._rows = rows
        self._kwargs = kwargs
        
        # A view shares the table registry of its parent.
        # (This allows cross-table queries to still work inside a view)
        self._tables = parent_db._tables 

    def request_engine(self, name: str) -> Any:
        # 1. Ask the parent for the REAL root engine
        root_engine = self._parent.request_engine(name)
        
        # 2. Slice it down to just our specific rows and namespace!
        # (The engine's spawn_view will set _is_view=True automatically)
        return root_engine.spawn_view(
            namespace=self._namespace, 
            rows=self._rows, 
            **self._kwargs
        )

    def spawn_restricted_view(self, namespace: str, rows: IdxLike, **kwargs) -> Self:
        # Chaining! If a user views a view, we just wrap it again.
        # The engines are smart enough to intersect the masks.
        return RestrictedDBProxy(
            parent_db=self, 
            namespace=namespace, 
            rows=rows, 
            **kwargs
        )