import os
from typing import Type, Any

from ..table import BaseTable
from ..engine import BaseEngine
from ..types import IdxLike
from .base import BaseDB
from .proxy import RestrictedDBProxy

class LocalDB(BaseDB):
    """
    The Root Database Manager. Handles the physical `.dldb` directory 
    and acts as the connection pool for root engines.
    """
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        
        # Initialize the bundle directory
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def request_engine(self, name: str) -> BaseEngine:
        """Lazy-loads and caches engines so we don't open multiple connections."""
        if name in self._engines:
            return self._engines[name]

        if name == "duckdb":
            from ..engine.duckdb_engine import DuckDBEngine
            db_file = os.path.join(self.path, "metadata.db")
            self._engines[name] = DuckDBEngine(db_file)
            
        elif name == "pandas":
            from ..engine.pandas_engine import PandasEngine
            self._engines[name] = PandasEngine()
            
        # elif name == "zarr":
        #    from .zarr_engine import ZarrEngine
        #    tensor_dir = os.path.join(self.path, "tensors.zarr")
        #    self._engines[name] = ZarrEngine(tensor_dir)
            
        else:
            raise NotImplementedError(f"Engine '{name}' is not supported.")
            
        return self._engines[name]

    def spawn_restricted_view(self, namespace: str, rows: IdxLike, **kwargs) -> RestrictedDBProxy:
        """Spawns the synchronizer proxy for the requested table."""
        return RestrictedDBProxy(parent_db=self, namespace=namespace, rows=rows, **kwargs)

    # ------------------------------------------------------------------
    # User API
    # ------------------------------------------------------------------

    def create_table(self, name: str, table_cls: Type[BaseTable], **kwargs) -> BaseTable:
        """Instantiates a new table, registers it, and returns it."""
        if name in self._tables:
            raise ValueError(f"Table '{name}' already exists in this database.")
            
        table = table_cls(db=self, name=name, **kwargs)
        self._tables[name] = table
        return table
        
    def close(self):
        """Safely shuts down all active engines."""
        for engine in self._engines.values():
            if hasattr(engine, "close"):
                engine.close()