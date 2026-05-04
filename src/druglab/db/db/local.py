from __future__ import annotations

import os
from typing import Optional, Type, TYPE_CHECKING

from ..engine import BaseEngine
from ..types import ColsLike, IdxLike
from .base import BaseDB
from .proxy import RestrictedDBProxy

if TYPE_CHECKING:
    from ..table import BaseTable


class LocalDB(BaseDB):
    """
    The root database manager.

    Handles the physical ``.dldb`` directory and acts as the connection pool
    for root engines.  Engines are lazy-loaded and cached; only one connection
    per engine type is kept open at a time.

    Usage as a context manager
    --------------------------
    ``LocalDB`` implements ``__enter__`` / ``__exit__`` so it can be used in
    a ``with`` block, which guarantees engines are closed cleanly even if an
    exception is raised::

        with LocalDB("/path/to/my.dldb") as db:
            molecules = db.create_table("molecules", MoleculeTable)
            molecules.extend(mol_list)

    """

    def __init__(self, path: str):
        super().__init__()
        self.path = path

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "LocalDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
        # Do not suppress exceptions
        return None

    # ------------------------------------------------------------------
    # Engine routing
    # ------------------------------------------------------------------

    def request_engine(self, name: str) -> BaseEngine:
        """Lazy-loads and caches engines (one connection per type)."""
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
        #     from ..engine.zarr_engine import ZarrEngine
        #     tensor_dir = os.path.join(self.path, "tensors.zarr")
        #     self._engines[name] = ZarrEngine(tensor_dir)

        else:
            raise NotImplementedError(f"Engine '{name}' is not supported.")

        return self._engines[name]

    def spawn_restricted_view(
        self,
        namespace: str,
        rows: Optional[IdxLike],
        cols: Optional[ColsLike] = None,
        **kwargs,
    ) -> RestrictedDBProxy:
        return RestrictedDBProxy(
            parent_db=self,
            namespace=namespace,
            rows=rows,
            cols=cols,
            **kwargs,
        )

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

    def close(self) -> None:
        """Safely shuts down all active engines."""
        for engine in self._engines.values():
            if hasattr(engine, "close"):
                engine.close()
        self._engines.clear()