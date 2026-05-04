from typing import Optional, Dict, List
from typing_extensions import Self

import numpy as np
import pandas as pd
import duckdb

from ..types import IdxLike
from .base import BaseEngine

class DuckDBEngine(BaseEngine[pd.DataFrame]):
    """
    DuckDB implementation with schema isolation and native SQL pushdown.
    """
    
    def __init__(
        self, 
        path: str, 
        read_only: bool = False, 
        _conn: Optional[duckdb.DuckDBPyConnection] = None, 
        _masks: Optional[Dict[str, IdxLike]] = None,
        _is_view: bool = False
    ):
        self.path = path
        if _conn is not None:
            self.conn = _conn
        else:
            self.conn = duckdb.connect(path, read_only=read_only)
            
        self._masks = _masks if _masks is not None else {}
        self._is_view = _is_view

    def _get_table_name(self, namespace: str, what: str) -> str:
        return f"{namespace}_{what}"
    
    def _table_exists(self, table_name: str) -> bool:
        """Checks if a table exists in the DuckDB catalog."""
        return self.conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", 
            [table_name]
        ).fetchone()[0] > 0
    
    def materialize(
        self, 
        namespace: str, 
        what: str, 
        rows: Optional[IdxLike] = None, 
        *args, 
        **kwargs
    ) -> pd.DataFrame:
        """
        Executes the query against the dynamically named table, applying native SQL pushdown.
        """
        # Construct the table name dynamically
        table_name = self._get_table_name(namespace, what)

        if not self._table_exists(table_name):
            raise KeyError(f"Table '{table_name}' does not exist in the DuckDB engine catalog.")
        
        # Base query safely targets the isolated table
        base_query = f"SELECT * FROM {table_name}"
        
        # If the user passes new rows, we intersect them with the engine's 
        # internal view state.
        current_mask = self._masks.get(namespace)
        
        if rows is not None:
            active_mask = self._combine_masks(current_mask, rows)
        else:
            active_mask = current_mask
        
        # No mask? Pull the whole table.
        if active_mask is None:
            return self.conn.execute(base_query).df()
        
        # --- NATIVE PUSHDOWN LOGIC ---
        
        if isinstance(active_mask, slice):
            start = active_mask.start or 0
            step = active_mask.step or 1
            stop = active_mask.stop
            
            if step != 1:
                raise NotImplementedError("Stepped slices are not supported in SQL pushdown.")
            
            # Requires an explicit ORDER BY to guarantee deterministic slicing in SQL
            query = f"{base_query} ORDER BY _row_id"
            
            if stop is not None:
                limit = stop - start
                query += f" LIMIT {limit} OFFSET {start}"
            else:
                query += f" OFFSET {start}"
                
            return self.conn.execute(query).df()
        
        elif isinstance(active_mask, (list, np.ndarray)):
            indices = active_mask.tolist() if isinstance(active_mask, np.ndarray) else active_mask
            
            if not indices:
                return self.conn.execute(f"{base_query} LIMIT 0").df()
                
            row_str = ", ".join(map(str, indices))
            query = f"{base_query} WHERE _row_id IN ({row_str})"
            return self.conn.execute(query).df()

        raise ValueError(f"Unsupported row mask type: {type(active_mask)}")
    
    def spawn_view(self, namespace: str, rows: IdxLike) -> Self:
        new_masks = self._masks.copy()
        new_masks[namespace] = self._combine_masks(new_masks.get(namespace), rows)
        
        return self.__class__(
            path=self.path,
            _conn=self.conn,
            _masks=new_masks,
            _is_view=True
        )
    
    def write(self, namespace: str, what: str, data: pd.DataFrame, **kwargs) -> None:
        """
        Creates or appends data to the physical table.
        DuckDB magically reads the local Python variable named `data` directly into SQL.
        """
        # 1. Enforce View Read-Only Rule
        if self._is_view:
            raise PermissionError("Cannot write data through a restricted view. Write to the root engine.")

        # 2. Skip empty writes to prevent schema errors
        if data is None or data.empty:
            return

        table_name = self._get_table_name(namespace, what)

        # 3. Check if the table already exists in DuckDB's catalog
        table_exists = self.conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", 
            [table_name]
        ).fetchone()[0] > 0

        if not table_exists:
            # CREATE PATH: DuckDB infers the SQL schema directly from the Pandas DataFrame!
            self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM data")
        else:
            # APPEND PATH: We explicitly define columns to ensure alignment
            # in case the DataFrame columns are in a different order than the SQL table.
            cols = ", ".join([f'"{c}"' for c in data.columns])
            self.conn.execute(f"INSERT INTO {table_name} ({cols}) SELECT * FROM data")

    def export(self, target: Optional[str] = None, namespaces: Optional[List[str]] = None) -> Self:
        """Exports the current view into a completely independent DuckDB file."""
        import os

        if target is None:
            import tempfile
            import uuid
            # Generate a safe, unique path without physically creating the file yet
            target = os.path.join(tempfile.gettempdir(), f"druglab_engine_{uuid.uuid4().hex[:8]}.db")
        
        # Don't overwrite accidentally
        if os.path.exists(target):
            raise FileExistsError(f"Target path {target} already exists.")
            
        # 1. ATTACH the new, empty database directly to our current connection
        self.conn.execute(f"ATTACH '{target}' AS export_db")
        
        tables = self.conn.execute("SHOW TABLES").df()["name"].tolist()
        
        for tbl in tables:
            try:
                namespace, what = tbl.split("_", 1)
            except ValueError:
                continue
                
            # If the user only wants to export ONE table (e.g., "molecules"), skip the rest
            if namespaces and namespace not in namespaces:
                continue

            active_mask = self._masks.get(namespace)
            
            # 2. Build the base source query (Native Pushdown)
            source_query = f"SELECT * FROM {tbl}"
            if isinstance(active_mask, slice):
                start = active_mask.start or 0
                query_limit = f" LIMIT {active_mask.stop - start}" if active_mask.stop else ""
                source_query += f" ORDER BY _row_id{query_limit} OFFSET {start}"
            elif isinstance(active_mask, (list, np.ndarray)):
                if len(active_mask) == 0:
                    source_query += " LIMIT 0"
                else:
                    indices_str = ", ".join(map(str, active_mask))
                    source_query += f" WHERE _row_id IN ({indices_str})"

            # 3. THE MAGIC: Copy to the attached DB and re-index _row_id in one step!
            # EXCLUDE(_row_id) drops the old broken indices, and row_number() creates the perfect new ones.
            export_sql = f"""
                CREATE TABLE export_db.{tbl} AS 
                SELECT 
                    (row_number() OVER ()) - 1 AS _row_id, 
                    * EXCLUDE (_row_id) 
                FROM ({source_query})
            """
            self.conn.execute(export_sql)
            
        # 4. Safely detach and return the new root engine
        self.conn.execute("DETACH export_db")
        return self.__class__(path=target)

    def _combine_masks(self, current_mask: Optional[IdxLike], new_mask: IdxLike) -> IdxLike:
        if current_mask is None:
            return new_mask
        curr_arr = np.array(current_mask) if not isinstance(current_mask, slice) else np.arange(current_mask.stop)[current_mask]
        if isinstance(new_mask, slice):
            return curr_arr[new_mask]
        else:
            return curr_arr[np.array(new_mask)]

    def flush(self) -> None:
        self.conn.execute("CHECKPOINT")

    def close(self) -> None:
        self.conn.close()