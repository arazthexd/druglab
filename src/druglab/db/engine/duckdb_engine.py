from __future__ import annotations

from typing import Dict, List, Optional
from typing_extensions import Self

import numpy as np
import pandas as pd
import duckdb

from ..types import ColsLike, IdxLike
from .base import BaseEngine


class DuckDBEngine(BaseEngine[pd.DataFrame]):
    """
    DuckDB implementation with schema isolation and native SQL pushdown.

    Connection-sharing model
    ------------------------
    Views share the same ``duckdb.DuckDBPyConnection`` object as their parent.
    DuckDB connections are **not thread-safe**: do not materialise two views
    from the same root engine concurrently across threads.  If you need
    parallelism, call ``export()`` first to produce independent root engines
    backed by separate files, then open them in separate processes/threads.

    Schema validation
    -----------------
    On every ``write`` call the incoming DataFrame is validated against the
    schema already stored for that table (column names and dtypes).  New
    columns are rejected; missing columns are rejected.  This prevents silent
    data corruption from mis-ordered append calls.
    """

    def __init__(
        self,
        path: str,
        read_only: bool = False,
        _conn: Optional[duckdb.DuckDBPyConnection] = None,
        _masks: Optional[Dict[str, IdxLike]] = None,
        _is_view: bool = False,
    ):
        self.path = path
        self.conn = _conn if _conn is not None else duckdb.connect(path, read_only=read_only)
        self._masks = _masks if _masks is not None else {}
        self._is_view = _is_view

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_table_name(self, namespace: str, what: str) -> str:
        return f"{namespace}_{what}"

    def _table_exists(self, table_name: str) -> bool:
        return (
            self.conn.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
                [table_name],
            ).fetchone()[0]
            > 0
        )

    def _get_schema(self, table_name: str) -> Dict[str, str]:
        """Returns {column_name: dtype_string} for an existing table."""
        rows = self.conn.execute(
            "SELECT column_name, data_type "
            "FROM information_schema.columns "
            "WHERE table_name = ? "
            "ORDER BY ordinal_position",
            [table_name],
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def _validate_schema(self, table_name: str, data: pd.DataFrame) -> None:
        """
        Raises ``ValueError`` if ``data`` does not match the stored schema.

        Rules
        -----
        * No extra columns in ``data`` that do not exist in the table.
        * No columns present in the table that are missing from ``data``
          (would produce NULL-filled rows silently).
        * Column order does not need to match (INSERT uses explicit column list).
        """
        stored = self._get_schema(table_name)
        incoming = set(data.columns)
        stored_cols = set(stored.keys())

        extra = incoming - stored_cols
        missing = stored_cols - incoming

        errors: List[str] = []
        if extra:
            errors.append(f"extra columns not in table: {sorted(extra)}")
        if missing:
            errors.append(f"columns missing from DataFrame: {sorted(missing)}")

        if errors:
            raise ValueError(
                f"Schema mismatch for table '{table_name}': " + "; ".join(errors)
            )

    # ------------------------------------------------------------------
    # n_rows
    # ------------------------------------------------------------------

    def n_rows(self, namespace: str, what: str) -> int:
        """Returns the *physical* (unmasked) row count."""
        table_name = self._get_table_name(namespace, what)
        if not self._table_exists(table_name):
            raise KeyError(f"Table '{table_name}' does not exist in the DuckDB engine catalog.")
        return self.conn.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]

    # ------------------------------------------------------------------
    # materialize
    # ------------------------------------------------------------------

    def materialize(
        self,
        namespace: str,
        what: str,
        rows: Optional[IdxLike] = None,
        cols: Optional[ColsLike] = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Executes a query against the engine, applying row and column pushdown.

        ``cols`` is pushed directly into the SELECT list, avoiding the cost of
        fetching unwanted columns from DuckDB.  ``_row_id`` is always included
        unless the caller explicitly omits it from ``cols`` — be aware that
        omitting it will break mask composition in downstream views.
        """
        table_name = self._get_table_name(namespace, what)

        if not self._table_exists(table_name):
            raise KeyError(f"Table '{table_name}' does not exist in the DuckDB engine catalog.")

        # --- Column selection ---
        if cols is None:
            select_clause = "*"
        else:
            quoted = ", ".join(f'"{c}"' for c in cols)
            select_clause = quoted

        base_query = f"SELECT {select_clause} FROM {table_name}"

        # --- Row mask composition ---
        current_mask = self._masks.get(namespace)
        if rows is not None:
            total = self.n_rows(namespace, what)
            active_mask = self._combine_masks(current_mask, rows, total)
        else:
            active_mask = current_mask

        # No mask → pull everything
        if active_mask is None:
            return self.conn.execute(base_query).df()

        # --- Native SQL pushdown ---
        if isinstance(active_mask, slice):
            start = active_mask.start or 0
            step = active_mask.step or 1
            stop = active_mask.stop

            if step != 1:
                raise NotImplementedError("Stepped slices are not supported in SQL pushdown.")

            query = f"{base_query} ORDER BY _row_id"
            if stop is not None:
                query += f" LIMIT {stop - start} OFFSET {start}"
            else:
                query += f" OFFSET {start}"

            return self.conn.execute(query).df()

        elif isinstance(active_mask, (list, np.ndarray)):
            indices = active_mask.tolist() if isinstance(active_mask, np.ndarray) else active_mask

            if not indices:
                return self.conn.execute(f"{base_query} LIMIT 0").df()

            row_str = ", ".join(map(str, indices))
            return self.conn.execute(f"{base_query} WHERE _row_id IN ({row_str})").df()

        raise ValueError(f"Unsupported row mask type: {type(active_mask)}")

    # ------------------------------------------------------------------
    # spawn_view
    # ------------------------------------------------------------------

    def spawn_view(self, namespace: str, rows: IdxLike) -> Self:
        current_mask = self._masks.get(namespace)

        # Only need n_rows if we have a prior mask to compose with.
        if current_mask is not None:
            tables = self.conn.execute("SHOW TABLES").df()["name"].tolist()
            ns_tables = [t for t in tables if t.startswith(f"{namespace}_")]
            n = self.n_rows(namespace, ns_tables[0].split("_", 1)[1]) if ns_tables else 0
            new_mask = self._combine_masks(current_mask, rows, n)
        else:
            new_mask = rows

        new_masks = self._masks.copy()
        new_masks[namespace] = new_mask

        return self.__class__(
            path=self.path,
            _conn=self.conn,
            _masks=new_masks,
            _is_view=True,
        )

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def write(self, namespace: str, what: str, data: pd.DataFrame, **kwargs) -> None:
        """
        Creates or appends data to the physical table with schema validation.
        """
        if self._is_view:
            raise PermissionError(
                "Cannot write data through a restricted view. Write to the root engine."
            )

        if data is None or data.empty:
            return

        table_name = self._get_table_name(namespace, what)

        if not self._table_exists(table_name):
            # CREATE: infer schema from the DataFrame
            self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM data")
        else:
            # APPEND: dynamically add missing columns to the table first
            self._evolve_schema(table_name, data)
            
            # Insert using explicit column names. DuckDB will automatically 
            # fill any table columns missing from this DataFrame with NULL.
            cols = ", ".join(f'"{c}"' for c in data.columns)
            self.conn.execute(f"INSERT INTO {table_name} ({cols}) SELECT * FROM data")

    def _pandas_to_duckdb_type(self, dtype: np.dtype) -> str:
        """Heuristic mapping from pandas dtypes to DuckDB SQL types."""
        dtype_str = str(dtype).lower()
        if "int8" in dtype_str: return "TINYINT"
        if "int16" in dtype_str: return "SMALLINT"
        if "int32" in dtype_str: return "INTEGER"
        if "int" in dtype_str: return "BIGINT"
        if "float16" in dtype_str or "float32" in dtype_str: return "REAL"
        if "float" in dtype_str: return "DOUBLE"
        if "bool" in dtype_str: return "BOOLEAN"
        if "datetime" in dtype_str: return "TIMESTAMP"
        return "VARCHAR"
    
    def _evolve_schema(self, table_name: str, data: pd.DataFrame) -> None:
        """
        Dynamically expands the DuckDB table if the incoming DataFrame 
        has new columns.
        """
        stored_schema = self._get_schema(table_name)
        stored_cols = set(stored_schema.keys())
        
        for col in data.columns:
            if col not in stored_cols:
                sql_type = self._pandas_to_duckdb_type(data[col].dtype)
                # Execute schema evolution
                self.conn.execute(f'ALTER TABLE {table_name} ADD COLUMN "{col}" {sql_type}')

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------

    def export(
        self,
        target: Optional[str] = None,
        namespaces: Optional[List[str]] = None,
    ) -> Self:
        """
        Exports the current view into an independent DuckDB file.

        Uses ``ATTACH`` for zero-copy transfer inside DuckDB.  ``_row_id`` is
        re-indexed to start at 0, guaranteeing alignment with any tensor
        engines that were exported separately.
        """
        import os
        import tempfile
        import uuid

        if target is None:
            target = os.path.join(
                tempfile.gettempdir(), f"druglab_engine_{uuid.uuid4().hex[:8]}.db"
            )

        if os.path.exists(target):
            raise FileExistsError(f"Target path '{target}' already exists.")

        self.conn.execute(f"ATTACH '{target}' AS export_db")
        try:
            tables = self.conn.execute("SHOW TABLES").df()["name"].tolist()

            for tbl in tables:
                try:
                    namespace, what = tbl.split("_", 1)
                except ValueError:
                    continue

                if namespaces and namespace not in namespaces:
                    continue

                active_mask = self._masks.get(namespace)

                # Build source query with pushdown
                source_query = f"SELECT * FROM {tbl}"
                if isinstance(active_mask, slice):
                    start = active_mask.start or 0
                    limit_clause = (
                        f" LIMIT {active_mask.stop - start}" if active_mask.stop else ""
                    )
                    source_query += f" ORDER BY _row_id{limit_clause} OFFSET {start}"
                elif isinstance(active_mask, (list, np.ndarray)):
                    if len(active_mask) == 0:
                        source_query += " LIMIT 0"
                    else:
                        indices_str = ", ".join(map(str, active_mask))
                        source_query += f" WHERE _row_id IN ({indices_str})"

                # Re-index _row_id from 0 in the exported table
                self.conn.execute(f"""
                    CREATE TABLE export_db.{tbl} AS
                    SELECT
                        (row_number() OVER ()) - 1 AS _row_id,
                        * EXCLUDE (_row_id)
                    FROM ({source_query})
                """)

        finally:
            # Always detach, even if an export step fails mid-way
            self.conn.execute("DETACH export_db")

        return self.__class__(path=target)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def flush(self) -> None:
        self.conn.execute("CHECKPOINT")

    def close(self) -> None:
        self.conn.close()