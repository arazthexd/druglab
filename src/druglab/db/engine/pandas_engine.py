from __future__ import annotations

from typing import Dict, List, Optional
from typing_extensions import Self

import numpy as np
import pandas as pd

from ..types import ColsLike, IdxLike
from .base import BaseEngine


class PandasEngine(BaseEngine[pd.DataFrame]):
    """
    In-memory DataFrame engine.  Perfect for testing, small datasets,
    or fully RAM-bound workflows.

    Schema validation
    -----------------
    On every ``write`` call to an existing table the incoming DataFrame columns
    are validated against the stored schema.  Extra or missing columns raise
    ``ValueError`` immediately rather than silently producing misaligned data.
    """

    def __init__(
        self,
        _store: Optional[Dict[str, pd.DataFrame]] = None,
        _masks: Optional[Dict[str, IdxLike]] = None,
        _is_view: bool = False,
    ):
        self._store = _store if _store is not None else {}
        self._masks = _masks if _masks is not None else {}
        self._is_view = _is_view

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_table_name(self, namespace: str, what: str) -> str:
        return f"{namespace}_{what}"

    def _validate_schema(self, table_name: str, data: pd.DataFrame) -> None:
        """
        Raises ``ValueError`` if ``data`` does not match the stored schema.
        """
        stored_cols = set(self._store[table_name].columns)
        incoming_cols = set(data.columns)

        extra = incoming_cols - stored_cols
        missing = stored_cols - incoming_cols

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
        if table_name not in self._store:
            raise KeyError(f"Table '{table_name}' does not exist in the Pandas engine.")
        return len(self._store[table_name])

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
        Returns a copy of the requested slice, with optional column selection.

        ``cols`` is applied *after* row filtering, selecting only the requested
        columns from the resulting DataFrame.  ``None`` returns all columns.
        """
        table_name = self._get_table_name(namespace, what)

        if table_name not in self._store:
            raise KeyError(f"Table '{table_name}' does not exist in the Pandas engine.")

        df = self._store[table_name]

        # --- Row mask composition ---
        current_mask = self._masks.get(namespace)
        if rows is not None:
            total = self.n_rows(namespace, what)
            active_mask = self._combine_masks(current_mask, rows, total)
        else:
            active_mask = current_mask

        if active_mask is None:
            result = df.copy()
        elif isinstance(active_mask, slice):
            result = df.iloc[active_mask].copy()
        elif isinstance(active_mask, (list, np.ndarray)):
            result = df.iloc[np.asarray(active_mask, dtype=np.intp)].copy()
        else:
            raise ValueError(f"Unsupported row mask type: {type(active_mask)}")

        # --- Column selection ---
        if cols is not None:
            missing = [c for c in cols if c not in result.columns]
            if missing:
                raise KeyError(f"Columns not found in '{table_name}': {missing}")
            result = result[cols]

        return result

    # ------------------------------------------------------------------
    # spawn_view
    # ------------------------------------------------------------------

    def spawn_view(self, namespace: str, rows: IdxLike) -> Self:
        current_mask = self._masks.get(namespace)

        # Resolve n_rows only if we need to compose masks
        if current_mask is not None:
            tables = [k for k in self._store if k.startswith(f"{namespace}_")]
            if tables:
                n = len(self._store[tables[0]])
            else:
                n = 0
            new_mask = self._combine_masks(current_mask, rows, n)
        else:
            new_mask = rows

        new_masks = self._masks.copy()
        new_masks[namespace] = new_mask

        return self.__class__(
            _store=self._store,
            _masks=new_masks,
            _is_view=True,
        )

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def write(self, namespace: str, what: str, data: pd.DataFrame, **kwargs) -> None:
        if self._is_view:
            raise PermissionError(
                "Cannot write data through a restricted view. Write to the root engine."
            )

        if data is None or data.empty:
            return

        table_name = self._get_table_name(namespace, what)

        if table_name not in self._store:
            # CREATE: deep-copy to ensure the engine owns its data
            self._store[table_name] = data.copy()
        else:
            # APPEND: validate schema first
            self._validate_schema(table_name, data)
            self._store[table_name] = pd.concat(
                [self._store[table_name], data],
                ignore_index=True,
            )

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------

    def export(
        self,
        target: Optional[str] = None,
        namespaces: Optional[List[str]] = None,
    ) -> Self:
        """
        Exports the current in-memory view to a brand-new isolated engine.
        ``target`` is accepted for API consistency but is ignored (RAM-bound).
        """
        new_engine = PandasEngine()

        for tbl_name in self._store.keys():
            try:
                namespace, what = tbl_name.split("_", 1)
            except ValueError:
                continue

            if namespaces and namespace not in namespaces:
                continue

            df_subset = self.materialize(namespace, what)

            if "_row_id" in df_subset.columns:
                df_subset = df_subset.copy()
                df_subset["_row_id"] = np.arange(len(df_subset))

            new_engine.write(namespace, what, df_subset)

        return new_engine