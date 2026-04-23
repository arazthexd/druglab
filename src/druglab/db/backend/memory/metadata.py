"""
druglab.db.backend.memory.metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In-memory storage mixin for metadata.

Currently, the only supported mixin is ``MemoryMetadataMixin`` which manages metadata
in RAM as a Pandas DataFrame. However, other mixins may be supported in the future.
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from ...indexing import RowSelection
from ..base.mixins import BaseMetadataMixin

if TYPE_CHECKING:
    from druglab.db.indexing import INDEX_LIKE

__all__ = ['MemoryMetadataMixin']

class MemoryMetadataMixin(BaseMetadataMixin):
    """
    In-memory metadata storage mixin utilizing Pandas DataFrames.
    
    This mixin manages tabular metadata natively in RAM. Operations rely on 
    Pandas `.iloc` and `.loc` indexing to enforce strict query pushdown and 
    avoid unnecessary DataFrame copies.
    """

    # ------------------------------------------------------------------
    # Initialization Hooks
    # ------------------------------------------------------------------

    def initialize_storage_context(
        self, 
        metadata: Optional[pd.DataFrame] = None, 
        **kwargs: Any
    ) -> None:
        """Initialize storage context via ``metadata`` kwarg.
        
        Parameters
        ----------
        metadata : pd.DataFrame, optional
            The initial metadata DataFrame.
        **kwargs
            Additional keyword arguments. These are passed in the MRO chain.
        """
        self._metadata = metadata 
        super().initialize_storage_context(**kwargs)

    def bind_capabilities(self):
        if self._metadata is None:
            self._metadata = pd.DataFrame(index=range(len(self)))
        return super().bind_capabilities()
    
    # ------------------------------------------------------------------
    # Metadata Mixin API
    # ------------------------------------------------------------------

    def get_metadata(
        self,
        idx: Optional[INDEX_LIKE] = None,
        cols: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        sel = RowSelection.from_raw(idx, len(self._metadata))

        if sel.is_full:
            df = self._metadata
        else:
            df = self._metadata.iloc[sel.positions]

        if cols is not None:
            if isinstance(cols, str):
                cols = [cols]
            df = df[cols]

        return df.reset_index(drop=True)

    def add_metadata_column(
        self,
        name: str,
        value: Union[pd.Series, np.ndarray, List[Any]],
        idx: Optional[INDEX_LIKE] = None,
        na: Any = None,
        **kwargs
    ) -> None:
        sel = RowSelection.from_raw(idx, len(self._metadata))
        value = np.asarray(value)

        if sel.is_full:
            self._metadata[name] = value
        else:
            resolved = sel.positions
            if na is None and np.issubdtype(value.dtype, np.integer):
                raise ValueError(
                    "na must be provided when populating integer columns"
                )
            if na is None and np.issubdtype(value.dtype, np.floating):
                na = np.nan
            if value.shape[0] != len(resolved):
                raise ValueError(
                    f"Expected {len(resolved)} values, got {value.shape[0]}"
                )
            
            # Create a full array of `na` and populate the targeted indices
            arr = np.full(len(self._metadata), na, dtype=np.asarray(value).dtype)
            arr[resolved] = value
            self._metadata[name] = arr

    def update_metadata(
        self,
        values: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
        idx: Optional[INDEX_LIKE] = None,
        **kwargs
    ) -> None:
        sel = RowSelection.from_raw(idx, len(self._metadata))

        if isinstance(values, pd.DataFrame):
            val_dict = {col: values[col].values for col in values.columns}
        elif isinstance(values, pd.Series):
            if values.name is None:
                raise ValueError("Series must have a name to update metadata.")
            val_dict = {values.name: values.values}
        else:
            val_dict = values

        for col, val in val_dict.items():
            if col not in self._metadata.columns:
                raise KeyError(
                    f"Column '{col}' does not exist in metadata. "
                    "Use add_metadata_column."
                )
            if sel.is_full:
                self._metadata[col] = val
            else:
                self._metadata.iloc[
                    sel.positions,
                    self._metadata.columns.get_loc(col)
                ] = val

    def drop_metadata_columns(
        self,
        cols: Optional[Union[str, List[str]]] = None
    ) -> None:
        if cols is None:
            self._metadata = pd.DataFrame(index=self._metadata.index)
        else:
            if isinstance(cols, str):
                cols = [cols]
            self._metadata = self._metadata.drop(columns=cols)

    def _n_metadata_rows(self) -> int:
        """
        Get the total number of rows in the metadata DataFrame.

        Returns
        -------
        int
            Length of the internal metadata DataFrame. If the DataFrame is empty
            (0 rows, 0 columns), this function returns the length of the object 
            store (i.e., the number of objects stored in the backend).

        Notes
        -----
        The implementation first checks if the DataFrame is empty (0 rows, 0 columns).
        If it is, the function returns the length of the object store. Otherwise, it returns
        the length of the DataFrame.
        """
        n_rows, n_columns = self._metadata.shape
        if n_rows == 0 and n_columns == 0:
            # Genuinely empty backend
            return len(self)
        return n_rows

    def _validate_metadata(self) -> None:
        """
        Validate internal metadata consistency. 
        (No-op for in-memory as Pandas inherently enforces rectangular integrity).
        """
        # In-memory pandas dataframes inherently enforce structural integrity
        pass
    
    # ------------------------------------------------------------------
    # Materialization Hooks
    # ------------------------------------------------------------------
    
    def _gather_materialized_state(
        self,
        target_path: Optional[Path] = None,
        index_map: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Return ``{"metadata": sliced_or_full_copy}`` for ``clone_concrete``."""
        result = super()._gather_materialized_state(
            target_path=target_path, index_map=index_map
        )
        if index_map is not None:
            result["metadata"] = (
                self._metadata.iloc[index_map].reset_index(drop=True).copy()
            )
        else:
            result["metadata"] = self._metadata.copy()
        return result

    # ------------------------------------------------------------------
    # Persistence Hooks
    # ------------------------------------------------------------------

    def save_storage_context(self, path: Path, **kwargs: Any) -> None:
        """
        Persist metadata to ``<path>/metadata.parquet`` (or ``.csv`` fallback).
 
        An empty (0-row, 0-col) DataFrame still writes a row-count stub as CSV
        so that the bundle preserves the number of objects.
        """
        if not self._metadata.empty:
            try:
                self._metadata.to_parquet(path / "metadata.parquet", index=False)
            except Exception:
                self._metadata.to_csv(path / "metadata.csv", index=False)
        else:
            # Row-count stub so the loader knows how many objects exist.
            pd.DataFrame(index=range(len(self))).to_csv(
                path / "metadata.csv", index=True
            )
        super().save_storage_context(path, **kwargs)
 
    @classmethod
    def load_storage_context(
        cls,
        path: Path,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Read metadata from ``<path>/metadata.parquet`` or ``metadata.csv``
        and add it to the accumulated kwargs under the key ``"metadata"``.
        """
        parquet_path = path / "metadata.parquet"
        csv_path = path / "metadata.csv"
        if parquet_path.exists():
            metadata = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            metadata = pd.read_csv(csv_path)
        else:
            metadata = pd.DataFrame()
        result = super().load_storage_context(path, **kwargs)
        result["metadata"] = metadata
        return result

    # ------------------------------------------------------------------
    # Other Utilities
    # ------------------------------------------------------------------

    def try_numerize_metadata(self, columns: Optional[List[str]] = None) -> None:
        """
        Attempt to numerize columns in the metadata DataFrame.

        NOTE: This is just a utility function for in-memory metadata saved as pandas dataframes.
        It is not a general utility for any storage backend.

        Parameters
        ----------
        columns : Optional[List[str]], default None
            The column(s) to numerize. If None, numerize all columns.
        """
        if columns is None:
            columns = self._metadata.columns.tolist()
        else:
            if isinstance(columns, str):
                columns = [columns]

        for col in columns:
            if col in self._metadata.columns:
                try:
                    self._metadata[col] = pd.to_numeric(self._metadata[col])
                except (ValueError, TypeError):
                    pass