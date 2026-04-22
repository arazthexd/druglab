"""
druglab.db.backend.base.mixins.metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Abstract base interface for metadata handling in storage backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union, Dict, TYPE_CHECKING

import numpy as np
import pandas as pd

from ._lifecycle import _LifecycleBase

if TYPE_CHECKING:
    from druglab.db.indexing import INDEX_LIKE

__all__ = ['BaseMetadataMixin']

class BaseMetadataMixin(_LifecycleBase, ABC):
    """Mixin for metadata handling in backends."""

    @abstractmethod
    def get_metadata(
        self,
        idx: Optional[INDEX_LIKE] = None,
        cols: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """
        Fetch metadata rows with strict query pushdown.

        Parameters
        ----------
        idx
            Row selector. ``None`` → all rows.
            Accepts ``int``, ``slice``, ``List[int]``, or ``np.ndarray``.
        cols
            Column selector. ``None`` → all columns.
            Accepts a single column name or a list of names.

        Returns
        -------
        pd.DataFrame
            Always a DataFrame (even when a single row/column is selected).
        """

    @abstractmethod
    def add_metadata_column(
        self,
        name: str,
        value: Union[pd.Series, np.ndarray, List[Any]],
        idx: Optional[INDEX_LIKE] = None,
        na: Any = None,
        **kwargs
    ) -> None:
        """
        Schema Evolution: Add a completely new metadata column to the backend.

        Parameters
        ----------
        name : str
            The name of the new column to create.
        value : Union[pd.Series, np.ndarray, List[Any]]
            The data for the new column. If *idx* is None, this must have the 
            same length as the backend. If *idx* is provided, it must match 
            the elements in *idx*. Pandas indices are ignored.
        idx : Optional[INDEX_LIKE], default None
            Row selector. If provided, the new column will be populated with 
            *value* at these specific indices, and all other rows will be 
            filled with the *na* value.
        na : Any, default None
            The value to use for rows not specified by *idx*. If None, the 
            backend should infer an appropriate null type based on *value*.
        """

    def add_metadata_columns(
        self,
        columns: Dict[str, Union[pd.Series, np.ndarray, List[Any]]],
        idx: Optional[INDEX_LIKE] = None,
        na: Any = None,
        **kwargs
    ) -> None:
        """
        Schema Evolution: Add multiple new metadata columns to the backend.

        Can be overridden to provide a more efficient bulk-insert implementation.

        Parameters
        ----------
        columns : Dict[str, Union[pd.Series, np.ndarray, List[Any]]]
            Dictionary mapping new column names to their data arrays.
        idx : Optional[INDEX_LIKE], default None
            Row selector applied to all incoming arrays.
        na : Any, default None
            The value to use for missing rows.
        """
        for name, value in columns.items():
            self.add_metadata_column(name, value, idx=idx, na=na, **kwargs)

    @abstractmethod
    def update_metadata(
        self,
        values: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
        idx: Optional[INDEX_LIKE] = None,
        **kwargs
    ) -> None:
        """
        Perform a partial, in-place update of *existing* metadata columns.

        This method should raise a KeyError (or backend-equivalent) if the user 
        attempts to update a column that does not exist. Use `add_metadata_column(s)` 
        to alter the schema.

        Parameters
        ----------
        values : Union[pd.DataFrame, pd.Series, Dict[str, Any]]
            The new data to insert. Keys/columns must match existing metadata. 
            Pandas indices are ignored; updates rely strictly on positional alignment.
        idx : Optional[INDEX_LIKE], default None
            Row selector. ``None`` → apply to all rows.
        """

    @abstractmethod
    def drop_metadata_columns(
        self,
        cols: Optional[Union[str, List[str]]] = None
    ) -> None:
        """
        Drop metadata columns given as *cols*. If *cols* is None, drop all columns.

        Parameters
        ----------
        cols : Optional[Union[str, List[str]]], default None
            Column or list of columns to drop. If None, drop all columns, 
            effectively resetting the metadata schema.
        """

    def set_metadata(self, df: pd.DataFrame) -> None:
        """
        Replace the entire metadata store with *df*.

        This drops all existing metadata and replaces it with the schema and 
        values of the provided DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The new DataFrame that will completely overwrite the existing metadata.
        """
        if self._n_metadata_rows() != df.shape[0]:
            raise ValueError(
                f"metadata has {df.shape[0]} rows but backend has {self._n_metadata_rows()} objects"
            )
        self.drop_metadata_columns()
        
        # Convert DataFrame to dictionary of arrays for efficient batch insertion
        column_dict = {col: df[col].values for col in df.columns}
        self.add_metadata_columns(column_dict)

    @abstractmethod
    def _n_metadata_rows(self) -> int:
        """Return the number of rows in the metadata table."""

    def _validate_metadata(self) -> None:
        """Validate the backend's metadata schema."""
        return
    
    def post_initialize_validate(self) -> None:
        """Validate metadata domain after full init; then propagate."""
        self._validate_metadata()
        super().post_initialize_validate()