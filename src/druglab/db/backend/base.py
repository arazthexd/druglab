"""
druglab.db.backend.base
~~~~~~~~~~~~~~~~~~~~~~~~
Abstract base interface that all storage backends must implement.

The interface enforces strict Query Pushdown: index/slice arguments must be
passed directly to the backend so that out-of-core implementations (Zarr,
SQLite, HDF5) can read exactly the bytes they need without loading full
arrays into memory first.

Index normalisation is handled by ``druglab.db.indexing``, which is the
single source of truth for all row-addressing in DrugLab.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Union, Dict

import numpy as np
import pandas as pd

# Re-export INDEX_LIKE from the canonical indexing module so that existing
# imports of ``INDEX_LIKE`` from this module continue to work unchanged.
from druglab.db.indexing import (
    INDEX_LIKE,
    RowSelection,
    normalize_row_index,
    coerce_bool_mask,
    validate_take_index,
)

__all__ = [
    "INDEX_LIKE",
    "RowSelection",
    "normalize_row_index",
    "coerce_bool_mask",
    "validate_take_index",
    "BaseMetadataMixin",
    "BaseObjectMixin",
    "BaseFeatureMixin",
    "BaseStorageBackend",
]


class BaseMetadataMixin(ABC):
    """
    Mixin for metadata handling in backends.
    """

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


class BaseObjectMixin(ABC):
    """
    Mixin for object handling in backends.
    """

    @abstractmethod
    def get_objects(
        self,
        idx: Optional[INDEX_LIKE] = None
    ) -> Union[Any, List[Any]]:
        """
        Fetch one or multiple objects with backend-level query pushdown.

        Parameters
        ----------
        idx
            ``int``        → return a single object.
            ``slice``      → return a list of objects.
            ``List[int]``  → return a list of objects in the specified order.
            ``np.ndarray`` → return a list of objects based on boolean/int mask.
            ``None``       → return all objects.
            
        Returns
        -------
        Union[Any, List[Any]]
            A single object if idx is an int, otherwise a list of objects.
        """

    @abstractmethod
    def update_objects(
        self,
        objs: Union[Any, List[Any]],
        idx: Optional[INDEX_LIKE] = None,
        **kwargs
    ) -> None:
        """
        Perform a partial or full update of stored objects.

        This method replaces the legacy `put_object` to enforce vector-first 
        writes, eliminating N+1 query bottlenecks in out-of-core backends.

        Parameters
        ----------
        objs : Union[Any, List[Any]]
            The new object(s) to insert. If `idx` is an integer, this should be 
            a single object. Otherwise, it must be a sequence of objects matching 
            the length of the resolved index.
        idx : Optional[INDEX_LIKE], default None
            Row selector. ``None`` → apply to all rows (length of objs must match 
            length of the backend).
        """

    def set_objects(self, objs: List[Any], **kwargs) -> None:
        """
        Replace the entire object store with a new list of objects.

        Parameters
        ----------
        objs : List[Any]
            The new list of objects that will completely overwrite the existing store.
        """
        if self._n_objects() != len(objs):
            raise ValueError(
                f"new objects has {len(objs)} items but backend has {self._n_objects()}"
            )
        self.update_objects(objs, **kwargs)

    @abstractmethod
    def _n_objects(self) -> int:
        """Return the number of rows in the object table."""

    def _validate_objects(self) -> None:
        """Validate the backend's object schema."""
        return


class BaseFeatureMixin(ABC):
    """
    Mixin for feature handling in backends.
    """

    @abstractmethod
    def get_feature(
        self,
        name: str,
        idx: Optional[INDEX_LIKE] = None,
    ) -> np.ndarray:
        """
        Fetch a feature array with strict query pushdown.

        Parameters
        ----------
        name
            Feature key.
        idx
            Row selector.  If None, return the full array. Otherwise,
            selected rows will only be returned.

        Returns
        -------
        np.ndarray
            The (possibly subset) feature array.
        """

    def get_features(
        self,
        names: Optional[List[str]] = None,
        idx: Optional[INDEX_LIKE] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Fetch multiple feature arrays with strict query pushdown.
        
        Can be overridden to provide a more efficient implementation.
        
        Parameters
        ----------
        names
            List of feature keys.
        idx
            Row selector.  If None, return the full arrays. Otherwise,
            selected rows will only be returned for each feature.
        
        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary mapping feature keys to feature arrays.
        """

        if names is None:
            names = self.get_feature_names()
        return {name: self.get_feature(name, idx) for name in names}

    @abstractmethod
    def update_feature(
        self,
        name: str,
        array: np.ndarray,
        idx: Optional[INDEX_LIKE] = None,
        na: Any = None,
        **kwargs
    ) -> None:
        """Add or perform a partial, in-place update of a feature array."""

    def update_features(
        self,
        arrays: Dict[str, np.ndarray],
        idx: Optional[INDEX_LIKE] = None,
        na: Any = None,
        **kwargs
    ) -> None:
        """Add or perform a partial, in-place update of multiple feature arrays."""
        for name, array in arrays.items():
            self.update_feature(name, array, idx, na, **kwargs)

    @abstractmethod
    def drop_feature(self, name: str) -> None:
        """Remove a feature by name. Raises ``KeyError`` if absent."""

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return the list of stored feature keys."""

    @abstractmethod
    def get_feature_shape(self, name: str) -> tuple:
        """Return the shape of a feature array."""

    def _n_feature_rows(self) -> int:
        """Return the number of rows in each feature array."""
        if not self.get_feature_names():
            return len(self)
        return self.get_feature_shape(self.get_feature_names()[0])[0]

    def _validate_features(self) -> None:
        """Validate the backend's feature schema."""
        if not self.get_feature_names():
            return
        n = self._n_feature_rows()
        for name in self.get_feature_names():
            if n != self.get_feature_shape(name)[0]:
                raise ValueError(
                    f"Feature '{name}' has {self.get_feature_shape(name)[0]} rows, "
                    f"expected {n}"
                )


class BaseStorageBackend(
    BaseMetadataMixin,
    BaseObjectMixin,
    BaseFeatureMixin
):
    """
    The single unified interface for managing DrugLab table state.
    """

    def validate(self) -> None:
        """
        STRONGLY SUGGESTED: Validates the entire backend by checking
        individual domain integrity and ensuring dimension alignment.
        """
        # 1. Get the global, official length (provided by the concrete backend)
        expected_len = len(self)

        # 2. Run domain-specific validations and capture their lengths
        self._validate_metadata()
        self._validate_features()
        self._validate_objects()

        meta_len = self._n_metadata_rows()
        feat_len = self._n_feature_rows()
        obj_len = self._n_objects()

        # 3. Cross-validate lengths
        if not (expected_len == meta_len == feat_len == obj_len):
            raise ValueError(
                f"Backend Dimension Mismatch!\n"
                f"Global Length: {expected_len}\n"
                f"Metadata Rows: {meta_len}\n"
                f"Feature Rows:  {feat_len}\n"
                f"Object Count:  {obj_len}"
            )
        
        # 4. Any other global state validations can go here...
        return