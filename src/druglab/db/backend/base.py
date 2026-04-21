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

# ===========================================================================
# Life Cycle Hooks (Clean MRO)
# ===========================================================================

class _LifecycleBase:
    """
    Mixin base that defines the three cooperative lifecycle hooks.
 
    All domain mixins and capability mixins inherit from this class so that
    ``super()``-based cooperative calls propagate correctly through any MRO.
 
    Hooks
    -----
    initialize_storage_context(**kwargs)
        Called early in ``__init__`` after each mixin's own state is ready.
        Use this to wire up storage-layer internals.  Must call
        ``super().initialize_storage_context`` first so the full chain fires.
 
    bind_capabilities()
        Called once after the entire ``__init__`` MRO chain completes, before
        ``post_initialize_validate``.  Must call ``super().bind_capabilities()``.
 
    post_initialize_validate()
        Called last, for cross-domain consistency assertions.  Must call
        ``super().post_initialize_validate()`` first.
    """
 
    def initialize_storage_context(self, **kwargs: Any) -> None:
        """
        Cooperative lifecycle hook: finalize mixin-level storage setup.
 
        Implementors must call ``super().initialize_storage_context(**kwargs)``
        before their own logic so the full cooperative chain fires.
 
        Unknown kwargs are intentionally swallowed at the terminal node so
        that cooperative chains don't break when different mixins consume
        different subsets of kwargs.
        """
        # Terminal node -- absorbs remaining kwargs.
 
    def bind_capabilities(self) -> None:
        """
        Cooperative lifecycle hook: wire up inter-mixin capability references.
 
        Called once after the full ``__init__`` chain completes, before
        ``post_initialize_validate``.  Must call ``super().bind_capabilities()``.
        """
        # Terminal node -- absorbs remaining kwargs.
 
    def post_initialize_validate(self) -> None:
        """
        Cooperative lifecycle hook: cross-domain consistency validation.
 
        Called last, after both prior hooks.  Raise ``ValueError`` to signal
        an invalid initial state.  Must call ``super().post_initialize_validate()``.
        """
        # Terminal node -- absorbs remaining kwargs.

# ===========================================================================
# Base Mixins
# ===========================================================================

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


class BaseObjectMixin(_LifecycleBase, ABC):
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
        Perform an in-place partial or full update of stored objects.

        Parameters
        ----------
        objs : Union[Any, List[Any]]
            The object or sequence of objects to insert.
        idx : Optional[INDEX_LIKE], default None
            The specific index/indices to overwrite. If None, the entire 
            internal list is replaced by `objs`.

        Raises
        ------
        ValueError
            If `idx` is a sequence but its length does not match `objs`.
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
        """
        Get the total number of stored objects.

        Returns
        -------
        int
            Length of the internal object list.
        """

    def _validate_objects(self) -> None:
        """Validate the backend's object schema."""
        return
    
    def post_initialize_validate(self) -> None:
        """Validate object domain after full init; then propagate."""
        self._validate_objects()
        super().post_initialize_validate()


class BaseFeatureMixin(_LifecycleBase, ABC):
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
            
    def post_initialize_validate(self) -> None:
        """Validate feature domain after full init; then propagate."""
        self._validate_features()
        super().post_initialize_validate()

# ===========================================================================
# Base Storage Backend
# ===========================================================================

class BaseStorageBackend(
    BaseMetadataMixin,
    BaseObjectMixin,
    BaseFeatureMixin
):
    """
    Minimal unified interface for managing DrugLab table state.
 
    Lifecycle orchestration
    ------------------------
    ``__init__`` fires three hooks in order after the cooperative MRO chain:
 
    1. ``initialize_storage_context(**kwargs)`` -- domain setup
    2. ``bind_capabilities()``                  -- inter-mixin wiring
    3. ``post_initialize_validate()``            -- consistency checks
 
    Concrete backends assembling multiple mixins do **not** need to override
    ``__init__`` for boilerplate: each mixin handles its own state, and the
    hooks handle the rest.
    """

    def __init__(self, **kwargs: Any) -> None:
        # This is the terminal node of the cooperative __init__ chain.
        # Domain mixins above us (MemoryObjectMixin, MemoryMetadataMixin, etc.)
        # each consume their own recognized kwarg (objects=, metadata=,
        # features=) and forward the rest via super().__init__(**remaining).
        # Any unrecognized kwargs that reach here are silently absorbed rather
        # than forwarded to object.__init__ (which accepts none).
        # This allows custom mixin __init__ methods to accept extra kwargs
        # (e.g. connection_string=) without breaking the chain.
        #
        # NOTE: do NOT call super().__init__(**kwargs) here -- object.__init__
        # rejects keyword arguments.
        # super().__init__() is intentionally NOT called with kwargs.
        # (object.__init__() takes no extra arguments.)
 
        # Fire lifecycle hooks in declared order.
        # Hooks receive the full original kwargs dict so specialized mixins
        # can consume what they need in initialize_storage_context.
        self.initialize_storage_context(**kwargs)
        self.bind_capabilities()
        self.post_initialize_validate()

    def validate(self) -> None:
        """
        Validate backend-wide dimensional consistency.

        STRONGLY SUGGESTED: Validates the entire backend by checking
        individual domain integrity and ensuring dimension alignment.

        Raises
        ------
        ValueError
            If any dimension mismatch is detected.
        """
        expected_len = len(self)
        meta_len = self._n_metadata_rows()
        feat_len = self._n_feature_rows()
        obj_len = self._n_objects()
 
        if not (expected_len == meta_len == feat_len == obj_len):
            raise ValueError(
                f"Backend Dimension Mismatch!\n"
                f"Global Length: {expected_len}\n"
                f"Metadata Rows: {meta_len}\n"
                f"Feature Rows:  {feat_len}\n"
                f"Object Count:  {obj_len}"
            )