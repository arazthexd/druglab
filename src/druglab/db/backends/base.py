"""
druglab.db.backends.base
~~~~~~~~~~~~~~~~~~~~~~~~
Abstract base interface that all storage backends must implement.

The interface enforces strict Query Pushdown: index/slice arguments must be
passed directly to the backend so that out-of-core implementations (Zarr,
SQLite, HDF5) can read exactly the bytes they need without loading full
arrays into memory first.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Type alias for index arguments
# ---------------------------------------------------------------------------

INDEX_LIKE = Union[int, slice, List[int]]


class BaseStorageBackend(ABC):
    """
    The single unified interface for managing DrugLab table state.

    All concrete backends implement this interface.  The ``idx`` arguments
    in every read method are mandatory in the API so that future disk-backed
    implementations can push the selection all the way to the I/O layer
    rather than loading an entire array and then slicing it in Python.

    Backend implementations should interpret ``idx=None`` as "return all
    rows" and should handle ``int``, ``slice``, and ``List[int]`` forms.
    """

    # ------------------------------------------------------------------
    # Metadata API
    # ------------------------------------------------------------------

    @abstractmethod
    def get_metadata(
        self,
        idx: Optional[INDEX_LIKE] = None,
        cols: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """
        Fetch metadata rows.

        Parameters
        ----------
        idx
            Row selector.  ``None`` → all rows.
            Accepts ``int``, ``slice``, or ``List[int]``.
        cols
            Column selector.  ``None`` → all columns.
            Accepts a single column name or a list of names.

        Returns
        -------
        pd.DataFrame
            Always a DataFrame (even when a single row/column is selected).
        """

    @abstractmethod
    def update_metadata(self, df: pd.DataFrame) -> None:
        """Replace the entire metadata store with *df*."""

    # ------------------------------------------------------------------
    # Object API
    # ------------------------------------------------------------------

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of stored objects."""

    @abstractmethod
    def get_objects(self, idx: INDEX_LIKE) -> Union[Any, List[Any]]:
        """
        Fetch one or multiple objects with backend-level query pushdown.

        Parameters
        ----------
        idx
            ``int``        → return a single object.
            ``slice``      → return a list of objects.
            ``List[int]``  → return a list of objects in the specified order.
        """

    @abstractmethod
    def put_object(self, index: int, obj: Any) -> None:
        """Overwrite the object at *index*."""

    # ------------------------------------------------------------------
    # Feature API
    # ------------------------------------------------------------------

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
            Row selector.  ``None`` → return the full array.
            Accepts ``int``, ``slice``, or ``List[int]``.

        Returns
        -------
        np.ndarray
            The (possibly subset) feature array.
        """

    @abstractmethod
    def add_feature(self, name: str, array: np.ndarray) -> None:
        """Add or overwrite a feature array."""

    @abstractmethod
    def drop_feature(self, name: str) -> None:
        """Remove a feature by name.  Raises ``KeyError`` if absent."""

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return the list of stored feature keys."""

    # ------------------------------------------------------------------
    # State & Synchronization
    # ------------------------------------------------------------------

    @abstractmethod
    def create_view(self, indices: Sequence[int]) -> "BaseStorageBackend":
        """
        Return a synchronized, independent view of this backend restricted
        to *indices*.

        The view contains copies of the selected metadata rows, objects, and
        feature sub-arrays.  Mutating the view must **not** affect the
        original backend and vice-versa.

        Parameters
        ----------
        indices
            Ordered sequence of integer row indices.

        Returns
        -------
        BaseStorageBackend
            A new backend instance of the same concrete type.
        """

    @abstractmethod
    def save(self, path: Path, serializer: Optional[Callable] = None) -> None:
        """
        Persist backend data inside a ``.dlb`` bundle directory.

        Parameters
        ----------
        path
            The bundle directory (already created by the orchestrating
            ``BaseTable.save()``).  The backend writes its data files here.
        serializer
            Optional callable ``(obj) -> bytes`` for object serialisation.
            If ``None`` the backend uses its own default strategy.
        """