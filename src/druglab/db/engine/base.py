from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar
from typing_extensions import Self

from ..types import ColsLike, IdxLike

EngineIOT = TypeVar("EngineIOT")


class BaseEngine(ABC, Generic[EngineIOT]):
    """Abstract contract for all DrugLab storage backends.

    BaseEngine defines how data is read from, written to, and masked within 
    different storage technologies (e.g., DuckDB for tabular data, Zarr for 
    tensors). 

    Filtering Model:
        *   **Rows:** Engines maintain a persistent row mask per namespace that 
            restricts visibility. 
        *   **Columns:** Filtering is applied dynamically during materialization; 
            engines do not cache column selections.

    Implementation Guide for Subclasses:
        When implementing a new engine (e.g., `ZarrEngine`):
        1. `n_rows` must return the physical (unmasked) count of the first axis.
        2. `materialize` should map `rows` to axis-0 and `cols` to axis-1 (if 
           applicable).
        3. `export` should re-align indices so the resulting storage is 
           contiguous starting from index 0.

    Note:
        For tensor engines, if `cols` is passed for an array with more than two 
        dimensions where axis-1 isn't a feature dimension, the implementation 
        should raise a `NotImplementedError`.
    """

    @abstractmethod
    def n_rows(self, namespace: str, what: str) -> int:
        """Returns the physical, unmasked row count for a given table.

        This count represents the total number of items stored in the backend 
        before any view-level masking is applied. It is used primarily to 
        resolve open-ended slices (e.g., `slice(5, None)`).

        Args:
            namespace: Logical table group (e.g., "molecules").
            what: Sub-table identifier (e.g., "fingerprints").

        Returns:
            int: The total number of rows physically present in storage.
        """
        pass

    @abstractmethod
    def materialize(
        self,
        namespace: str,
        what: str,
        rows: Optional[IdxLike] = None,
        cols: Optional[ColsLike] = None,
        *args,
        **kwargs,
    ) -> EngineIOT:
        """Reads data from the backend, applying row and column filters.

        Args:
            namespace: Logical table group (e.g., "molecules").
            what: Sub-table identifier (e.g., "metadata").
            rows: Additional row mask applied on top of the engine's current 
                internal mask. If None, uses all rows visible to the engine.
            cols: List of column names or feature indices to select. 
                If None, retrieves all available columns/features.
            *args: Engine-specific positional arguments.
            **kwargs: Engine-specific keyword arguments.

        Returns:
            EngineIOT: The retrieved data in the engine's native format 
                (e.g., pd.DataFrame, np.ndarray, or zarr.Array).
        """
        pass

    @abstractmethod
    def spawn_view(self, namespace: str, rows: IdxLike) -> Self:
        """Returns a new engine instance restricted to a subset of rows.

        Args:
            namespace: The namespace to which the mask should be applied.
            rows: The index mask (slice or array) defining the new view.

        Returns:
            Self: A new engine instance with the composed row mask.
        """
        pass

    @abstractmethod
    def write(self, namespace: str, what: str, data: EngineIOT, **kwargs) -> None:
        """Writes or appends data to the underlying storage.

        Args:
            namespace: Logical table group.
            what: Sub-table identifier.
            data: Data to be written, matching the engine's expected format.
            **kwargs: Additional parameters such as compression or partitioning.
        """
        pass

    @abstractmethod
    def export(
        self,
        target: Optional[str] = None,
        namespaces: Optional[List[str]] = None,
    ) -> Self:
        """Materializes the current masked state into a new independent storage.

        Unlike `spawn_view`, which creates a virtual mask, `export` creates a 
        physically new version of the data where the current visible rows are 
        stored contiguously from index 0.

        Args:
            target: Path or URI for the new storage location.
            namespaces: Optional list of namespaces to include in the export. 
                If None, all namespaces are exported.

        Returns:
            Self: A new engine instance pointing to the exported data.
        """
        pass

    def _combine_masks(
        self,
        current_mask: Optional[IdxLike],
        new_mask: IdxLike,
        n_rows: int,
    ) -> IdxLike:
        """Intersects two masks and resolves relative indexing.

        This helper handles the logic of applying a new mask to an already 
        masked view. It resolves open-ended slices using the physical `n_rows` 
        to ensure consistency.

        Args:
            current_mask: The mask currently active on the engine.
            new_mask: The new mask to apply (relative to the current mask).
            n_rows: The physical size of the table (from `self.n_rows`).

        Returns:
            IdxLike: A combined mask representing the intersection of both.

        Example:
            If `current_mask` is `[10, 11, 12]` and `new_mask` is `[0, 2]`, 
            the result is `[10, 12]`.
        """
        import numpy as np

        def _resolve(mask: IdxLike) -> np.ndarray:
            if isinstance(mask, slice):
                return np.arange(n_rows)[mask]
            return np.asarray(mask, dtype=np.intp)

        if current_mask is None:
            return new_mask

        curr_arr = _resolve(current_mask)

        if isinstance(new_mask, slice):
            return curr_arr[new_mask]
        else:
            return curr_arr[np.asarray(new_mask, dtype=np.intp)]