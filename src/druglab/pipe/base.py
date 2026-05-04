"""
druglab.pipe.base
~~~~~~~~~~~~~~~~~
Base classes for all pipeline building blocks.
"""

from __future__ import annotations

import copy
import hashlib
import json
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from druglab.db.backend.overlay import OverlayBackend
from druglab.db.table import BaseTable, HistoryEntry
from druglab.pipe.cache import BaseCache, default_cache

def _chunk_index_ranges(n_rows: int, n_chunks: int) -> List[np.ndarray]:
    """Split ``range(n_rows)`` into up to ``n_chunks`` contiguous index arrays."""
    if n_rows <= 0 or n_chunks <= 0:
        return []
    boundaries = np.linspace(0, n_rows, num=min(n_chunks, n_rows) + 1, dtype=np.intp)
    return [
        np.arange(boundaries[i], boundaries[i + 1], dtype=np.intp)
        for i in range(len(boundaries) - 1)
        if boundaries[i] < boundaries[i + 1]
    ]

def _process_itemblock_chunk(args: Tuple["ItemBlock", BaseTable]) -> BaseTable:
    """Worker entrypoint for ``ItemBlock`` multiprocessing scatter-gather."""
    block, chunk_table = args
    return block._process_chunk(chunk_table)

class BaseBlock(ABC):
    """
    The absolute base for any pipeline block.
    """

    required_features: List[str] = []
    required_metadata: List[str] = []

    def __init__(
        self,
        name: Optional[str] = None,
        n_workers: int = 1,
        batch_size: Optional[int] = None,
        use_cache: bool = False,
        copy_table: bool = True,
    ):
        self.name = name or self.__class__.__name__
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.copy_table = copy_table

    def get_config(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary of the block's config."""
        return {
            "name": self.name,
            "n_workers": self.n_workers,
            "batch_size": self.batch_size,
            "use_cache": self.use_cache,
        }

    def run(self, table: Optional[BaseTable] = None) -> BaseTable:
        """
        Main execution point. Handles table copying and history appending.
        """
        t = table.copy() if self.copy_table and table is not None else table
        rows_in = len(t) if t is not None else 0

        # Execute the core logic defined by subclasses
        out_table = self._process(t)
        if out_table is None:
            raise ValueError(
                f"{self.__class__.__name__} returned None as its block output. "
                "This is likely a bug in the block implementation."
            )
        if not isinstance(out_table, BaseTable):
            raise TypeError(
                f"{self.__class__.__name__} must return a subclass of BaseTable, "
                f"got {type(out_table).__name__}."
            )

        rows_out = len(out_table) if out_table is not None else 0
        if out_table is not None:
            out_table.append_history(
                HistoryEntry.now(
                    operation=self.__class__.__name__,
                    config=self.get_config(),
                    rows_in=rows_in,
                    rows_out=rows_out,
                )
            )
        return out_table
    
    @property
    def prefetch_features(self) -> List[str]:
        """Feature names the orchestrator should prefetch before detaching overlays."""
        return list(self.required_features)

    @property
    def prefetch_metadata(self) -> List[str]:
        """Metadata columns the orchestrator should prefetch before detaching overlays."""
        return list(self.required_metadata)

    @abstractmethod
    def _process(self, table: Optional[BaseTable]) -> BaseTable:
        """Subclasses implement their specific logic here."""
        pass

    def yield_batches(self) -> Iterator[BaseTable]:
        """
        Used primarily by IO blocks when `batch_size` is declared.
        Must yield constructed BaseTable chunks.
        """
        raise NotImplementedError(f"{self.name} does not support yielding batches directly.")


class ItemBlock(BaseBlock):
    """
    A block that processes items one-by-one (with optional multiprocessing).
    Handles the boilerplate for caching and parallel execution.
    """

    def __init__(self, cache_backend: Optional[BaseCache] = None, **kwargs):
        super().__init__(**kwargs)
        self.cache = cache_backend or default_cache

    def _get_item_key(self, item: Any) -> str:
        """
        Generate a unique cache key for an item + this block's config.
        Subclasses dealing with RDKit Mols might override this to use SMILES.
        """
        config_hash = hashlib.md5(json.dumps(self.get_config(), sort_keys=True).encode()).hexdigest()
        item_hash = hash(str(item))
        return f"{self.name}_{config_hash}_{item_hash}"

    def _iter_objects_with_cache(
        self,
        table: BaseTable,
    ) -> Tuple[List[Tuple[int, Any]], List[Any], List[int]]:
        """Return cached results plus uncached objects and row indices."""
        cached_results: List[Tuple[int, Any]] = []
        items_to_process: List[Any] = []
        indices_to_process: List[int] = []

        for i in range(table.n):
            item = table.get_objects(i)
            if self.use_cache:
                key = self._get_item_key(item)
                cached = self.cache.get(key)
                if cached is not None:
                    cached_results.append((i, cached))
                    continue
            items_to_process.append(item)
            indices_to_process.append(i)

        return cached_results, items_to_process, indices_to_process

    def _process_chunk(self, table: BaseTable) -> BaseTable:
        """Run item-wise processing and apply results on a single table chunk."""
        _, items_to_process, _ = self._iter_objects_with_cache(table)
        results = [self._process_item(item) for item in items_to_process]
        if self.use_cache:
            for item, res in zip(items_to_process, results):
                self.cache.set(self._get_item_key(item), res)
        return self._apply_results(table, results)

    def _process(self, table: Optional[BaseTable]) -> BaseTable:
        if table is None:
            raise ValueError(f"{self.name} requires an input table.")

        if self.n_workers <= 1:
            cached_results, items_to_process, indices_to_process = self._iter_objects_with_cache(table)
            computed_results = [self._process_item(x) for x in items_to_process]

            if self.use_cache:
                for item, res in zip(items_to_process, computed_results):
                    self.cache.set(self._get_item_key(item), res)

            ordered_pairs = cached_results + list(zip(indices_to_process, computed_results))
            ordered_pairs.sort(key=lambda x: x[0])
            return self._apply_results(table, [value for _, value in ordered_pairs])

        if self.use_cache:
            raise RuntimeError(
                "ItemBlock multiprocessing does not support shared cache writes. "
                "Set use_cache=False when n_workers > 1."
            )

        base_backend = table.backend
        chunk_indices = _chunk_index_ranges(table.n, self.n_workers)
        detached_chunks: List[BaseTable] = []

        for row_idx in chunk_indices:
            overlay_backend = OverlayBackend(base_backend, row_idx)
            overlay_table = table._new_instance_from_backend(overlay_backend, history=list(table.history))
            overlay_backend.prefetch(
                features=self.prefetch_features or None,
                meta_cols=self.prefetch_metadata or None,
            )
            # ItemBlock always reads objects. Prime object delta before detach.
            overlay_table.update_objects(overlay_table.get_objects())
            overlay_backend.detach()
            detached_chunks.append(overlay_table)

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            processed_chunks = list(
                executor.map(
                    _process_itemblock_chunk,
                    [(self, chunk) for chunk in detached_chunks],
                )
            )

        for chunk_table in processed_chunks:
            if not isinstance(chunk_table.backend, OverlayBackend):
                raise TypeError("Multiprocessing worker must return a table backed by OverlayBackend.")
            chunk_table.backend.attach(base_backend)

        for chunk_table in processed_chunks:
            chunk_table.commit()

        return table

    @abstractmethod
    def _process_item(self, item: Any) -> Any:
        """Process a single item. Executed in worker processes."""
        pass

    @abstractmethod
    def _apply_results(self, table: BaseTable, results: List[Any]) -> BaseTable:
        """Merge the calculated list of results back into the table."""
        pass