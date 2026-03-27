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
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from druglab.db.base import BaseTable, HistoryEntry
from druglab.pipe.cache import BaseCache, default_cache


class BaseBlock(ABC):
    """
    The absolute base for any pipeline block.
    """

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

        rows_out = len(out_table) if out_table is not None else 0
        if out_table is not None:
            out_table.append_history(
                HistoryEntry.now(
                    block_name=self.__class__.__name__,
                    config=self.get_config(),
                    rows_in=rows_in,
                    rows_out=rows_out,
                )
            )
        return out_table

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
        # Default fallback: hash the object string representation + config
        config_hash = hashlib.md5(json.dumps(self.get_config(), sort_keys=True).encode()).hexdigest()
        item_hash = hash(str(item)) 
        return f"{self.name}_{config_hash}_{item_hash}"

    def _process(self, table: Optional[BaseTable]) -> BaseTable:
        if table is None:
            raise ValueError(f"{self.name} requires an input table.")

        results = []
        items_to_process = []
        indices_to_process = []

        # 1. Check cache for all items
        for i, item in enumerate(table.objects):
            if self.use_cache:
                key = self._get_item_key(item)
                cached = self.cache.get(key)
                if cached is not None:
                    results.append((i, cached))
                    continue
            
            items_to_process.append(item)
            indices_to_process.append(i)

        # 2. Multiprocessing execution for cache misses
        if items_to_process:
            if self.n_workers > 1:
                with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                    new_results = list(executor.map(self._process_item, items_to_process))
            else:
                new_results = [self._process_item(x) for x in items_to_process]

            # 3. Save to cache
            if self.use_cache:
                for item, res in zip(items_to_process, new_results):
                    self.cache.set(self._get_item_key(item), res)

            # Re-align with original indices
            for idx, res in zip(indices_to_process, new_results):
                results.append((idx, res))

        # 4. Sort results back to original table order
        results.sort(key=lambda x: x[0])
        ordered_results = [r[1] for r in results]

        # 5. Delegate applying results to specific subclasses (Featurizer, Filter, etc.)
        return self._apply_results(table, ordered_results)

    @abstractmethod
    def _process_item(self, item: Any) -> Any:
        """Process a single item. Executed in worker processes."""
        pass

    @abstractmethod
    def _apply_results(self, table: BaseTable, results: List[Any]) -> BaseTable:
        """Merge the calculated list of results back into the table."""
        pass