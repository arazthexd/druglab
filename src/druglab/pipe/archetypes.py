"""
druglab.pipe.archetypes
~~~~~~~~~~~~~~~~~~~~~~~
Standard block types: Featurizers, Filters, Preparations, IO.
"""

import numpy as np
from typing import Any, List, Optional, Callable
from druglab.db.table import BaseTable
from druglab.pipe.base import ItemBlock, BaseBlock

class BaseFeaturizer(ItemBlock):
    """Calculates features and adds them to the table's feature dictionary."""
    
    def _apply_results(self, table: BaseTable, results: List[Any]) -> BaseTable:
        # Convert list of vectors into a numpy array
        feature_array = np.array(results)
        
        # Get feature name to save into the table
        feat_name = self.get_feature_name()
        
        # USE NEW STRICT API: update_feature
        table.update_feature(feat_name, feature_array)
        
        return table

    def get_feature_name(self) -> str:

        # Name the feature cleanly based on the config to track variations,
        # but ignore execution-level parameters that don't change the math.
        ignore_keys = {"name", "n_workers", "use_cache", "batch_size", "copy_table"}
        safe_config = "_".join(
            [f"{k}={v}" for k, v in self.get_config().items() if k not in ignore_keys]
        )
        feat_name = f"{self.name}_{safe_config}" if safe_config else self.name
        return feat_name

class BaseFilter(ItemBlock):
    """Evaluates items and drops rows where _process_item returns False."""

    def _apply_results(self, table: BaseTable, results: List[bool]) -> BaseTable:
        mask = np.array(results, dtype=bool)
        # Leverage BaseTable's built-in subset logic
        return table.subset(mask, copy_objects=False)


class BasePreparation(ItemBlock):
    """Modifies the items themselves (e.g., standardizing, desalting)."""

    def _apply_results(self, table: BaseTable, results: List[Any]) -> BaseTable:
        # Replaces the objects. (Assumes 1:1 mapping. If enumeration is needed, 
        # this logic would need to expand to adjust metadata/features lengths).
        table.objects = results 
        return table


class IOBlock(BaseBlock):
    """
    Abstract base for blocks that pull data from druglab.io into a BaseTable.
    They do not take input tables.
    """
    def _process(self, table: Optional[BaseTable]) -> BaseTable:
        if table is not None:
            import warnings
            warnings.warn(f"{self.name} is an IO Block but received an input table. The input will be ignored.")
        return self._load_table()

    def _load_table(self) -> BaseTable:
        raise NotImplementedError
    
# ---------------------------------------------------------------------------
# Functional Wrappers for Quick Block Creation
# ---------------------------------------------------------------------------

class FunctionFeaturizer(BaseFeaturizer):
    """Creates a featurizer block from a simple python function."""
    def __init__(self, func: Callable[[Any], Any], name: Optional[str] = None, **kwargs):
        super().__init__(name=name or func.__name__, **kwargs)
        self.func = func

    def _process_item(self, item: Any) -> Any:
        return self.func(item)


class FunctionFilter(BaseFilter):
    """Creates a filter block from a boolean python function."""
    def __init__(self, func: Callable[[Any], bool], name: Optional[str] = None, **kwargs):
        super().__init__(name=name or func.__name__, **kwargs)
        self.func = func

    def _process_item(self, item: Any) -> bool:
        return self.func(item)


class FunctionPreparation(BasePreparation):
    """Creates a preparation block from a python function that modifies the item."""
    def __init__(self, func: Callable[[Any], Any], name: Optional[str] = None, **kwargs):
        super().__init__(name=name or func.__name__, **kwargs)
        self.func = func

    def _process_item(self, item: Any) -> Any:
        return self.func(item)