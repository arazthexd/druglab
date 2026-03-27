"""
druglab.pipe
~~~~~~~~~~~~
Composable processing blocks and pipelines for DrugLab tables.
"""

from druglab.pipe.cache import BaseCache, DictCache, default_cache
from druglab.pipe.base import BaseBlock, ItemBlock
from druglab.pipe.pipeline import Pipeline
from druglab.pipe.archetypes import (
    BaseFeaturizer,
    BaseFilter,
    BasePreparation,
    IOBlock,
    FunctionFeaturizer,
    FunctionFilter,
    FunctionPreparation,
)
from druglab.pipe.blocks import (
    MorganFeaturizer,
    MWFilter,
    KekulizePreparation,
    MemoryIOBlock,
)

__all__ = [
    # Core
    "Pipeline",
    "BaseBlock",
    "ItemBlock",
    "BaseCache",
    "DictCache",
    "default_cache",
    
    # Archetypes & Wrappers
    "BaseFeaturizer",
    "BaseFilter",
    "BasePreparation",
    "IOBlock",
    "FunctionFeaturizer",
    "FunctionFilter",
    "FunctionPreparation",
    
    # Example Implementations
    "MorganFeaturizer",
    "MWFilter",
    "KekulizePreparation",
    "MemoryIOBlock",
]