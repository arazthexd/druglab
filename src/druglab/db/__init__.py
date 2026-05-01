"""
druglab.db
~~~~~~~~~~
Database-like table structures for groups of molecules and reactions.
"""

from .table import (
    BaseTable,
    HistoryEntry,
    MoleculeTable,
    ReactionTable,
    ConformerTable,
    META,
    M,
    OBJ,
    O,
    FEAT,
    F,
)
from .backend import (
    BaseStorageBackend,
    CompositeStorageBackend,
    EagerMemoryBackend,
    MemoryMetadataStore,
    MemoryObjectStore,
    MemoryFeatureStore,
)
from .indexing import (
    INDEX_LIKE,
    RowSelection,
    normalize_row_index,
    coerce_bool_mask,
    validate_take_index,
)

__all__ = [
    # Core tables
    "BaseTable",
    "HistoryEntry",
    "MoleculeTable",
    "ReactionTable",
    "ConformerTable",
    # Indexing constants
    "META",
    "M",
    "OBJ",
    "O",
    "FEAT",
    "F",
    # Backends
    "BaseStorageBackend",
    "CompositeStorageBackend",
    "EagerMemoryBackend",
    "MemoryMetadataStore",
    "MemoryObjectStore",
    "MemoryFeatureStore",
    # Indexing utilities
    "INDEX_LIKE",
    "RowSelection",
    "normalize_row_index",
    "coerce_bool_mask",
    "validate_take_index",
]