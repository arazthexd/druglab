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
    EagerMemoryBackend,
    MemoryMetadataMixin,
    MemoryObjectMixin,
    MemoryFeatureMixin,
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
    "EagerMemoryBackend",
    "MemoryMetadataMixin",
    "MemoryObjectMixin",
    "MemoryFeatureMixin",
    # Indexing utilities
    "INDEX_LIKE",
    "RowSelection",
    "normalize_row_index",
    "coerce_bool_mask",
    "validate_take_index",
]