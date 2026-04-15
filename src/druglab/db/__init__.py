"""
druglab.db
~~~~~~~~~~
Database-like table structures for groups of molecules and reactions.

Classes
-------
BaseTable
    Abstract base. Owns the four-property contract:
    - objects   : List[T]                  — the molecules / reactions
    - metadata  : pd.DataFrame             — scalar properties, row-aligned
    - features  : Dict[str, np.ndarray]    — large vector arrays, row-aligned
    - history   : List[HistoryEntry]       — append-only pipeline audit log

MoleculeTable
    Concrete subclass for RDKit Mol objects.

ReactionTable
    Concrete subclass for RDKit ChemicalReaction objects.

ConformerTable
    Subclass of MoleculeTable where every row holds exactly one conformer.
    Created via ``MoleculeTable.unroll_conformers()`` and collapsed back
    via ``ConformerTable.collapse()`` or ``MoleculeTable.update_from_conformers()``.

Multi-axis Indexing Constants
------------------------------
Use these with ``table[AXIS, ...]`` for backend query pushdown:

    META = M = 'metadata'
    OBJ  = O = 'object'
    FEAT = F = 'feature'

    table[FEAT, 'fps', 0:100]        # load only rows 0-99
    table[META, ['MolWt'], 5]        # single metadata row
    table[OBJ, 3]                    # single object

Storage Backends
----------------
    BaseStorageBackend   — abstract interface
    EagerMemoryBackend   — fully in-memory (default)
"""

from druglab.db.base import BaseTable, HistoryEntry, META, M, OBJ, O, FEAT, F
from druglab.db.molecule import MoleculeTable
from druglab.db.reaction import ReactionTable
from druglab.db.conformer import ConformerTable
from druglab.db.backends import (
    BaseStorageBackend,
    EagerMemoryBackend,
    MemoryMetadataMixin,
    MemoryObjectMixin,
    MemoryFeatureMixin,
)

__all__ = [
    # Core
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
]