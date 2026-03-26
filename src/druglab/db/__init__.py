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
"""

from druglab.db.base import BaseTable, HistoryEntry
from druglab.db.molecule import MoleculeTable
from druglab.db.reaction import ReactionTable

__all__ = [
    "BaseTable",
    "HistoryEntry",
    "MoleculeTable",
    "ReactionTable",
]