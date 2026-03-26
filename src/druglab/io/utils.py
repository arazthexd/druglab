"""
Helper utilities for working with I/O records natively.
"""

from typing import Iterable, List, Optional
from druglab.io._record import MoleculeRecord, ReactionRecord

def get_mols(records: Iterable[MoleculeRecord], drop_invalid: bool = True) -> List[Optional[object]]:
    """
    Extract a raw list of RDKit Mol objects from an iterable of MoleculeRecords.
    
    Parameters
    ----------
    records:
        Iterable of MoleculeRecord objects (e.g. from BatchReader or read_file).
    drop_invalid:
        If True, None values (failed parses) are omitted from the output.
    """
    if drop_invalid:
        return [r.mol for r in records if r.mol is not None]
    return [r.mol for r in records]

def get_rxns(records: Iterable[ReactionRecord], drop_invalid: bool = True) -> List[Optional[object]]:
    """
    Extract a raw list of RDKit ChemicalReaction objects from ReactionRecords.
    """
    if drop_invalid:
        return [r.rxn for r in records if r.rxn is not None]
    return [r.rxn for r in records]