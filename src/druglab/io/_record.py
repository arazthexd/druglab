"""
Lightweight data containers returned by all druglab.io readers.

These classes carry a parsed object (RDKit Mol / ChemicalReaction) together
with the raw metadata extracted from the source file.  They intentionally do
*not* depend on ``druglab.db`` so that ``druglab.io`` remains a standalone
layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


@dataclass
class MoleculeRecord:
    """A single parsed molecule together with its file-level metadata.

    Attributes
    ----------
    mol:
        The RDKit ``Chem.Mol`` object.  May be *None* if the record could not
        be parsed and ``on_error="skip"`` was requested.
    name:
        Molecule name / title field from the file (empty string if absent).
    properties:
        Dict of property-name → value extracted from the source file (e.g.
        SDF data fields or CSV columns).
    source:
        Path to the originating file.
    index:
        Zero-based record index within *source*.
    """

    mol: Optional[Any]  # rdkit.Chem.Mol at runtime
    name: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    index: int = 0

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def is_valid(self) -> bool:
        """Return *True* if :attr:`mol` is not *None*."""
        return self.mol is not None

    def to_smiles(self) -> Optional[str]:
        """Return the canonical SMILES string, or *None* for invalid records."""
        if self.mol is None:
            return None
        try:
            from rdkit.Chem import MolToSmiles  # type: ignore
            return MolToSmiles(self.mol)
        except Exception:
            return None

    def __repr__(self) -> str:
        smiles = self.to_smiles() or "???"
        return (
            f"MoleculeRecord(name={self.name!r}, smiles={smiles!r}, "
            f"source={self.source!r}, index={self.index})"
        )


@dataclass
class ReactionRecord:
    """A single parsed reaction together with its file-level metadata.

    Attributes
    ----------
    rxn:
        The RDKit ``AllChem.ChemicalReaction`` object.  May be *None* if the
        record could not be parsed and ``on_error="skip"`` was requested.
    name:
        Reaction name / identifier extracted from the file.
    properties:
        Dict of property-name → value extracted from the source file.
    source:
        Path to the originating file.
    index:
        Zero-based record index within *source*.
    """

    rxn: Optional[Any]  # rdkit.Chem.rdChemReactions.ChemicalReaction at runtime
    name: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    index: int = 0

    def is_valid(self) -> bool:
        """Return *True* if :attr:`rxn` is not *None*."""
        return self.rxn is not None

    def n_reactants(self) -> int:
        if self.rxn is None:
            return 0
        return self.rxn.GetNumReactantTemplates()

    def n_products(self) -> int:
        if self.rxn is None:
            return 0
        return self.rxn.GetNumProductTemplates()

    def __repr__(self) -> str:
        return (
            f"ReactionRecord(name={self.name!r}, "
            f"reactants={self.n_reactants()}, products={self.n_products()}, "
            f"source={self.source!r}, index={self.index})"
        )

AnyRecord = Union["MoleculeRecord", "ReactionRecord"]