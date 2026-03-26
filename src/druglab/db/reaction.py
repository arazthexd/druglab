"""
druglab.db.reaction
~~~~~~~~~~~~~~~~~~~
ReactionTable: a BaseTable specialised for RDKit ChemicalReaction objects.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from druglab.db.base import BaseTable, HistoryEntry

try:
    from rdkit.Chem import AllChem, rdChemReactions
    _RDKIT = True
except ImportError:
    _RDKIT = False

if TYPE_CHECKING:
    from druglab.io._record import ReactionRecord


def _require_rdkit() -> None:
    if not _RDKIT:
        raise ImportError(
            "RDKit is required for ReactionTable. "
            "Install it with: conda install -c conda-forge rdkit"
        )


class ReactionTable(BaseTable["rdChemReactions.ChemicalReaction"]):
    """
    Table of RDKit ChemicalReaction objects.

    Construction
    ------------
        ReactionTable.from_smarts(["[C:1]>>[C:1]O", ...])
        ReactionTable.from_rxn_files(["rxn1.rxn", "rxn2.rxn"])
        ReactionTable.from_reactions([rxn1, rxn2])

    Additional views
    ----------------
    ``reactant_tables`` and ``product_tables`` return MoleculeTable views
    of reactant / product molecules extracted from the reactions.
    """

    # ------------------------------------------------------------------
    # BaseTable abstract interface
    # ------------------------------------------------------------------

    def _serialize_object(self, obj: "rdChemReactions.ChemicalReaction") -> bytes:
        _require_rdkit()
        return obj.ToBinary() if obj is not None else b""

    def _deserialize_object(self, raw: bytes) -> "rdChemReactions.ChemicalReaction":
        _require_rdkit()
        if not raw:
            return None
        return rdChemReactions.ChemicalReaction(raw)

    @staticmethod
    def _deserialize_object_static(raw: bytes) -> "rdChemReactions.ChemicalReaction":
        _require_rdkit()
        if not raw:
            return None
        return rdChemReactions.ChemicalReaction(raw)

    def _object_type_name(self) -> str:
        return "ChemicalReaction"

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_reactions(
        cls,
        reactions: Iterable["rdChemReactions.ChemicalReaction"],
        metadata: Optional[pd.DataFrame] = None,
    ) -> "ReactionTable":
        """Build a table from an iterable of RDKit ChemicalReaction objects."""
        _require_rdkit()
        rxn_list = list(reactions)
        return cls(objects=rxn_list, metadata=metadata)
    
    @classmethod
    def from_records(
        cls,
        records: Iterable["ReactionRecord"] 
    ) -> "ReactionTable":
        """
        Bridge method: Convert druglab.io ReactionRecords into a ReactionTable.
        """
        rxns = []
        meta_rows = []
        for r in records:
            rxns.append(r.rxn)
            row = {"name": r.name, "source": r.source, "index": r.index}
            row.update(r.properties)
            meta_rows.append(row)
        
        metadata = pd.DataFrame(meta_rows)
        return cls(objects=rxns, metadata=metadata)

    @classmethod
    def from_file(
        cls,
        path: str,
        **reader_kwargs
    ) -> "ReactionTable":
        """
        Load a table from any supported reaction file format.
        Uses druglab.io under the hood.
        """
        _require_rdkit()
        from druglab.io import read_file
        records = read_file(path, **reader_kwargs)
        return cls.from_records(records)

    @classmethod
    def from_smarts(
        cls,
        smarts: Iterable[str],
        *,
        metadata: Optional[pd.DataFrame] = None,
        smarts_col: str = "reaction_smarts",
    ) -> "ReactionTable":
        """
        Parse reaction SMARTS strings.

        Invalid SMARTS produce a ``None`` entry with a warning.
        """
        _require_rdkit()
        smarts_list = list(smarts)
        rxns: List[Optional["rdChemReactions.ChemicalReaction"]] = []
        for sma in smarts_list:
            rxn = AllChem.ReactionFromSmarts(sma)
            if rxn is None:
                import warnings
                warnings.warn(
                    f"Could not parse reaction SMARTS: {sma!r}", stacklevel=2
                )
            rxns.append(rxn)

        if metadata is None:
            metadata = pd.DataFrame({smarts_col: smarts_list})
        elif smarts_col not in metadata.columns:
            metadata = metadata.copy()
            metadata[smarts_col] = smarts_list

        return cls(objects=rxns, metadata=metadata)

    @classmethod
    def from_rxn_files(
        cls,
        paths: Iterable[str],
        *,
        metadata: Optional[pd.DataFrame] = None,
    ) -> "ReactionTable":
        """Load reactions from a list of .rxn file paths."""
        _require_rdkit()
        paths = list(paths)
        rxns = []
        for p in paths:
            rxn = AllChem.ReactionFromRxnFile(p)
            if rxn is None:
                import warnings
                warnings.warn(f"Could not load reaction from: {p!r}", stacklevel=2)
            rxns.append(rxn)

        if metadata is None:
            metadata = pd.DataFrame({"rxn_file": paths})
        return cls(objects=rxns, metadata=metadata)
    
    # ------------------------------------------------------------------
    # Export & Output Bridging
    # ------------------------------------------------------------------
    
    def to_records(self) -> List["ReactionRecord"]: # type: ignore
        """
        Bridge method: Convert this table back into druglab.io ReactionRecords.
        """
        from druglab.io._record import ReactionRecord
        records = []
        for i, rxn in enumerate(self._objects):
            row_dict = self._metadata.iloc[i].to_dict()
            name = row_dict.pop("name", "")
            source = row_dict.pop("source", "")
            index = row_dict.pop("index", i)
            
            records.append(ReactionRecord(
                rxn=rxn,
                name=str(name) if not pd.isna(name) else "",
                properties={k: v for k, v in row_dict.items() if not pd.isna(v)},
                source=str(source),
                index=int(index)
            ))
        return records

    def to_file(self, path: str, **writer_kwargs) -> None:
        """Write the table to a supported reaction format via druglab.io."""
        from druglab.io import write_file
        write_file(self.to_records(), path, **writer_kwargs)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def smarts(self) -> List[Optional[str]]:
        """Reaction SMARTS for each reaction (None for invalid entries)."""
        _require_rdkit()
        return [
            AllChem.ReactionToSmarts(rxn) if rxn is not None else None
            for rxn in self._objects
        ]
    
    @property
    def rxns(self) -> List["rdChemReactions.ChemicalReaction"]:
        """
        Flat list of all valid RDKit ChemicalReaction objects in the table.
        """
        return [rxn for rxn in self._objects if rxn is not None]

    @property
    def n_reactants(self) -> List[int]:
        """Number of reactant templates per reaction."""
        return [
            rxn.GetNumReactantTemplates() if rxn is not None else 0
            for rxn in self._objects
        ]

    @property
    def n_products(self) -> List[int]:
        """Number of product templates per reaction."""
        return [
            rxn.GetNumProductTemplates() if rxn is not None else 0
            for rxn in self._objects
        ]

    @property
    def valid_mask(self) -> np.ndarray:
        return np.array([rxn is not None for rxn in self._objects])

    # ------------------------------------------------------------------
    # Molecule views
    # ------------------------------------------------------------------

    def reactant_tables(self) -> List["MoleculeTable"]: # type: ignore
        """
        Return one MoleculeTable per reaction containing its reactant templates.
        """
        _require_rdkit()
        from druglab.db.molecule import MoleculeTable
        tables = []
        for rxn in self._objects:
            if rxn is None:
                tables.append(MoleculeTable())
            else:
                mols = list(rxn.GetReactants())
                tables.append(MoleculeTable.from_mols(mols))
        return tables

    def product_tables(self) -> List["MoleculeTable"]: # type: ignore
        """
        Return one MoleculeTable per reaction containing its product templates.
        """
        _require_rdkit()
        from druglab.db.molecule import MoleculeTable
        tables = []
        for rxn in self._objects:
            if rxn is None:
                tables.append(MoleculeTable())
            else:
                mols = list(rxn.GetProducts())
                tables.append(MoleculeTable.from_mols(mols))
        return tables

    # ------------------------------------------------------------------
    # Validity helpers
    # ------------------------------------------------------------------

    def drop_invalid(self) -> "ReactionTable":
        """Return a new table with None/invalid reactions removed."""
        mask = self.valid_mask
        return self.subset(np.where(mask)[0])

    def validate_reactions(self) -> List[bool]:
        """
        Run RDKit's reaction validation on each reaction.
        Returns a list of booleans (True = passed).
        """
        _require_rdkit()
        results = []
        for rxn in self._objects:
            if rxn is None:
                results.append(False)
            else:
                errors = rdChemReactions.PreprocessReaction(rxn)
                results.append(len(errors) == 0)
        return results