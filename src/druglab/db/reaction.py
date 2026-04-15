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
    """Table of RDKit ChemicalReaction objects."""

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

    @classmethod
    def from_reactions(
        cls,
        reactions: Iterable["rdChemReactions.ChemicalReaction"],
        metadata: Optional[pd.DataFrame] = None,
    ) -> "ReactionTable":
        _require_rdkit()
        rxn_list = list(reactions)
        return cls(objects=rxn_list, metadata=metadata)

    @classmethod
    def from_records(
        cls,
        records: Iterable["ReactionRecord"]
    ) -> "ReactionTable":
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
        _require_rdkit()
        smarts_list = list(smarts)
        rxns = []
        for sma in smarts_list:
            rxn = AllChem.ReactionFromSmarts(sma)
            if rxn is None:
                import warnings
                warnings.warn(f"Could not parse reaction SMARTS: {sma!r}", stacklevel=2)
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

    def to_records(self) -> List["ReactionRecord"]:  # type: ignore
        from druglab.io._record import ReactionRecord
        records = []
        meta = self._backend.get_metadata()
        for i, rxn in enumerate(self._backend._objects):
            row_dict = meta.iloc[i].to_dict()
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
        from druglab.io import write_file
        write_file(self.to_records(), path, **writer_kwargs)

    @property
    def smarts(self) -> List[Optional[str]]:
        _require_rdkit()
        return [
            AllChem.ReactionToSmarts(rxn) if rxn is not None else None
            for rxn in self._backend._objects
        ]

    @property
    def rxns(self) -> List["rdChemReactions.ChemicalReaction"]:
        return [rxn for rxn in self._backend._objects if rxn is not None]

    @property
    def n_reactants(self) -> List[int]:
        return [
            rxn.GetNumReactantTemplates() if rxn is not None else 0
            for rxn in self._backend._objects
        ]

    @property
    def n_products(self) -> List[int]:
        return [
            rxn.GetNumProductTemplates() if rxn is not None else 0
            for rxn in self._backend._objects
        ]

    @property
    def valid_mask(self) -> np.ndarray:
        return np.array([rxn is not None for rxn in self._backend._objects])

    def reactant_tables(self) -> List["MoleculeTable"]:  # type: ignore
        _require_rdkit()
        from druglab.db.molecule import MoleculeTable
        tables = []
        for rxn in self._backend._objects:
            if rxn is None:
                tables.append(MoleculeTable())
            else:
                mols = list(rxn.GetReactants())
                tables.append(MoleculeTable.from_mols(mols))
        return tables

    def product_tables(self) -> List["MoleculeTable"]:  # type: ignore
        _require_rdkit()
        from druglab.db.molecule import MoleculeTable
        tables = []
        for rxn in self._backend._objects:
            if rxn is None:
                tables.append(MoleculeTable())
            else:
                mols = list(rxn.GetProducts())
                tables.append(MoleculeTable.from_mols(mols))
        return tables

    def drop_invalid(self) -> "ReactionTable":
        mask = self.valid_mask
        return self.subset(np.where(mask)[0])

    def validate_reactions(self) -> List[bool]:
        _require_rdkit()
        results = []
        for rxn in self._backend._objects:
            if rxn is None:
                results.append(False)
            else:
                errors = rdChemReactions.PreprocessReaction(rxn)
                results.append(len(errors) == 0)
        return results