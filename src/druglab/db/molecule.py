"""
druglab.db.molecule
~~~~~~~~~~~~~~~~~~~
MoleculeTable: a BaseTable specialised for RDKit Mol objects.

If RDKit is not installed the class still imports cleanly; methods that
require RDKit raise ``ImportError`` at call time.
"""

from __future__ import annotations

import pickle
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from druglab.db.base import BaseTable, HistoryEntry

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    _RDKIT = True
except ImportError:
    _RDKIT = False


def _require_rdkit() -> None:
    if not _RDKIT:
        raise ImportError(
            "RDKit is required for MoleculeTable. "
            "Install it with: conda install -c conda-forge rdkit"
        )


class MoleculeTable(BaseTable["Chem.Mol"]):
    """
    Table of RDKit Mol objects.

    Construction
    ------------
    Prefer the factory class-methods over calling __init__ directly:

        MoleculeTable.from_smiles(["CCO", "c1ccccc1"])
        MoleculeTable.from_sdf("compounds.sdf")
        MoleculeTable.from_mols([mol1, mol2])

    Properties added beyond BaseTable
    ----------------------------------
    smiles : List[str]
        Canonical SMILES for each molecule (computed on access).
    """

    # ------------------------------------------------------------------
    # BaseTable abstract interface
    # ------------------------------------------------------------------

    def _serialize_object(self, obj: "Chem.Mol") -> bytes:
        _require_rdkit()
        mol_bytes = obj.ToBinary() if obj is not None else b""
        return mol_bytes

    def _deserialize_object(self, raw: bytes) -> "Chem.Mol":
        _require_rdkit()
        if not raw:
            return None
        return Chem.Mol(raw)

    @staticmethod
    def _deserialize_object_static(raw: bytes) -> "Chem.Mol":
        _require_rdkit()
        if not raw:
            return None
        return Chem.Mol(raw)

    def _object_type_name(self) -> str:
        return "Mol"

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_mols(
        cls,
        mols: Iterable["Chem.Mol"],
        metadata: Optional[pd.DataFrame] = None,
    ) -> "MoleculeTable":
        """Build a table from an iterable of RDKit Mol objects."""
        _require_rdkit()
        mol_list = list(mols)
        return cls(objects=mol_list, metadata=metadata)

    @classmethod
    def from_smiles(
        cls,
        smiles: Iterable[str],
        *,
        sanitize: bool = True,
        metadata: Optional[pd.DataFrame] = None,
        smiles_col: str = "smiles",
    ) -> "MoleculeTable":
        """
        Parse SMILES strings into a MoleculeTable.

        Invalid SMILES produce a ``None`` entry in ``objects`` and a
        warning is printed.  The original SMILES is always stored in
        ``metadata[smiles_col]``.
        """
        _require_rdkit()
        smiles_list = list(smiles)
        mols: List[Optional["Chem.Mol"]] = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi, sanitize=sanitize)
            if mol is None:
                import warnings
                warnings.warn(f"Could not parse SMILES: {smi!r}", stacklevel=2)
            mols.append(mol)

        if metadata is None:
            metadata = pd.DataFrame({smiles_col: smiles_list})
        elif smiles_col not in metadata.columns:
            metadata = metadata.copy()
            metadata[smiles_col] = smiles_list

        return cls(objects=mols, metadata=metadata)

    @classmethod
    def from_sdf(
        cls,
        path: str,
        *,
        sanitize: bool = True,
        remove_hs: bool = False,
        max_mols: Optional[int] = None,
    ) -> "MoleculeTable":
        """
        Load molecules from an SDF file.

        All SD properties are collected into the metadata DataFrame.
        """
        _require_rdkit()
        from rdkit.Chem import PandasTools

        df = PandasTools.LoadSDF(
            path,
            smilesName="smiles",
            molColName="_mol",
            includeFingerprints=False,
            removeHs=remove_hs,
        )
        if max_mols is not None:
            df = df.iloc[:max_mols]

        mols = list(df.pop("_mol"))
        metadata = df.reset_index(drop=True)
        return cls(objects=mols, metadata=metadata)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def smiles(self) -> List[Optional[str]]:
        """Canonical SMILES for each molecule (None for invalid mols)."""
        _require_rdkit()
        return [
            Chem.MolToSmiles(mol) if mol is not None else None
            for mol in self._objects
        ]

    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean array: True where the molecule is not None."""
        return np.array([
            not any([
                mol is None, 
                mol.GetNumAtoms() == 0
            ]) for mol in self._objects
        ])

    # ------------------------------------------------------------------
    # Molecule-specific operations
    # ------------------------------------------------------------------

    def to_smiles(self) -> List[Optional[str]]:
        """Alias for ``self.smiles``."""
        return self.smiles

    def drop_invalid(self) -> "MoleculeTable":
        """Return a new table with None/invalid molecules removed."""
        mask = self.valid_mask
        return self.subset(np.where(mask)[0])

    def add_rdkit_descriptors(
        self,
        descriptors: Optional[List[str]] = None,
        *,
        prefix: str = "",
    ) -> None:
        """
        Compute RDKit molecular descriptors and add them as metadata columns.

        Parameters
        ----------
        descriptors
            List of descriptor names from ``rdkit.Chem.Descriptors``. Options can be
            checked here: ``rdkit.Chem.Descriptors.descList``
            Defaults to a standard physicochemical set:
            MW, LogP, HBA, HBD, TPSA, RotBonds.
        prefix
            Optional prefix for the column names.
        """
        _require_rdkit()
        if descriptors is None:
            descriptors = ["MolWt", "MolLogP", "NumHAcceptors",
                           "NumHDonors", "TPSA", "NumRotatableBonds"]

        desc_fns = {name: getattr(Descriptors, name) for name in descriptors}
        for name, fn in desc_fns.items():
            col = prefix + name
            self._metadata[col] = [
                fn(mol) if mol is not None else float("nan")
                for mol in self._objects
            ]

    def filter_by_metadata(self, query: str) -> "MoleculeTable":
        """
        Return a subset table where ``metadata.query(query)`` is True.

        Example::

            light = table.filter_by_metadata("MolWt < 500")
        """
        idx = self._metadata.query(query).index.tolist()
        return self.subset(idx)

    # ------------------------------------------------------------------
    # SDF output
    # ------------------------------------------------------------------

    def to_sdf(self, path: str) -> None:
        """Write the table to an SDF file."""
        _require_rdkit()
        from rdkit.Chem import SDWriter
        writer = SDWriter(path)
        for i, mol in enumerate(self._objects):
            if mol is None:
                continue
            row = self._metadata.iloc[i]
            for col in self._metadata.columns:
                mol.SetProp(str(col), str(row[col]))
            writer.write(mol)
        writer.close()