"""
druglab.db.molecule
~~~~~~~~~~~~~~~~~~~
MoleculeTable: a BaseTable specialised for RDKit Mol objects.

If RDKit is not installed the class still imports cleanly; methods that
require RDKit raise ``ImportError`` at call time.
"""

from __future__ import annotations

import copy
import pickle
from typing import Dict, Iterable, List, Optional, Sequence, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from druglab.db.base import BaseTable, HistoryEntry

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    _RDKIT = True
except ImportError:
    _RDKIT = False

if TYPE_CHECKING:
    from druglab.io._record import MoleculeRecord
    from druglab.db.conformer import ConformerTable


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
    def from_records(
        cls,
        records: Iterable["MoleculeRecord"]  # type: ignore
    ) -> "MoleculeTable":
        """
        Bridge method: Convert druglab.io MoleculeRecords into a MoleculeTable.
        """
        mols = []
        meta_rows = []
        for r in records:
            mols.append(r.mol)
            row = {"name": r.name, "source": r.source, "index": r.index}
            row.update(r.properties)
            meta_rows.append(row)
        
        metadata = pd.DataFrame(meta_rows)
        return cls(objects=mols, metadata=metadata)
    
    @classmethod
    def from_file(
        cls,
        path: str,
        **reader_kwargs
    ) -> "MoleculeTable":
        """
        Load a table from any supported chemical file format (SDF, SMILES, CSV).
        Uses druglab.io under the hood.
        """
        _require_rdkit()
        from druglab.io import read_file
        records = read_file(path, **reader_kwargs)
        return cls.from_records(records)

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
        Legacy/Explicit SDF loader. 
        It is recommended to use MoleculeTable.from_file() instead.
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
    # Export & Output Bridging
    # ------------------------------------------------------------------

    def to_records(self) -> List["MoleculeRecord"]: # type: ignore
        """
        Bridge method: Convert this table back into druglab.io MoleculeRecords.
        """
        from druglab.io._record import MoleculeRecord
        records = []
        for i, mol in enumerate(self._objects):
            row_dict = self._metadata.iloc[i].to_dict()
            name = row_dict.pop("name", "")
            source = row_dict.pop("source", "")
            index = row_dict.pop("index", i)
            
            records.append(MoleculeRecord(
                mol=mol,
                name=str(name) if not pd.isna(name) else "",
                properties={k: v for k, v in row_dict.items() if not pd.isna(v)},
                source=str(source),
                index=int(index)
            ))
        return records

    def to_file(self, path: str, **writer_kwargs) -> None:
        """
        Write the table to any supported chemical file format.
        Uses druglab.io under the hood.
        """
        from druglab.io import write_file
        write_file(self.to_records(), path, **writer_kwargs)

    def to_sdf(self, path: str, overwrite: bool = False) -> None:
        """Alias for to_file for writing to SDF specifically."""
        self.to_file(path, overwrite=overwrite)

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
    def mols(self) -> List["Chem.Mol"]:
        """
        Flat list of all valid RDKit Mol objects in the table. 
        Useful for passing directly to external libraries.
        """
        return [mol for mol in self._objects if mol is not None]

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
    # 3D Conformer handling
    # ------------------------------------------------------------------

    def unroll_conformers(
        self,
        id_col: str = "parent_index",
    ) -> "ConformerTable":
        """
        Explode a multi-conformer MoleculeTable into a ConformerTable where
        every row holds exactly one conformer.

        Molecules with no conformers or ``None`` entries are silently skipped.

        Parameters
        ----------
        id_col
            Name of the metadata column that will store the integer index of
            the parent molecule.  Defaults to ``"parent_index"``.

        Returns
        -------
        ConformerTable
            A new table with one row per conformer.  Features are initialised
            to an empty dict to avoid Out-of-Memory errors on large datasets.

        Notes
        -----
        * Each new ``Chem.Mol`` is a fresh copy of the parent graph with
          exactly one conformer injected, so serialisation via
          ``mol.ToBinary()`` stays lightweight.
        * The original ``MoleculeTable`` is **not** modified.
        """
        _require_rdkit()
        from druglab.db.conformer import ConformerTable

        new_objects: List[Chem.Mol] = []
        meta_rows: List[dict] = []

        for parent_idx, mol in enumerate(self._objects):
            if mol is None or mol.GetNumConformers() == 0:
                continue

            parent_row = self._metadata.iloc[parent_idx].to_dict()

            for conf in mol.GetConformers():
                conf_id = conf.GetId()

                # Build a lightweight single-conformer copy
                single = Chem.RWMol(Chem.Mol(mol))
                single.RemoveAllConformers()
                # Copy must use a new Conformer object (not the same reference)
                new_conf = Chem.Conformer(conf)
                single.AddConformer(new_conf, assignId=True)

                new_objects.append(single.GetMol())

                row = dict(parent_row)
                row[id_col] = parent_idx
                row["conf_id"] = conf_id
                meta_rows.append(row)

        new_meta = pd.DataFrame(meta_rows).reset_index(drop=True)

        history = list(self._history) + [
            HistoryEntry.now(
                block_name="MoleculeTable.unroll_conformers",
                config={"id_col": id_col},
                rows_in=self.n,
                rows_out=len(new_objects),
            )
        ]

        return ConformerTable(
            objects=new_objects,
            metadata=new_meta,
            features={},   # intentionally empty — no OOM risk
            history=history,
        )

    def update_from_conformers(
        self,
        conf_table: "ConformerTable",
        id_col: str = "parent_index",
        metadata_agg: Optional[Dict] = None,
        features_agg: Optional[Dict] = None,
        drop_empty: bool = True,
    ) -> "MoleculeTable":
        """
        Merge a processed ConformerTable back into this MoleculeTable.

        This allows a filtered / featurised ConformerTable to overwrite the
        conformers and selectively update scalar properties of the parent table.

        Parameters
        ----------
        conf_table
            The ConformerTable produced by (and derived from) this table.
        id_col
            The metadata column in ``conf_table`` that maps rows back to
            parent indices.  Must match the ``id_col`` used in
            ``unroll_conformers()``.
        metadata_agg
            Aggregation rules for collapsing ConformerTable metadata into
            parent-level scalars, e.g. ``{"Energy": "min"}``.
            Only columns listed here are written back to the parent metadata.
        features_agg
            Aggregation rules for feature arrays, e.g. ``{"shape_fp": "mean"}``.
            Supported: ``"mean"``, ``"min"``, ``"max"``, ``"sum"``,
            ``"first"``, ``"last"``.
            Only keys listed here are written back to the parent features.
        drop_empty
            If ``True`` (default), parent molecules whose conformers were all
            filtered out of ``conf_table`` are removed from the result.
            If ``False``, those molecules are kept but their conformer list is
            cleared.

        Returns
        -------
        MoleculeTable
            A new MoleculeTable respecting functional immutability.
        """
        _require_rdkit()

        if id_col not in conf_table.metadata.columns:
            raise ValueError(
                f"Column '{id_col}' not found in ConformerTable metadata. "
                f"Available: {list(conf_table.metadata.columns)}"
            )

        meta_agg = metadata_agg or {}
        feat_agg = features_agg or {}

        # Build a mapping: parent_index -> list of conf_table row indices
        parent_to_rows: Dict[int, List[int]] = {}
        for row_idx, parent_idx in enumerate(conf_table.metadata[id_col]):
            parent_to_rows.setdefault(int(parent_idx), []).append(row_idx)

        # Deep-copy objects so the original table is untouched
        new_objects = copy.deepcopy(self._objects)
        new_meta = self._metadata.copy(deep=True)
        new_features = {k: v.copy() for k, v in self._features.items()}

        keep_mask = np.ones(self.n, dtype=bool)

        for parent_idx in range(self.n):
            mol = new_objects[parent_idx]
            if mol is None:
                keep_mask[parent_idx] = False
                continue

            row_indices = parent_to_rows.get(parent_idx)

            if not row_indices:
                # All conformers were filtered out for this molecule
                if drop_empty:
                    keep_mask[parent_idx] = False
                else:
                    mol.RemoveAllConformers()
                continue

            # Inject surviving conformers back into the parent mol
            mol.RemoveAllConformers()
            for ri in row_indices:
                conf_mol = conf_table.objects[ri]
                if conf_mol is None or conf_mol.GetNumConformers() == 0:
                    continue
                conf = conf_mol.GetConformer(0)
                mol.AddConformer(Chem.Conformer(conf), assignId=True)

            # Aggregate metadata columns back
            group_df = conf_table.metadata.iloc[row_indices]
            for col, agg_fn in meta_agg.items():
                if col not in group_df.columns:
                    continue
                col_vals = group_df[col]
                if agg_fn == "first":
                    val = col_vals.iloc[0]
                elif agg_fn == "last":
                    val = col_vals.iloc[-1]
                elif agg_fn == "min":
                    val = col_vals.min()
                elif agg_fn == "max":
                    val = col_vals.max()
                elif agg_fn == "mean":
                    val = col_vals.mean()
                elif agg_fn == "sum":
                    val = col_vals.sum()
                elif callable(agg_fn):
                    val = agg_fn(col_vals)
                else:
                    val = col_vals.iloc[0]
                new_meta.at[parent_idx, col] = val

            # Aggregate feature arrays back
            for feat_name, agg_fn in feat_agg.items():
                if feat_name not in conf_table.features:
                    continue
                feat_arr = conf_table.features[feat_name]
                group_feats = feat_arr[row_indices]
                if agg_fn == "mean":
                    agg_val = group_feats.mean(axis=0)
                elif agg_fn == "min":
                    agg_val = group_feats.min(axis=0)
                elif agg_fn == "max":
                    agg_val = group_feats.max(axis=0)
                elif agg_fn == "sum":
                    agg_val = group_feats.sum(axis=0)
                elif agg_fn == "first":
                    agg_val = group_feats[0]
                elif agg_fn == "last":
                    agg_val = group_feats[-1]
                else:
                    agg_val = group_feats.mean(axis=0)

                if feat_name not in new_features:
                    # Initialise feature array with zeros (correct shape from conf_table)
                    new_features[feat_name] = np.zeros(
                        (self.n,) + agg_val.shape, dtype=agg_val.dtype
                    )
                new_features[feat_name][parent_idx] = agg_val

        # Apply the keep mask
        keep_indices = np.where(keep_mask)[0]
        filtered_objects = [new_objects[i] for i in keep_indices]
        filtered_meta = new_meta.iloc[keep_indices].reset_index(drop=True)
        filtered_features = {k: v[keep_indices] for k, v in new_features.items()}

        history = list(self._history) + [
            HistoryEntry.now(
                block_name="MoleculeTable.update_from_conformers",
                config={
                    "id_col": id_col,
                    "drop_empty": drop_empty,
                    "metadata_agg": str(meta_agg),
                    "features_agg": str(feat_agg),
                },
                rows_in=self.n,
                rows_out=len(filtered_objects),
            )
        ]

        return MoleculeTable(
            objects=filtered_objects,
            metadata=filtered_meta,
            features=filtered_features,
            history=history,
        )

    # ------------------------------------------------------------------
    # SDF output (kept for backward compat — to_file is the preferred API)
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