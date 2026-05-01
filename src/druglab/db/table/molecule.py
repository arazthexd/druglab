"""
druglab.db.molecule
~~~~~~~~~~~~~~~~~~~
MoleculeTable: a BaseTable specialised for RDKit Mol objects.

Object serialization uses SMILES strings (``molecules.smi``) rather than
pickle, making bundles robust against RDKit C++ ABI changes and Python
version upgrades.  3-D conformers are serialised as SDF blocks
(``molecules.sdf``) when any molecule in the table carries conformers.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from ..backend import EagerMemoryBackend
from .base import BaseTable, HistoryEntry

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    _RDKIT = True
except ImportError:
    _RDKIT = False

if TYPE_CHECKING:
    from ...io._record import MoleculeRecord
    from .conformer import ConformerTable


def _require_rdkit() -> None:
    if not _RDKIT:
        raise ImportError(
            "RDKit is required for MoleculeTable. "
            "Install it with: conda install -c conda-forge rdkit"
        )


# ---------------------------------------------------------------------------
# Codec helpers (module-level so they are picklable if ever needed by
# multiprocessing, though they themselves do not use pickle).
# ---------------------------------------------------------------------------

def _write_molecules(objects: List, dir_path: Path) -> None:
    """
    Persist a list of RDKit Mol objects to disk.

    Strategy
    --------
    * Molecules **with 3-D conformers** → ``molecules.sdf`` (SDF blocks).
    * Molecules **without conformers** → ``molecules.smi`` (SMILES, one per line).

    Both strategies can coexist: the reader detects which file is present.
    An empty line or the literal ``None`` token represents a ``None`` molecule.
    """
    _require_rdkit()

    has_conformers = any(
        m is not None and m.GetNumConformers() > 0 for m in objects
    )

    if has_conformers:
        # SDF path — preserves 3-D coordinates.
        sdf_path = dir_path / "molecules.sdf"
        writer = Chem.SDWriter(str(sdf_path))
        for mol in objects:
            if mol is None:
                # SDWriter cannot write None; write a sentinel blank mol.
                writer.write(Chem.MolFromSmiles("*"))
            else:
                writer.write(mol)
        writer.close()
        # Also write a flag so the reader knows None positions.
        none_positions = [i for i, m in enumerate(objects) if m is None]
        (dir_path / "none_positions.txt").write_text(
            "\n".join(map(str, none_positions)), encoding="utf-8"
        )
    else:
        # SMILES path — compact, human-readable, ABI-stable.
        smi_path = dir_path / "molecules.smi"
        lines = []
        for mol in objects:
            if mol is None:
                lines.append("__NONE__")
            else:
                smi = Chem.MolToSmiles(mol)
                lines.append(smi if smi else "__NONE__")
        smi_path.write_text("\n".join(lines), encoding="utf-8")


def _read_molecules(dir_path: Path) -> List:
    """Restore molecules from ``molecules.sdf`` or ``molecules.smi``."""
    _require_rdkit()

    sdf_path = dir_path / "molecules.sdf"
    smi_path = dir_path / "molecules.smi"

    if sdf_path.exists():
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
        none_positions: set = set()
        none_path = dir_path / "none_positions.txt"
        if none_path.exists():
            text = none_path.read_text(encoding="utf-8").strip()
            if text:
                none_positions = {int(x) for x in text.splitlines()}
        mols = []
        for i, mol in enumerate(supplier):
            mols.append(None if i in none_positions else mol)
        return mols

    if smi_path.exists():
        lines = smi_path.read_text(encoding="utf-8").splitlines()
        mols = []
        for line in lines:
            line = line.strip()
            if not line or line == "__NONE__":
                mols.append(None)
            else:
                mols.append(Chem.MolFromSmiles(line))
        return mols

    return []


class MoleculeTable(BaseTable["Chem.Mol"]):
    """
    Table of RDKit Mol objects.

    Object codec
    ------------
    * ``_get_default_object_writer`` → writes ``molecules.smi`` (or ``.sdf``
      when conformers are present), completely replacing the old pickle path.
    * ``_get_default_object_reader`` → reads the matching file back.
    """

    # ------------------------------------------------------------------
    # Domain-safe I/O codec  (Task 2)
    # ------------------------------------------------------------------

    def _get_default_object_writer(self) -> Callable[[List, Path], None]:
        return _write_molecules

    @classmethod
    def _get_default_object_reader(cls) -> Callable[[Path], List]:
        return _read_molecules

    # Kept for back-compat / direct use by subclasses.
    def _serialize_object(self, obj: "Chem.Mol") -> bytes:
        _require_rdkit()
        return obj.ToBinary() if obj is not None else b""

    @classmethod
    def _deserialize_object(cls, raw: bytes) -> "Chem.Mol":
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
    def from_mols(cls, mols: Iterable["Chem.Mol"], metadata: Optional[pd.DataFrame] = None) -> "MoleculeTable":
        _require_rdkit()
        return cls(objects=list(mols), metadata=metadata)

    @classmethod
    def from_records(cls, records) -> "MoleculeTable":
        mols, meta_rows = [], []
        for r in records:
            mols.append(r.mol)
            row = {"name": r.name, "source": r.source, "index": r.index}
            row.update(r.properties)
            meta_rows.append(row)
        return cls(objects=mols, metadata=pd.DataFrame(meta_rows))

    @classmethod
    def from_file(cls, path: str, **reader_kwargs) -> "MoleculeTable":
        _require_rdkit()
        from druglab.io import read_file
        return cls.from_records(read_file(path, **reader_kwargs))

    @classmethod
    def from_smiles(
        cls,
        smiles: Iterable[str],
        *,
        sanitize: bool = True,
        metadata: Optional[pd.DataFrame] = None,
        smiles_col: str = "smiles",
    ) -> "MoleculeTable":
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
    def from_sdf(cls, path: str, *, sanitize: bool = True, remove_hs: bool = False, max_mols: Optional[int] = None) -> "MoleculeTable":
        _require_rdkit()
        from rdkit.Chem import PandasTools
        df = PandasTools.LoadSDF(path, smilesName="smiles", molColName="_mol", includeFingerprints=False, removeHs=remove_hs)
        if max_mols is not None:
            df = df.iloc[:max_mols]
        mols = list(df.pop("_mol"))
        return cls(objects=mols, metadata=df.reset_index(drop=True))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_records(self):
        from druglab.io._record import MoleculeRecord
        records = []
        meta = self.get_metadata()
        for i, mol in enumerate(self.get_objects()):
            row_dict = meta.iloc[i].to_dict()
            name  = row_dict.pop("name", "")
            source = row_dict.pop("source", "")
            index = row_dict.pop("index", i)
            records.append(MoleculeRecord(
                mol=mol, name=str(name) if not pd.isna(name) else "",
                properties={k: v for k, v in row_dict.items() if not pd.isna(v)},
                source=str(source), index=int(index),
            ))
        return records

    def to_file(self, path: str, **writer_kwargs) -> None:
        from druglab.io import write_file
        write_file(self.to_records(), path, **writer_kwargs)

    def to_sdf(self, path: str, overwrite: bool = False) -> None:
        _require_rdkit()
        from rdkit.Chem import SDWriter
        p = Path(path)
        if p.exists() and not overwrite:
            raise FileExistsError(f"File '{path}' already exists.")
        writer = SDWriter(path)
        meta = self.get_metadata()
        for i, mol in enumerate(self.get_objects()):
            if mol is None:
                continue
            mol = Chem.Mol(mol)
            row = meta.iloc[i]
            for col in meta.columns:
                mol.SetProp(str(col), str(row[col]))
            writer.write(mol)
        writer.close()

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def smiles(self) -> List[Optional[str]]:
        _require_rdkit()
        return [Chem.MolToSmiles(mol) if mol is not None else None for mol in self.get_objects()]

    @property
    def mols(self) -> List["Chem.Mol"]:
        return [m for m in self.get_objects() if m is not None]

    @property
    def valid_mask(self) -> np.ndarray:
        return np.array([
            not any([mol is None, mol.GetNumAtoms() == 0])
            for mol in self.get_objects()
        ])

    def to_smiles(self) -> List[Optional[str]]:
        return self.smiles

    def drop_invalid(self) -> "MoleculeTable":
        return self.subset(np.where(self.valid_mask)[0])

    def add_rdkit_descriptors(self, descriptors: Optional[List[str]] = None, *, prefix: str = "") -> None:
        _require_rdkit()
        if descriptors is None:
            descriptors = ["MolWt", "MolLogP", "NumHAcceptors", "NumHDonors", "TPSA", "NumRotatableBonds"]
        desc_fns = {name: getattr(Descriptors, name) for name in descriptors}
        new_cols = {
            prefix + name: [fn(mol) if mol is not None else float("nan") for mol in self.get_objects()]
            for name, fn in desc_fns.items()
        }
        self.add_metadata_columns(new_cols)

    # ------------------------------------------------------------------
    # 3-D Conformer handling
    # ------------------------------------------------------------------

    def unroll_conformers(self, id_col: str = "parent_index") -> "ConformerTable":
        _require_rdkit()
        from .conformer import ConformerTable

        new_objects, meta_rows = [], []
        meta = self.get_metadata()

        for parent_idx, mol in enumerate(self.get_objects()):
            if mol is None or mol.GetNumConformers() == 0:
                continue
            parent_row = meta.iloc[parent_idx].to_dict()
            for conf in mol.GetConformers():
                single = Chem.RWMol(Chem.Mol(mol))
                single.RemoveAllConformers()
                single.AddConformer(Chem.Conformer(conf), assignId=True)
                new_objects.append(single.GetMol())
                row = dict(parent_row)
                row[id_col] = parent_idx
                row["conf_id"] = conf.GetId()
                meta_rows.append(row)

        history = list(self._history) + [
            HistoryEntry.now(
                block_name="MoleculeTable.unroll_conformers",
                config={"id_col": id_col},
                rows_in=self.n, rows_out=len(new_objects),
            )
        ]
        return ConformerTable(
            objects=new_objects,
            metadata=pd.DataFrame(meta_rows).reset_index(drop=True),
            features={},
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
        _require_rdkit()
        conf_meta = conf_table.get_metadata()
        if id_col not in conf_meta.columns:
            raise ValueError(f"Column '{id_col}' not found in ConformerTable metadata.")

        meta_agg = metadata_agg or {}
        feat_agg = features_agg or {}

        parent_to_rows: Dict[int, List[int]] = {}
        for row_idx, parent_idx in enumerate(conf_meta[id_col]):
            parent_to_rows.setdefault(int(parent_idx), []).append(row_idx)

        new_objects = copy.deepcopy(self.get_objects())
        new_meta = self.get_metadata().copy(deep=True)
        new_features = {k: self.get_feature(k).copy() for k in self.feature_names}
        keep_mask = np.ones(self.n, dtype=bool)

        for parent_idx in range(self.n):
            mol = new_objects[parent_idx]
            if mol is None:
                keep_mask[parent_idx] = False
                continue
            row_indices = parent_to_rows.get(parent_idx)
            if not row_indices:
                if drop_empty:
                    keep_mask[parent_idx] = False
                else:
                    mol.RemoveAllConformers()
                continue
            mol.RemoveAllConformers()
            for ri in row_indices:
                conf_mol = conf_table.get_objects(ri)
                if conf_mol is None or conf_mol.GetNumConformers() == 0:
                    continue
                mol.AddConformer(Chem.Conformer(conf_mol.GetConformer(0)), assignId=True)

            group_df = conf_meta.iloc[row_indices]
            for col, agg_fn in meta_agg.items():
                if col not in group_df.columns:
                    continue
                col_vals = group_df[col]
                if agg_fn == "first":    val = col_vals.iloc[0]
                elif agg_fn == "last":   val = col_vals.iloc[-1]
                elif agg_fn == "min":    val = col_vals.min()
                elif agg_fn == "max":    val = col_vals.max()
                elif agg_fn == "mean":   val = col_vals.mean()
                elif agg_fn == "sum":    val = col_vals.sum()
                elif callable(agg_fn):   val = agg_fn(col_vals)
                else:                    val = col_vals.iloc[0]
                new_meta.at[parent_idx, col] = val

            for feat_name, agg_fn in feat_agg.items():
                if feat_name not in conf_table.feature_names:
                    continue
                group_feats = conf_table.get_feature(feat_name)[row_indices]
                if agg_fn == "mean":    agg_val = group_feats.mean(axis=0)
                elif agg_fn == "min":   agg_val = group_feats.min(axis=0)
                elif agg_fn == "max":   agg_val = group_feats.max(axis=0)
                elif agg_fn == "sum":   agg_val = group_feats.sum(axis=0)
                elif agg_fn == "first": agg_val = group_feats[0]
                elif agg_fn == "last":  agg_val = group_feats[-1]
                else:                   agg_val = group_feats.mean(axis=0)
                if feat_name not in new_features:
                    new_features[feat_name] = np.zeros((self.n,) + agg_val.shape, dtype=agg_val.dtype)
                new_features[feat_name][parent_idx] = agg_val

        keep_indices = np.where(keep_mask)[0]
        history = list(self._history) + [
            HistoryEntry.now(
                block_name="MoleculeTable.update_from_conformers",
                config={"id_col": id_col, "drop_empty": drop_empty,
                        "metadata_agg": str(meta_agg), "features_agg": str(feat_agg)},
                rows_in=self.n, rows_out=int(keep_indices.sum()),
            )
        ]
        return MoleculeTable(
            objects=[new_objects[i] for i in keep_indices],
            metadata=new_meta.iloc[keep_indices].reset_index(drop=True),
            features={k: v[keep_indices] for k, v in new_features.items()},
            history=history,
        )