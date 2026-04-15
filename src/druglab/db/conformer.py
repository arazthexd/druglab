"""
druglab.db.conformer
~~~~~~~~~~~~~~~~~~~~
ConformerTable: a MoleculeTable where every row holds exactly one conformer.

Workflow
--------
1.  ``MoleculeTable.unroll_conformers()``   → ConformerTable   (explode)
2.  Run any druglab.pipe pipeline on the ConformerTable (filter, featurize …)
3.  ``ConformerTable.collapse()``           → MoleculeTable    (fold back)
    or
    ``MoleculeTable.update_from_conformers(conf_table)``       (in-place merge)

Design constraints
------------------
* ConformerTable stores NO reference to its parent MoleculeTable — fully decoupled.
* Each ``Chem.Mol`` in ``_objects`` carries exactly one conformer.
* Features are initialised empty on creation to avoid OOM from copying large arrays.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from druglab.db.base import HistoryEntry
from druglab.db.molecule import MoleculeTable, _require_rdkit

try:
    from rdkit import Chem
    _RDKIT = True
except ImportError:
    _RDKIT = False


class ConformerTable(MoleculeTable):
    """
    A MoleculeTable variant where every row represents a single conformer.

    Each ``Chem.Mol`` in ``objects`` is guaranteed to hold exactly one
    conformer.  All existing MoleculeTable pipelines, featurizers and IO
    tools work on ConformerTable without modification.

    Typical usage
    -------------
    >>> conf_table = mol_table.unroll_conformers()
    >>> conf_table = pipeline.run(conf_table)          # filter / featurize
    >>> mol_table_3d = conf_table.collapse("parent_index")
    """

    # ------------------------------------------------------------------
    # Collapse: fold conformers back into a MoleculeTable
    # ------------------------------------------------------------------

    def collapse(
        self,
        groupby: str = "parent_index",
        metadata_agg: Optional[Dict[str, str]] = None,
        features_agg: Optional[Dict[str, str]] = None,
    ) -> "MoleculeTable":
        """
        Pack single-conformer rows back into multi-conformer molecules.

        Parameters
        ----------
        groupby
            Metadata column used to identify which rows belong to the same
            parent molecule.  Defaults to ``"parent_index"``.
        metadata_agg
            Aggregation rules for scalar metadata columns, passed directly to
            ``pandas.GroupBy.agg()``.  Example: ``{"Energy": "min"}``.
            Columns not listed here get their first value.
        features_agg
            Aggregation rules for feature arrays.  Supported operations:
            ``"mean"``, ``"min"``, ``"max"``, ``"sum"``, ``"first"``, ``"last"``.
            Example: ``{"shape_fp": "mean"}``.

        Returns
        -------
        MoleculeTable
            A fresh MoleculeTable with multi-conformer Mol objects.
        """
        _require_rdkit()

        meta = self._backend.get_metadata()

        if groupby not in meta.columns:
            raise ValueError(
                f"Column '{groupby}' not found in metadata. "
                f"Available columns: {list(meta.columns)}"
            )

        meta_agg = metadata_agg or {}
        feat_agg = features_agg or {}

        # Group row indices by parent key
        groups: Dict = {}
        for row_idx, key in enumerate(meta[groupby]):
            groups.setdefault(key, []).append(row_idx)

        new_objects: List[Optional[Chem.Mol]] = []
        new_meta_rows: List[dict] = []
        new_features: Dict[str, List[np.ndarray]] = {
            k: [] for k in self._backend.get_feature_names()
        }

        for key in sorted(groups.keys(), key=lambda x: (isinstance(x, float), x)):
            row_indices = groups[key]

            base_mol = self._backend._objects[row_indices[0]]
            if base_mol is None:
                new_objects.append(None)
            else:
                merged = Chem.RWMol(Chem.Mol(base_mol))
                # Already has the first conformer from base_mol
                for idx in row_indices[1:]:
                    other = self._backend._objects[idx]
                    if other is None or other.GetNumConformers() == 0:
                        continue
                    conf = other.GetConformer(0)
                    merged.AddConformer(conf, assignId=True)
                new_objects.append(merged.GetMol())

            group_df = meta.iloc[row_indices]
            row_dict: dict = {}
            for col in group_df.columns:
                if col == groupby:
                    row_dict[col] = key
                    continue
                agg_fn = meta_agg.get(col, "first")
                col_vals = group_df[col]
                if agg_fn == "first":   row_dict[col] = col_vals.iloc[0]
                elif agg_fn == "last":  row_dict[col] = col_vals.iloc[-1]
                elif agg_fn == "min":   row_dict[col] = col_vals.min()
                elif agg_fn == "max":   row_dict[col] = col_vals.max()
                elif agg_fn == "mean":  row_dict[col] = col_vals.mean()
                elif agg_fn == "sum":   row_dict[col] = col_vals.sum()
                elif callable(agg_fn):  row_dict[col] = agg_fn(col_vals)
                else:                   row_dict[col] = col_vals.iloc[0]
            new_meta_rows.append(row_dict)

            for feat_name in self._backend.get_feature_names():
                feat_array = self._backend.get_feature(feat_name)
                agg_fn = feat_agg.get(feat_name, "mean")
                group_feats = feat_array[row_indices]
                if agg_fn == "mean":    new_features[feat_name].append(group_feats.mean(axis=0))
                elif agg_fn == "min":   new_features[feat_name].append(group_feats.min(axis=0))
                elif agg_fn == "max":   new_features[feat_name].append(group_feats.max(axis=0))
                elif agg_fn == "sum":   new_features[feat_name].append(group_feats.sum(axis=0))
                elif agg_fn == "first": new_features[feat_name].append(group_feats[0])
                elif agg_fn == "last":  new_features[feat_name].append(group_feats[-1])
                else:                   new_features[feat_name].append(group_feats.mean(axis=0))

        new_meta = pd.DataFrame(new_meta_rows).reset_index(drop=True)
        stacked_features = {
            k: np.stack(v, axis=0) for k, v in new_features.items() if v
        }

        history = list(self._history) + [
            HistoryEntry.now(
                block_name="ConformerTable.collapse",
                config={
                    "groupby": groupby,
                    "metadata_agg": str(meta_agg),
                    "features_agg": str(feat_agg),
                },
                rows_in=self.n,
                rows_out=len(new_objects),
            )
        ]

        return MoleculeTable(
            objects=new_objects,
            metadata=new_meta,
            features=stacked_features,
            history=history,
        )

    # ------------------------------------------------------------------
    # Repr override so the class name is clear
    # ------------------------------------------------------------------

    def _object_type_name(self) -> str:
        return "ConformerMol"