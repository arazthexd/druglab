from typing import List, Any
import h5py

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdChemReactions as rdRxn

from .base import BaseFeaturizer

class RxnOneHotFeaturizer(BaseFeaturizer):
    def __init__(self, smarts: List[str] = None, path: str = None):
        super().__init__()
        if smarts is None:
            if path is None:
                smarts = []
            else:
                smarts = self._read_file(path)
        self._smarts = smarts
        self._fnames = [f"rxn_{i}" for i in range(len(self._smarts))]

    def _read_file(self, path: str):
        if path.endswith((".h5", ".hdf5")):
            with h5py.File(path, "r") as f:
                if "smarts" in f:
                    smarts = f["smarts"][:]
                elif "objects" in f:
                    smarts = f["objects"][:]
        elif path.endswith((".smi", ".txt", ".sma")):
            with open(path, "r") as f:
                smarts = f.readlines()
        else:
            raise ValueError(f"Unknown file type: {path}")
        return smarts

    def featurize(self, rxn: rdRxn.ChemicalReaction) -> np.ndarray:
        fp = np.zeros((1, len(self._smarts)))
        fp[0, self._smarts.index(rdRxn.ReactionToSmarts(rxn))] = 1
        return fp.astype(bool)
    
    def fit(self, rxns: List[rdRxn.ChemicalReaction]):
        self._smarts = list(set([rdRxn.ReactionToSmarts(rxn) for rxn in rxns]))
        self._fnames = [f"rxn_{i}" for i in range(len(rxns))]
        return self

    def save_dict(self):
        d = super().save_dict()
        d["smarts"] = self._smarts
        return d
    
    def _load(self, d: dict | Any):
        super()._load(d)
        self._smarts = d["smarts"][:]
        