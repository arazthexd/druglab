from typing import Tuple, Any

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as rdFP

from .base import BaseFeaturizer

class MorganFPFeaturizer(BaseFeaturizer):
    def __init__(self, radius: int = 2, size: int = 1024):
        super().__init__()
        self.radius = radius
        self.size = size
        self.generator = rdFP.GetMorganGenerator(radius, fpSize=size)
        self._fnames = [f"morgan{radius}_{i}" for i in range(size)]

    def featurize(self, mol: Chem.Mol | Tuple[Chem.Mol, int]) -> np.ndarray:
        if isinstance(mol, tuple):
            mol = mol[0]
        fp: np.ndarray = self.generator.GetFingerprintAsNumPy(mol)
        fp = fp.reshape(1, -1)
        return fp.astype(bool)
    
    def save_dict(self):
        d = super().save_dict()
        d["radius"] = self.radius
        d["size"] = self.size
        return d

    def _load(self, d: dict | Any):
        super()._load(d)
        self.radius = np.array(d["radius"]).item()
        self.size = np.array(d["size"]).item()
        self.generator = rdFP.GetMorganGenerator(self.radius, fpSize=self.size)