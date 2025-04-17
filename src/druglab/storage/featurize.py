from typing import List

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as rdFP

class BaseFeaturizer:
    def __init__(self):
        self._fnames: List[str] = []

    def featurize(self, object) -> np.ndarray:
        raise NotImplementedError()
    
    @property
    def fnames(self) -> List[str]:
        return self._fnames
    
class MorganFPFeaturizer(BaseFeaturizer):
    def __init__(self, radius: int = 2, size: int = 1024):
        super().__init__()
        self.radius = radius
        self.size = size
        self.generator = rdFP.GetMorganGenerator(radius, fpSize=size)
        self._fnames = [f"morgan{radius}_{i}" for i in range(size)]

    def featurize(self, mol: Chem.Mol) -> np.ndarray:
        fp: np.ndarray = self.generator.GetFingerprintAsNumPy(mol)
        return fp.astype(np.bool)
