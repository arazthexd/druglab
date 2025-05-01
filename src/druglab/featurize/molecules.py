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

    def featurize(self, mol: Chem.Mol) -> np.ndarray:
        fp: np.ndarray = self.generator.GetFingerprintAsNumPy(mol)
        fp = fp.reshape(1, -1)
        return fp.astype(bool)
    
    def get_params(self):
        return {
            "radius": self.radius,
            "size": self.size
        }
    
    def set_params(self, **kwargs):
        self.radius = kwargs.get("radius", self.radius)
        self.size = kwargs.get("size", self.size)
        self.generator = rdFP.GetMorganGenerator(self.radius, fpSize=self.size)
        self._fnames = [f"morgan{self.radius}_{i}" for i in range(self.size)]