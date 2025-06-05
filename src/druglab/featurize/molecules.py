from typing import Tuple, Any, List

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as rdFP

from .base import BaseFeaturizer

class MoleculeFeaturizer(BaseFeaturizer):
    pass

class MorganFPFeaturizer(MoleculeFeaturizer):
    def __init__(self, 
                 size: int = 1024,
                 radius: int = 2,
                 count: bool = False,
                 chirality: bool = False):
        super().__init__(dtype=bool if not count else np.uint8)
        self.radius = radius
        self.size = size
        self.count = count
        self.chirality = chirality
        self._fnames = [f"MFP|{radius}|{i+1}/{size}" for i in range(size)]

    def featurize_(self, mol: Chem.Mol, *args) -> np.ndarray:
        fp: np.ndarray = self.generator.GetFingerprintAsNumPy(mol)
        return fp
    
    @property
    def fnames(self) -> List[str]:
        return self._fnames
    
    @property
    def generator(self) -> rdFP.FingerprintGenerator64:
        return rdFP.GetMorganGenerator(radius=self.radius, 
                                       fpSize=self.size,
                                       countSimulation=self.count,
                                       includeChirality=self.chirality)
    
    @property
    def name(self) -> str:
        return f"MorganFP|{self.radius}|{self.size}"