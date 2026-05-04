
from typing import Any, Dict, TypeVar, Generic

import numpy as np

from rdkit import Chem

from .chem import ChemTable

FeatureT = TypeVar("FeatureT")

class MoleculeRecord:
    mol: Chem.Mol
    metadata: Dict[str, Any]
    features: Dict[str, np.ndarray]

class MoleculeTable(ChemTable[MoleculeRecord], Generic[FeatureT]):
    pass