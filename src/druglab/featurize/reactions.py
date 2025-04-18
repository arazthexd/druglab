from typing import List

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdChemReactions as rdRxn

from .base import BaseFeaturizer

class RxnOneHotFeaturizer(BaseFeaturizer):
    def __init__(self, smiles: List[str] = None):
        super().__init__()
        if smiles is None:
            smiles = []
        self._smiles = smiles
        self._fnames = [f"rxn_{i}" for i in range(len(self._smiles))]

    def featurize(self, rxn: rdRxn.ChemicalReaction) -> np.ndarray:
        fp = np.zeros((1, len(self._smiles)))
        fp[0, self._smiles.index(rdRxn.ReactionToSmiles(rxn))] = 1
        return fp.astype(np.bool)
    
    def fit(self, rxns: List[rdRxn.ChemicalReaction]):
        self._smiles = list(set([rdRxn.ReactionToSmiles(rxn) for rxn in rxns]))
        self._fnames = [f"rxn_{i}" for i in range(len(rxns))]
        return self

    def get_params(self):
        return {
            "smiles": self._smiles,
        }
    
    def set_params(self, **kwargs):
        self._smiles = kwargs["smiles"]
        self._fnames = [f"rxn_{i}" for i in range(len(self._smiles))]
        