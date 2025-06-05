from typing import Tuple, Any, List
from abc import ABC
from enum import Enum

import numpy as np

from rdkit import Chem, rdBase
from rdkit.Chem import rdChemReactions as rdRxn, rdFingerprintGenerator as rdFP
from rdkit.Chem.rdChemReactions import ChemicalReaction as Rxn

from .base import BaseFeaturizer

class ReactionFeaturizer(BaseFeaturizer):
    pass

class DifferenceFPFeaturizer(ReactionFeaturizer):
    def __init__(self,
                 mol_featurizer: BaseFeaturizer,
                 include_agents: bool = False,
                 agent_weight: int = 1,
                 nonagent_weight: int = 10) -> None:
        super().__init__(dtype=np.int8)
        self.mol_featurizer = mol_featurizer
        self.include_agents = include_agents
        self.agent_weight = agent_weight
        self.nonagent_weight = nonagent_weight
        self._fnames = ["Diff"+fn for fn in mol_featurizer.fnames]

    def featurize_(self, rxn: Rxn, *args) -> np.ndarray:
        blocker = rdBase.BlockLogs()
        
        rfp = sum(self.mol_featurizer.featurize(r) for r in rxn.GetReactants())
        pfp = sum(self.mol_featurizer.featurize(p) for p in rxn.GetProducts())
        if not self.include_agents:
            return pfp - rfp
        afp = sum(self.mol_featurizer.featurize(a) for a in rxn.GetAgents())
        return self.nonagent_weight * (pfp - rfp) + self.agent_weight * afp

    @property
    def name(self) -> str:
        return "Diff"+self.mol_featurizer.name
    
    @property
    def fnames(self) -> List[str]:
        return self._fnames
