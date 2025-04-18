from typing import Type, List

import numpy as np

from ..storage import BaseStorage, MolStorage, RxnStorage
from ..featurize import BaseFeaturizer
from .route import SynthesisRoute, ActionTypes

class SynRouteFeaturizer(BaseFeaturizer):
    def __init__(self,
                 rfeaturizer: BaseFeaturizer,
                 pfeaturizer: BaseFeaturizer,
                 rxnfeaturizer: BaseFeaturizer):
        super().__init__()
        self._fnames = rxnfeaturizer.fnames
        self.rfeaturizer = rfeaturizer
        self.pfeaturizer = pfeaturizer
        self.rxnfeaturizer = rxnfeaturizer

    def featurize(self, route: SynthesisRoute):
        rxnfeats = [self.rxnfeaturizer.featurize(rxn)
                    for rxn in route.reactions]
        rxnfeat: np.ndarray = np.mean(rxnfeats, axis=0)
        return rxnfeat.reshape(1, -1)