from typing import List, Optional, Type

import numpy as np

from .base import BaseFeaturizer

class CompositeFeaturizer(BaseFeaturizer):
    def __init__(self, 
                 featurizers: List[BaseFeaturizer] = None,
                 dtype: Optional[Type[np.dtype]] = None):
        super().__init__(dtype)
        if featurizers is None:
            featurizers = []
        self.featurizers = featurizers

    def featurize_(self, *objects) -> np.ndarray:
        all_feats = []
        for featurizer in self.featurizers:
            feats = featurizer.featurize(*objects)
            all_feats.append(feats)
        return np.concatenate(all_feats, dtype=self.dtype)

    @property
    def fnames(self) -> List[str]:
        return [f"{i}:{fn}"
                for i, featurizer in enumerate(self.featurizers)
                for fn in featurizer.fnames]