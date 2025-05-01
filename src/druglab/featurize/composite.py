from typing import List

import numpy as np

from .base import BaseFeaturizer

class CompositeFeaturizer(BaseFeaturizer):
    def __init__(self, featurizers: List[BaseFeaturizer]):
        super().__init__()
        self.featurizers = featurizers

    def add_featurizer(self, featurizer: BaseFeaturizer):
        self.featurizers.append(featurizer)

    def featurize(self, object) -> np.ndarray:
        all_feats = []
        for featurizer in self.featurizers:
            feats = featurizer.featurize(object)
            all_feats.append(feats)
        return np.concatenate(all_feats, axis=1)
    
    def get_params(self):
        return {
            f"{i}-{k}": v
            for i, f in enumerate(self.featurizers)
            for k, v in f.get_params().items()
        }
    
    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            try:
                i, k = k.split("-", maxsplit=1)
                i = int(i)
            except:
                for f in self.featurizers:
                    if k in f.get_params():
                        f.set_params({k: v})
                        break
                continue

            self.featurizers[i].set_params({k: v})

    @property
    def fnames(self) -> List[str]:
        return [f"{i}:{fn}"
                for i, featurizer in enumerate(self.featurizers)
                for fn in featurizer.fnames]