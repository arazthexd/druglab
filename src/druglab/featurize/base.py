from typing import List, Any
import dill

import numpy as np

class BaseFeaturizer:
    def __init__(self):
        self._fnames: List[str] = []

    def featurize(self, object) -> np.ndarray:
        raise NotImplementedError()
    
    def fit(self, objects: List[Any]):
        return self

    def get_params(self):
        return {}
    
    def set_params(self, **kwargs):
        pass

    def save(self, path):
        try:
            with open(path, "wb") as f:
                dill.dump(self, f)
        except:
            with open(path, "wb") as f:
                dill.dump({
                    "generator": self.__class__,
                    "params": self.get_params()
                }, f)
    
    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            fizer = dill.load(f)
        
        if isinstance(fizer, BaseFeaturizer):
            return fizer
        else:
            return fizer["generator"](**fizer["params"])
    
    @property
    def fnames(self) -> List[str]:
        return self._fnames