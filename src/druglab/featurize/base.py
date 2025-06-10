from __future__ import annotations
from typing import List, Type, Optional
import dill
from abc import ABC, abstractmethod

import numpy as np

class BaseFeaturizer(ABC):
    def __init__(self, dtype: Optional[Type[np.dtype]] = None):
        self._dtype = dtype
        
    def featurize(self, *objects) -> np.ndarray:
        """Featurize objects (not a batch!) into a numpy array.
        
        This method should not consider the dimension for stacking object feats
            and will be one dimensional if features only need one dimension.
        """
        feats = self.featurize_(*objects)
        return feats.astype(self.dtype)
    
    @abstractmethod
    def featurize_(self, *objects) -> np.ndarray:
        pass
    
    def save(self, path):
        with open(path, "wb") as f:
            dill.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            fzer = dill.load(f)
        return fzer
        
    @property
    @abstractmethod
    def fnames(self) -> List[str]:
        pass

    @property
    def dtype(self) -> Type[np.dtype]:
        return self._dtype
    
    @property
    def name(self) -> str | None:
        return None