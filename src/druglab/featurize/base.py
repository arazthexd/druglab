from typing import List, Any
import dill
import h5py

import numpy as np

from druglab import featurize as featurize_submodule

class BaseFeaturizer:
    def __init__(self):
        self._fnames: List[str] = []

    def featurize(self, object) -> np.ndarray:
        raise NotImplementedError()
    
    def fit(self, objects: List[Any]):
        return self
    
    def save_dict(self):
        return {
            "fnames": self._fnames,
        }
    
    def save(self, path, close: bool = True):
        f = self._save(self.save_dict(), path, close=False)
        f.attrs["_name_"] = self.__class__.__name__
        if close:
            f.close()
        else:
            return f
    
    @staticmethod
    def _save(save_dict: dict, path: str, close: bool = True):
        f = h5py.File(path, "w")
        for k, v in save_dict.items():
            f[k] = v
        if close:
            f.close()
        else:
            return f
    
    def _load(self, d: h5py.Dataset):
        self._fnames = d["fnames"][:]

    @staticmethod
    def load(src: str | h5py.Group):
        if isinstance(src, str):
            f = h5py.File(src, "r")
        else:
            f = src
        name = f.attrs["_name_"]
        featurizer: BaseFeaturizer = getattr(featurize_submodule, name)()
        featurizer._load(f)
        return featurizer
    
    @property
    def fnames(self) -> List[str]:
        return self._fnames