from __future__ import annotations
from typing import List, Any, Tuple, Type, Dict
import itertools
import h5py

import random

import mpire
from mpire import WorkerPool

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..featurize import BaseFeaturizer
from ..prepare import BasePreparation
from .. import featurize as featurize_submodule
from .. import storage as storage_submodule

def parallel_run(func, iterable, 
                 n_workers: int = 1,
                 desc: str = ""):
    if n_workers == 1:
        results = [func(a) for a in iterable]
    elif n_workers > 1:
        with WorkerPool(n_workers) as pool:
            results = pool.map(func, iterable, progress_bar=True,
                               concatenate_numpy_output=False,
                               progress_bar_options={"desc": desc})
    elif n_workers == -1:
        with WorkerPool(mpire.cpu_count()) as pool:
            results = pool.map(func, iterable, progress_bar=True,
                               concatenate_numpy_output=False,
                               progress_bar_options={"desc": desc})
    else:
        raise ValueError("Invalid n_workers: {}".format(n_workers))
    
    failed_idx = [i for i, x in enumerate(results) if x is None]
    first_success_idx = next(i for i, x in enumerate(results) if x is not None)

    if len(failed_idx) > 0:
        print("WARNING: Failed to run for {} objects".format(len(failed_idx)))
    
    if isinstance(results[first_success_idx], np.ndarray):
        results = np.concatenate([
            results[i] if results[i] is not None
            else np.ones(results[first_success_idx].shape) * np.nan
            for i in range(len(results))
        ], axis=0)
    
    return results
    
class BaseStorage:
    def __init__(self, 
                 objects: List[Any] = None, 
                 fdtype: Type[np.dtype] = np.float32,
                 feats: np.ndarray = None,
                 fnames: List[str] = None,
                 featurizers: List[BaseFeaturizer | dict] = None):
        if objects is None:
            objects = []
        self.objects: List[Any] = objects

        self._fdtype = fdtype if feats is None else feats.dtype
        self.fnames: List[str] = [] if fnames is None else fnames
        self.feats: np.ndarray = \
            np.empty((len(self), 
                      len(self.fnames)), 
                      dtype=fdtype) if feats is None else feats
        self.featurizers: List[BaseFeaturizer] = \
            [] if featurizers is None else featurizers
        
        if featurizers is None:
            featurizers = []
        self.featurizers = []
        for featurizer in featurizers:
            if isinstance(featurizer, dict):
                self.featurizers.append(BaseFeaturizer(**featurizer))
            elif featurizer is None:
                self.featurizers.append(None)
            else:
                try:
                    self.featurizers.append(BaseFeaturizer.load(featurizer))
                except:
                    self.featurizers.append(featurizer)
                
        self.knn: NearestNeighbors = None
    
    def sample(self, n: int = 1):
        idx = random.choices(range(len(self)), k=n)
        if n == 1:
            return self[idx[0]]
        return [self[i] for i in idx]
    
    def nearest(self, feats: np.ndarray, k: int = None) -> Tuple[np.ndarray,
                                                                 np.ndarray]:
        return self.knn.kneighbors(feats, 
                                   n_neighbors=k, 
                                   return_distance=True)
    
    def prepare(self, 
                preparation: BasePreparation,
                inplace: bool = True,
                n_workers: int = 1) -> Any:
        
        def prepare_obj(*obj):
            if len(obj) == 1:
                obj = obj[0]
            
            try: 
                return preparation.prepare(obj)
            except:
                return None
        
        prepped = parallel_run(prepare_obj,
                               self.objects,
                               n_workers=n_workers,
                               desc=f"Preparing {self.__class__.__name__}")
        
        if inplace:
            self.objects = prepped
        else:
            return self.__class__(prepped)
        
    def featurize(self, 
                  featurizer: BaseFeaturizer, 
                  overwrite: bool = False,
                  n_workers: int = 1):
        if overwrite:
            self.feats = np.empty((len(self), 0), dtype=self._fdtype)
            self.fnames = []
            self.featurizers = []

        def featurize_obj(*obj):
            if len(obj) == 1:
                obj = obj[0]
            
            try: 
                return featurizer.featurize(obj)
            except:
                return np.ones((1, len(featurizer.fnames))) * np.nan
        
        newfeats = parallel_run(featurize_obj,
                                self.objects,
                                n_workers=n_workers,
                                desc=f"Featurizing {self.__class__.__name__}")
        
        self.feats = np.concatenate((self.feats, newfeats), axis=1)
        self.fnames.extend(featurizer.fnames)
        self.featurizers.append(featurizer)

    def init_knn(self, knn: NearestNeighbors):
        self.knn = knn
        self.knn.fit(self.feats)

    def extend(self, storage: BaseStorage):
        if isinstance(storage, list):
            storage = self.__class__(storage)
        
        if len(self) == 0:
            self.objects = storage.objects.copy()
            self.feats = storage.feats.copy()
            self.fnames = storage.fnames.copy()
            return
        
        assert self.fnames == storage.fnames
        self.objects.extend(storage.objects)
        self.feats = np.concatenate((self.feats, storage.feats), axis=0)

    def subset(self, 
               idx: List[int] | np.ndarray, 
               inplace: bool = False) -> BaseStorage | None:
        feats = self.feats[idx]
        
        if not inplace:
            return self.__class__([self[i] for i in idx],
                                  feats=feats,
                                  fnames=self.fnames,
                                  featurizers=self.featurizers)
        
        remove_idx = [i for i in range(len(self)) if i not in idx]
        counter = 0
        for idx in remove_idx:
            del self[idx-counter]
            counter += 1
        self.feats = feats

    def _serialize_object(self, obj):
        raise NotImplementedError()

    def _unserialize_object(self, obj):
        raise NotImplementedError()
    
    def save_dict(self):
        d = {
            "objects": [self._serialize_object(obj) for obj in self.objects],
            "feats": self.feats,
            "fnames": self.fnames
        }
        for i, featurizer in enumerate(self.featurizers):
            for k, v in featurizer.save_dict().items():
                d[f"featurizer{i}/{k}"] = v
        return d
    
    def save(self, dst: str | h5py.Dataset, close: bool = True):
        f = self._save(self.save_dict(), dst, close=False)
        if close:
            f.close()
        else:
            return f

    def _save(self, 
              save_dict: dict, 
              dst: str | h5py.Dataset, 
              close: bool = True):
        if isinstance(dst, str):
            f = h5py.File(dst, "w")
        else:
            f = dst

        f.attrs["_name_"] = self.__class__.__name__
        for k, v in save_dict.items():
            f[k] = v
        for i, featurizer in enumerate(self.featurizers):
            f[f"featurizer{i}"].attrs["_name_"] = featurizer.__class__.__name__
        if close:
            f.close()
        else:
            return f

    def _load(self, d: h5py.Dataset | h5py.Group):
        self.objects = [self._unserialize_object(obj) for obj in d["objects"]]
        self.feats = d["feats"][:]
        self.fnames = d["fnames"][:]
        
        for k, v in d.items():
            if not k.startswith("featurizer"):
                continue
            featurizer: BaseFeaturizer = getattr(featurize_submodule, 
                                                 v.attrs["_name_"])()
            featurizer._load(v)
            self.featurizers.append(featurizer)
        
    @classmethod
    def load(cls, src: str | h5py.Dataset | h5py.Group):
        if isinstance(src, str):
            f = h5py.File(src, "r")
        else:
            f = src
        
        if cls != BaseStorage:
            store = cls()
            store._load(f)
        else:
            store: BaseStorage = getattr(storage_submodule, 
                                         f.attrs["_name_"])()
            store._load(f)
        
        if isinstance(src, str):
            f.close()

        return store

    def __getitem__(self, idx: int | List[int] | np.ndarray):
        if isinstance(idx, int):
            return self.objects[idx]
        
        if isinstance(idx, list) \
            or (isinstance(idx, np.ndarray) and idx.ndim == 1):
            return [self.objects[i] for i in idx]
        
        if isinstance(idx, np.ndarray) and idx.ndim == 2:
            return [[self.objects[i] for i in ids] for ids in idx]
        
        return self.objects[idx]
    
    def __setitem__(self, idx, obj):
        self.objects[idx] = obj

    def __delitem__(self, index):
        del self.objects[index]
    
    def __repr__(self):
        return self.__class__.__name__ + \
            "({} objects, {} feats)".format(len(self), self.feats.shape[1])
    
    def __len__(self):
        return len(self.objects)
    
    def __iter__(self):
        return iter(self.objects)
    
    def __contains__(self, obj):
        return obj in self.objects

    @property
    def fdtype(self):
        return self._fdtype
    
    @fdtype.setter
    def fdtype(self, fdtype):
        self._fdtype = fdtype
        self.feats = self.feats.astype(fdtype)

    @property
    def nfeats(self) -> int:
        return self.feats.shape[1]