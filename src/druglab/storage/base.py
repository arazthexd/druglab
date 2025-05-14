from __future__ import annotations
from typing import List, Any, Tuple, Type, Dict
import itertools

import random

import mpire
from mpire import WorkerPool

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..featurize import BaseFeaturizer

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
        self.feats: np.ndarray = \
            np.empty((len(self), 0), dtype=fdtype) if feats is None else feats
        self.fnames: List[str] = [] if fnames is None else fnames
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
        
        if n_workers == 1:
            newfeats = [featurizer.featurize(obj) for obj in self]
            newfeats = np.concatenate(newfeats)
        
        elif n_workers > 1:
            with WorkerPool(n_workers) as pool:
                newfeats = pool.map(featurize_obj, 
                                    self.objects, 
                                    progress_bar=True)
        
        elif n_workers == -1:
            with WorkerPool(mpire.cpu_count()) as pool:
                newfeats = pool.map(featurize_obj, 
                                    self.objects, 
                                    progress_bar=True)

        else:
            raise ValueError("Invalid n_workers: {}".format(n_workers))
        
        self.feats = np.concatenate((self.feats, newfeats), axis=1)
        self.fnames.extend(featurizer.fnames)
        self.featurizers.append(featurizer)

    def init_knn(self, knn: NearestNeighbors):
        self.knn = knn
        self.knn.fit(self.feats)

    def extend(self, storage: BaseStorage):
        if isinstance(storage, list):
            storage = self.__class__(storage)
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