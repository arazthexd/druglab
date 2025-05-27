from __future__ import annotations
from typing import List
from dataclasses import dataclass
import itertools

import numpy as np

from .pharmacophore import Pharmacophore, PharmacophoreList

@dataclass(repr=False)
class PharmProfile:
    tys: np.ndarray
    tyids: np.ndarray
    n_tyids: int

    vecs: np.ndarray
    dists: np.ndarray
    dirs: np.ndarray
    cos: np.ndarray

    subids: List[List[int]] = None

    def __post_init__(self):
        if self.subids is None:
            self.subids = [list(range(self.tys.shape[0]))]
    
    def __add__(self, other: PharmProfile):
        assert self.n_tyids == other.n_tyids

        idadd = 0
        for ids in self.subids:
            idadd = max(idadd, max(ids, default=-1)+1)

        new = PharmProfile(
            np.concatenate([self.tys, other.tys]),
            np.concatenate([self.tyids, other.tyids]),
            self.n_tyids,
            np.concatenate([self.vecs, other.vecs]),
            np.concatenate([self.dists, other.dists]),
            np.concatenate([self.dirs, other.dirs]),
            np.concatenate([self.cos, other.cos]),
            self.subids + [[i+idadd for i in ids] for ids in other.subids]
        )
        return new
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
    
    def __getitem__(self, idx):
        return PharmProfile(
            self.tys[self.subids[idx]],
            self.tyids[self.subids[idx]],
            self.n_tyids,
            self.vecs[self.subids[idx]],
            self.dists[self.subids[idx]],
            self.dirs[self.subids[idx]],
            self.cos[self.subids[idx]]
        )