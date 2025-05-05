from __future__ import annotations
from typing import List, Tuple, Dict, OrderedDict
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .ftypes import PharmFeatureType, PharmArrowType
from .pharmacophore import Pharmacophore

@dataclass(repr=False)
class PharmProfile:
    combtypeids: np.ndarray
    orig_atids: List[Tuple[Tuple[int]]]

    def __repr__(self):
        return (f"{self.__class__.__name__}({self.combtypeids.shape[0]})")
    
@dataclass(repr=False)
class PharmDistProfile(PharmProfile):
    dists: np.ndarray # (n_dists, n_disttypes)

    def _diff_matrix(self, other: PharmDistProfile):
        assert self.dists.shape[1] == other.dists.shape[1]
        diff = (self.dists[:, None, :] - other.dists[None, :, :]) ** 2

        mask = self.combtypeids[:, None] == other.combtypeids[None, :]

        diff = diff.mean(axis=2) ** 0.5
        diff[~mask] = 1000
        return diff
    
    def _score_matrix(self, diff: np.ndarray):
        score = 1 / (1 + 3 * diff)
        return score
    
    def match(self, other: PharmDistProfile):
        diff = self._diff_matrix(other)
        sidx, oidx = linear_sum_assignment(diff)
        score = self._score_matrix(diff)
        score = score[sidx, oidx]
        soids = list(zip(sidx, oidx))
        idx = np.flip(np.argsort(score))
        return score[idx], [soids[i] for i in idx]
    
    def difference(self, other: PharmDistProfile):
        diff = self._diff_matrix(other)
        sidx, oidx = linear_sum_assignment(diff)
        diff = diff[sidx, oidx]
        return diff.mean()
    
    def screen(self, 
               profs: List[PharmDistProfile],
               pharms: List[Pharmacophore]):
        
        scs = []
        for prof, pharm in zip(profs, pharms):
            sc, _ = self.match(prof)
            sc = sc.sum() / pharm.n_feats
            scs.append(sc)
        
        return np.array(scs)

        
    