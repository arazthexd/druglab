from __future__ import annotations
from typing import List, Tuple, Dict, OrderedDict, Union, Callable
from dataclasses import dataclass, field
from itertools import combinations, product

import numpy as np
from scipy.optimize import linear_sum_assignment

from .ftypes import PharmFeatureType, PharmArrowType
from .pharmacophore import Pharmacophore

@dataclass(repr=False)
class PharmProfile:
    typeids: np.ndarray # (n_data, ...) TODO: not a single id but ftids
    pharm: Pharmacophore
    featids: np.ndarray
    agg_func: Callable[[np.ndarray], int] = None
    orig_atids: List[Tuple[Tuple[int]]] = None

    def __post_init__(self):
        if self.agg_func is None:
            self.agg_func = np.sum
        if self.orig_atids is None:
            self.orig_atids = [(tuple(), ) 
                               for _ in range(self.typeids.shape[0])]

    def __repr__(self):
        return (f"{self.__class__.__name__}({self.typeids.shape[0]})")
    
    def match(self, 
              other: PharmProfile,
              return_vals: bool = False) -> Tuple[np.ndarray, ...]:
        scores = self._score_matrix(other)
        mask = (self.typeids[:, None] == other.typeids[None, :]).all(axis=-1)
        
        if self._maximize:
            scores[~mask] = 0
        else:
            scores[~mask] = 1000

        sidx, oidx = linear_sum_assignment(scores, maximize=self._maximize)
        scores = scores[sidx, oidx]
        if return_vals:
            return sidx, oidx, scores
        return sidx, oidx
    
    def score(self, other: PharmProfile) -> float:
        sidx, oidx, scores = self.match(other, return_vals=True)
        return self.agg_func(scores)
    
    def screen(self, 
               profiles: List[PharmProfile],
               normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        scs = []
        for prof in profiles:
            try:
                sc = self.score(prof)
                if normalize:
                    sc = sc / prof.pharm.n_feats
                scs.append(sc)
            except:
                print(f"An error occured for index {len(scs)}")
                if self._maximize:
                    scs.append(0)
                else:
                    scs.append(1000)
        
        scs = np.array(scs)
        idx = np.argsort(scs)
        if self._maximize:
            idx = np.flip(idx)
        return idx, scs[idx]
        
    def _score_matrix(self, other: PharmProfile) -> np.ndarray:
        raise NotImplementedError()
    
    @property 
    def _maximize(self) -> bool:
        return False
    
@dataclass(repr=False)
class PharmDefaultProfile(PharmProfile):
    dists: np.ndarray = None
    angles: np.ndarray = None

    def __post_init__(self):
        super().__post_init__()
        if self.dists is None:
            raise ValueError()
        if self.angles is None:
            raise ValueError()
        
    def _score_matrix(self, other: PharmDefaultProfile):
        assert self.dists.shape[1] == other.dists.shape[1]
        assert self.angles.shape[1] == other.angles.shape[1]
        delta_dists = (self.dists[:, None, :] - other.dists[None, :, :]) ** 2
        delta_angles = (self.angles[:, None, :] - other.angles[None, :, :]) ** 2
        delta_dists = delta_dists.mean(axis=-1) ** 0.5
        delta_angles = delta_angles.mean(axis=-1) ** 0.5

        score_dists = 1 / (1 + 3*delta_dists)
        score_angles = (1 + np.cos(delta_angles)) / 2
        return score_dists * score_angles
        
    @property
    def _maximize(self) -> bool:
        return True
    
@dataclass(repr=False)
class PharmProfileList:
    profiles: List[PharmProfile] = field(default_factory=list)

    def __repr__(self):
        return (f"{self.__class__.__name__}({len(self.profiles)})")

    def match(self, other: PharmProfileList):
        scores = np.empty((len(self.profiles), len(other.profiles)))
        for i, prof1 in enumerate(self.profiles):
            for j, prof2 in enumerate(other.profiles):
                scores[i, j] = prof1.score(prof2)
        # print(np.where(scores == scores.max()))
        idx1, idx2 = np.where(scores == scores.max())
        idx1: int = idx1[0]
        idx2: int = idx2[0]

        return idx1, idx2, scores[idx1, idx2]
    
    def screen(self, plists: List[PharmProfileList], normalize: bool = True):
        
        ids1, ids2, scores = [], [], []
        for plist in plists:
            idx1, idx2, score = self.match(plist)
            prof_self = self.profiles[idx1]
            prof_other = plist.profiles[idx2]
            if normalize:
                score = score / prof_other.pharm.n_feats
            ids1.append(idx1)
            ids2.append(idx2)
            scores.append(score)
        
        ids1 = np.array(ids1)
        ids2 = np.array(ids2)
        scores = np.array(scores)
        idx = np.argsort(scores)
        if self.profiles[0]._maximize:
            idx = np.flip(idx)
        return ids1[idx], ids2[idx], idx, scores[idx]
            
            

        
    



        
    