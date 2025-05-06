from __future__ import annotations
from typing import List, Tuple, Dict, OrderedDict
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist

from .ftypes import PharmFeatureType, PharmArrowType
from .pharmacophore import Pharmacophore
from .pprofile import (
    PharmProfile, PharmDAProfile
)

class PharmProfiler:

    def __init__(self, ftypes: OrderedDict[str, PharmFeatureType]):
        self.ftypes: List[PharmFeatureType] = list(ftypes.values())
        self.combids: List[Tuple[int]] = self.combinations(self.ftypes)

    def __repr__(self):
        return (f"{self.__class__.__name__}(nftypes={len(self.ftypes)}, "
                f"ncombids={len(self.combids)})")
    
    @staticmethod
    def combinations(ftypes: List[PharmFeatureType]):
        raise NotImplementedError()

    def profile(self, pharmacophore: Pharmacophore) -> PharmProfile:
        raise NotImplementedError()
    
    def _get_fts(self, pharmacophore: Pharmacophore, *ids: Tuple[int]):
        return (pharmacophore.get_ftype(i) for i in ids)
    
    def _get_ftids(self, pharmacophore: Pharmacophore, *ids: Tuple[int]):
        return tuple(self.ftypes.index(ft) 
                     for ft in self._get_fts(pharmacophore, *ids))
    
    def _get_dists(self, pharmacophore: Pharmacophore, *ids: Tuple[int]):
        pos = pharmacophore.pos[list(ids)]
        return pdist(pos)
    
    def _get_tips(self, pharmacophore: Pharmacophore, *ids: Tuple[int]):
        pos = pharmacophore.pos[list(ids)]
        vec = pharmacophore.vec[list(ids)]
        rad = pharmacophore.radius[list(ids)]
        assert np.isnan(vec).sum() == 0
        return pos + vec * rad
    
class PharmDefaultProfiler(PharmProfiler):
    @staticmethod
    def combinations(ftypes: List[PharmFeatureType]) \
        -> List[Tuple[int, int]]:
        
        arrowftids = [i for i, ft in enumerate(ftypes) 
                      if isinstance(ft, PharmArrowType)]
        return [
            (i, j)
            for i in arrowftids
            for j in range(len(ftypes))
        ]

    def profile(self, pharmacophore: Pharmacophore):
        ids1, ids2 = [], []
        ftids1, ftids2 = [], []
        for i, j in combinations(range(pharmacophore.n_feats), 2):
            ftids = self._get_ftids(pharmacophore, i, j)
            if ftids in self.combids:
                ids1.append(i)
                ids2.append(j)
                ftids1.append(ftids[0])
                ftids2.append(ftids[1])


        ftids = np.array([ftids1, ftids2]).T
        ids = np.array([ids1, ids2]).T
        dirs = pharmacophore.vec
        dirs = dirs[ids1]

        vectors = pharmacophore.pos[ids2] - pharmacophore.pos[ids1]
        distances = np.linalg.norm(vectors, axis=-1)

        keep_idx = np.where(distances > 0.05)
        ftids = ftids[keep_idx]
        dirs = dirs[keep_idx]
        vectors = vectors[keep_idx]
        distances: np.ndarray = distances[keep_idx]
        
        vectors = vectors / distances[:, None]

        angles = dirs * vectors
        angles = np.clip(np.sum(angles, axis=-1), -1, 1)
        angles: np.ndarray = np.arccos(angles)

        dp = PharmDAProfile(
            typeids=ftids,
            pharm=pharmacophore,
            featids=ids,
            dists=distances.reshape(-1, 1),
            angles=angles.reshape(-1, 1)
        )
        return dp