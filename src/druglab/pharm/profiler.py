from __future__ import annotations
from typing import List, Tuple, Dict, OrderedDict
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist

from .ftypes import PharmFeatureType, PharmArrowType
from .pharmacophore import Pharmacophore
from .pprofile import PharmProfile, PharmDistProfile

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
        return (self.ftypes.index(ft) 
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

    
class PharmFFFProfiler(PharmProfiler):

    @staticmethod
    def combinations(ftypes: List[PharmFeatureType]) \
        -> List[Tuple[int, int, int]]:
        
        return [(i,j,k) 
                for i in range(len(ftypes))
                for j in range(i, len(ftypes))
                for k in range(j, len(ftypes))]
    
    def profile(self, pharmacophore: Pharmacophore) -> PharmDistProfile:

        combtypeids = []
        all_dists = []
        orig_atids = []

        for i, j, k in combinations(range(pharmacophore.n_feats), r=3):
            ftidi, ftidj, ftidk = self._get_ftids(pharmacophore, i, j, k)
            
            try:
                combtypeid = self.combids.index((ftidi, ftidj, ftidk))
            except:
                continue
            
            dists = self._get_dists(pharmacophore, i, j, k)

            if min(dists) < 0.05:
                continue
            
            combtypeids.append(combtypeid)
            orig_atids.append(
                (
                    pharmacophore.orig_atids[i],
                    pharmacophore.orig_atids[j],
                    pharmacophore.orig_atids[k]
                )
            )
            all_dists.append(dists)

        keep_origatids = set()
        keep_idx = []
        for i, oatids in enumerate(orig_atids):
            if oatids not in keep_origatids:
                keep_origatids.add(oatids)
                keep_idx.append(i)

        combtypeids = np.array(combtypeids)[keep_idx]
        orig_atids = [orig_atids[i] for i in keep_idx]
        if len(all_dists) > 0:
            dists = np.stack(all_dists, axis=0)[keep_idx]
        else:
            dists = np.zeros((0, 3))
        
        return PharmDistProfile(
            combtypeids=combtypeids,
            orig_atids=orig_atids,
            dists=dists
        )
    
class PharmAFProfiler(PharmProfiler):
    
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
    
    def profile(self, pharmacophore: Pharmacophore) -> PharmDistProfile:
        
        combtypeids = []
        all_dists = []
        orig_atids = []

        for i, j in combinations(range(pharmacophore.n_feats), r=2):
            ftidi, ftidj = self._get_ftids(pharmacophore, i, j)
            
            try:
                combtypeid = self.combids.index((ftidi, ftidj))
            except:
                continue
            
            pos = pharmacophore.pos[[i, j]]
            pos = np.append(pos, self._get_tips(pharmacophore, i), axis=0)
            dists = pdist(pos)

            if min(dists) < 0.01:
                continue
            
            combtypeids.append(combtypeid)
            orig_atids.append(
                (
                    pharmacophore.orig_atids[i],
                    pharmacophore.orig_atids[j],
                )
            )
            all_dists.append(dists)

        keep_origatids = set()
        keep_idx = []
        for i, oatids in enumerate(orig_atids):
            if oatids not in keep_origatids:
                keep_origatids.add(oatids)
                keep_idx.append(i)

        combtypeids = np.array(combtypeids)[keep_idx]
        orig_atids = [orig_atids[i] for i in keep_idx]
        if len(all_dists) > 0:
            dists = np.stack(all_dists, axis=0)[keep_idx]
        else:
            dists = np.zeros((0, 3))
        
        return PharmDistProfile(
            combtypeids=combtypeids,
            orig_atids=orig_atids,
            dists=dists
        )