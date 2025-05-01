from __future__ import annotations
from typing import List, Tuple
import itertools

import numpy as np
from scipy.spatial.distance import cdist

from .ftypes import (
    PharmFeatureType,
    PharmArrowType, PharmSphereType
)
from .features import (
    PharmFeatures,
    PharmArrowFeats, PharmSphereFeats, 
)

class Pharmacophore:
    def __init__(self):
        self.ftypes: List[PharmFeatureType] = []
        self.feats: List[PharmFeatures] = []
    
    def add_single(self, 
                   ftype: PharmFeatureType, 
                   *args, 
                   atidx: Tuple[int] = None):
        assert isinstance(ftype, PharmFeatureType)

        if ftype.name not in self.ftype_names:
            self.ftypes.append(ftype)
            self.feats.append(ftype.initiate_features())
        
        ftypeidx = self.ftype_names.index(ftype.name)
        self.feats[ftypeidx].add_features(*args, atidx=[atidx])

    def draw(self, view) -> None:
        for ftype, feat in zip(self.ftypes, self.feats):
            feat.draw(view, ftype.drawopts)
    
    def get_distance(self, idx1: int, idx2: int) -> float:
        return np.linalg.norm(self.pos[idx1] - self.pos[idx2])
    
    def get_ftype(self, idx: int) -> PharmFeatureType:
        for i, cumul in enumerate(
            itertools.accumulate([feat.pos.shape[0] 
                                  for feat in self.feats])):
            if idx < cumul:
                return self.ftypes[i]

    def __add__(self, other: Pharmacophore) -> Pharmacophore:
        new = Pharmacophore()
        new.ftypes = self.ftypes.copy()
        new.feats = self.feats.copy()

        for ftype, feat in zip(other.ftypes, other.feats):
            if ftype.name in new.ftype_names:
                ftypeidx = new.ftype_names.index(ftype.name)
                new.feats[ftypeidx].add_features(*feat.tuple(), 
                                                 atidx=feat.origin_atidx)
            else:
                new.ftypes.append(ftype)
                new.feats.append(feat)
        
        return new
    
    def __radd__(self, other: Pharmacophore) -> Pharmacophore:
        if other == 0:
            return self
        else:
            return self.__add__(other)
    
    @property
    def ftype_names(self) -> List[str]:
        return [ftype.name for ftype in self.ftypes]
    
    @property
    def n_feats(self) -> int:
        return sum([feat.pos.shape[0] for feat in self.feats])
    
    @property
    def pos(self) -> np.ndarray:
        return np.concatenate([feat.pos for feat in self.feats], axis=0)
    
    @property
    def vec(self) -> np.ndarray:
        vs = np.zeros((self.n_feats, 3))

        stid = 0
        for feat in self.feats:
            if isinstance(feat, PharmArrowFeats):
                vs[stid:stid+feat.pos.shape[0]] = feat.vec
            else:
                vs[stid:stid+feat.pos.shape[0]] = np.nan
            stid += feat.pos.shape[0]
        
        return vs
    
    @property
    def radius(self) -> np.ndarray:
        return np.concatenate([feat.radius for feat in self.feats], axis=0)
        # TODO: I should be careful about adding other types of features...
    
    @property
    def orig_atids(self) -> List[Tuple[int]]:
        return sum([feat.origin_atidx for feat in self.feats], start=[])
    
    def tyidx(self, ftypes: List[PharmFeatureType]) -> np.ndarray:
        return np.concatenate([np.ones(feat.pos.shape[0])*ftypes.index(self.ftypes[i])
                               for i, feat in enumerate(self.feats)])
