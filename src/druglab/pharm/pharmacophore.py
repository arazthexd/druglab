from __future__ import annotations
from typing import List
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import pdist, squareform

from .ftypes import (
    BasePharmType,
    PharmSingleType, PharmSingleTypes,
    PharmPairType, PharmPairTypes
)
from .features import (
    PharmArrowSingles, PharmSphereSingles,
    PharmDistancePairs
)

@dataclass(repr=False)
class Pharmacophore:
    arrows: PharmArrowSingles
    spheres: PharmSphereSingles 
    distances: PharmDistancePairs

    @classmethod
    def empty(cls, 
              stypes: PharmSingleTypes, 
              ptypes: PharmPairTypes = None) -> Pharmacophore:
        arrows = PharmArrowSingles(types=stypes.arrows)
        spheres = PharmSphereSingles(types=stypes.spheres)

        if ptypes is None:
            ptypes = PharmPairTypes()
        distances = PharmDistancePairs(types=ptypes.distances)

        return cls(arrows=arrows, 
                   spheres=spheres, 
                   distances=distances)
    
    def infere_distances(self, overwrite: bool = True):

        if overwrite:
            self.distances = PharmDistancePairs(types=self.ptypes.distances)
        
        tyidx_arrow = self.arrows.tyidx
        tyidx_sphere = self.spheres.tyidx
        tyidx = np.concatenate((tyidx_arrow, 
                                tyidx_sphere+len(self.arrows.types)), axis=0)

        pos_arrow = self.arrows.pos
        pos_sphere = self.spheres.pos
        pos = np.concatenate((pos_arrow, pos_sphere), axis=0)
        dists_matrix = squareform(pdist(pos))

        for pairtypeidx, (tyname1, tyname2) in enumerate(
            zip(self.ptypes.distances.memnames1, 
                self.ptypes.distances.memnames2)):
            
            typeidx1 = self.stypes.names.index(tyname1)
            typeidx2 = self.stypes.names.index(tyname2)

            idx1 = np.where(tyidx == typeidx1)[0]
            idx2 = np.where(tyidx == typeidx2)[0]
            
            dists = []
            memidx = []
            for i in idx1:
                for j in idx2:
                    if i < j:  # ensure we only get each pair once
                        memidx.append([i, j])
                        dists.append(dists_matrix[i,j])
            dists = np.array(dists)

            pairs = PharmDistancePairs(
                types=self.ptypes.distances,
                tyidx=(np.ones(dists.shape[0]) * pairtypeidx).astype(int),
                memidx=np.array(memidx),
                diff=dists
            )
            self.add_distances(pairs)
    
    def add_stypes(self, stypes: PharmSingleTypes):
        self.arrows.types += stypes.arrows
        self.spheres.types += stypes.spheres

    def add_ptypes(self, ptypes: PharmPairTypes):
        self.distances.types += ptypes.distances
    
    def add_arrows(self, arrows: PharmArrowSingles):
        self.arrows += arrows

    def remove_arrows(self, idx: int | List[int] | np.ndarray):
        if isinstance(idx, int):
            idx = [idx]
        idx = list(idx)
        self.arrows = self.arrows[[i for i in range(len(self.arrows)) 
                                   if i not in idx]]
        
        dist_idx = [i for i in range(len(self.distances))
                    if i in self.distances.memidx.flatten()]
        self.remove_distances(dist_idx)
    
    def add_spheres(self, spheres: PharmSphereSingles):
        self.spheres += spheres

    def add_distances(self, distances: PharmDistancePairs):
        self.distances += distances

    def remove_distances(self, idx: int | List[int] | np.ndarray):
        if isinstance(idx, int):
            idx = [idx]
        idx = list(idx)
        self.distances = self.distances[[i for i in range(len(self.distances)) 
                                         if i not in idx]]

    def __add__(self, other: Pharmacophore):
        return Pharmacophore(
            arrows=self.arrows + other.arrows,
            spheres=self.spheres + other.spheres,
            distances=self.distances + other.distances
        )

    @property
    def types(self) -> List[BasePharmType]:
        return (
            list(self.arrows.types) 
            + list(self.spheres.types) 
            + list(self.distances.types)
        )
    
    @property
    def stypes(self) -> PharmSingleTypes:
        return self.arrows.types + self.spheres.types
    
    @property
    def ptypes(self) -> PharmPairTypes:
        return self.distances.types
    

