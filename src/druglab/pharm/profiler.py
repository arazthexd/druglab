from typing import List, Tuple, OrderedDict
import itertools

import numpy as np

from .ftypes import PharmArrowType, PharmSphereType, PharmFeatureType
from .features import PharmArrowFeats, PharmSphereFeats
from .pprofile import PharmProfile
from .pharmacophore import Pharmacophore, PharmacophoreList

class PharmProfiler:
    def __init__(self, 
                 ftypes: List[PharmFeatureType]):
        
        if isinstance(ftypes, dict):
            ftypes = list(ftypes.values())
        self.ftypes = ftypes

    def profile(self, pharm: Pharmacophore) -> PharmProfile:
        raise NotImplementedError()
    
class PharmDefaultProfiler(PharmProfiler):
    def __init__(self, 
                 ftypes: List[PharmFeatureType], 
                 ngroup: int = 2,
                 mindist: float = 0.0):
        
        super().__init__(ftypes)
        self.ngroup = ngroup
        self.mindist = mindist

        self.possible_gtys = np.array(list(
            itertools.combinations_with_replacement(
                range(len(self.ftypes)), r=self.ngroup
        ))).astype(int)
        
    def profile(self, 
                pharm: Pharmacophore | PharmacophoreList) -> PharmProfile:
        if isinstance(pharm, PharmacophoreList):
            profiles = [self.profile(pharm) for pharm in pharm.pharms]
            return sum(profiles)

        if pharm.n_feats < self.ngroup:
            nn = self.ngroup*(self.ngroup-1)//2
            return PharmProfile(
                tys=np.zeros((0, nn, 2), dtype=int),
                tyids=np.zeros((0,), dtype=int),
                n_tyids=self.possible_gtys.shape[0],
                vecs=np.zeros((0, nn, 3)),
                dists=np.zeros((0, nn)),
                dirs=np.zeros((0, nn, 2, 3)),
                cos=np.zeros((0, nn, 2)),
            )
        
        tys = np.array([
            self.ftypes.index(ftype) 
            for i, ftype in enumerate(pharm.ftypes) 
            for _ in range(pharm.feats[i].pos.shape[0])
        ])
        pos = pharm.pos
        dir = pharm.vec

        idx = np.argsort(tys)
        tys = tys[idx]
        pos = pos[idx]
        dir = dir[idx]

        idx = np.array(list(itertools.combinations(range(pharm.n_feats), 
                                                   r=self.ngroup))).astype(int)
        gtys = tys[idx]
        gpos = pos[idx]
        gdir = dir[idx]
        
        idx = np.array(list(itertools.combinations(range(gtys.shape[1]), 
                                                   r=2))).astype(int)
        pairtys = gtys[:, idx]
        pairvecs = gpos[:, idx[:, 1]] - gpos[:, idx[:, 0]]
        pairdirs = gdir[:, idx]
        pairdists = np.linalg.norm(pairvecs, axis=-1)
        paircos = np.divide(
            (pairdirs * pairvecs[:, :, None, :]).sum(axis=-1),
            np.linalg.norm(pairvecs, axis=-1, keepdims=True),
            out=np.ones(pairdirs.shape[:-1]) * np.nan,
            where=np.linalg.norm(pairvecs, axis=-1, keepdims=True) != 0
        )
        paircos = np.nan_to_num(paircos, nan=np.inf)

        idx = np.where((pairdists >= self.mindist).all(axis=-1))[0]
        pairtys = pairtys[idx]
        pairvecs = pairvecs[idx]
        pairdirs = pairdirs[idx]
        pairdists = pairdists[idx]
        paircos = paircos[idx]
        gtys = gtys[idx]

        idx = np.lexsort([pairdists, 
                        *paircos.transpose(2, 0, 1),
                        *pairtys.transpose(2, 0, 1)])
        rowidx = np.arange(idx.shape[0]).reshape(-1, 1)
        pairtys = pairtys[rowidx, idx]
        pairvecs = pairvecs[rowidx, idx]
        pairdirs = pairdirs[rowidx, idx]
        pairdists = pairdists[rowidx, idx]
        paircos = paircos[rowidx, idx]

        tyids = np.where(
            (gtys[:, None] == self.possible_gtys[None, :]).all(axis=-1))[1]
        idx = np.argsort(tyids)

        return PharmProfile(
            pairtys[idx],
            tyids[idx],
            self.possible_gtys.shape[0],
            pairvecs[idx],
            pairdists[idx],
            pairdirs[idx],
            paircos[idx]
        )