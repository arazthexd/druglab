from typing import List, Tuple
import itertools

import numpy as np

from .ftypes import PharmArrowType, PharmSphereType, PharmFeatureType
from .features import PharmArrowFeats, PharmSphereFeats
from .pprofile import PharmProfile
from .pharmacophore import Pharmacophore, PharmacophoreList

class PharmProfiler:
    def __init__(self, ftypes: List[PharmFeatureType], narrows: int = 3):
        self.ftypes = ftypes
        self.types_arw = [ty for ty in ftypes
                          if isinstance(ty, PharmArrowType)]
        self.types_sph = [ty for ty in ftypes 
                          if isinstance(ty, PharmSphereType)]
        self.narrows = narrows

        nfts = len(self.types_arw)+len(self.types_sph)
        self.possible_pairs = np.array(
            list(itertools.combinations_with_replacement(range(nfts), r=2))
        )

        self.possible_afpairs = np.array(
            [(i, j) for i in range(len(self.types_arw)) for j in range(nfts)]
        )

        self.n_groups = (self.narrows, 4-self.narrows)
        assert sum(self.n_groups) == 4

        self.possible_arwcombtys = list(
            itertools.combinations_with_replacement(
                range(len(self.types_arw)), r=self.n_groups[0]
            )
        )
        self.possible_sphcombtys = list(
            itertools.combinations_with_replacement(
                range(len(self.types_arw), 
                      len(self.types_sph) + len(self.types_arw)), 
                      r=self.n_groups[1]
            )
        )
        self.possible_combtys = np.array([
            (*p1, *p2) 
            for p1, p2 in itertools.product(self.possible_arwcombtys, 
                                            self.possible_sphcombtys)
        ])
    
    def _analyze(self, pharm: Pharmacophore):
        tys_arw, tys_sph = [], []
        pos_arw, pos_sph = [], []
        rad_arw, rad_sph = [], []
        dir_arw = []
        for i, feat in enumerate(pharm.feats):
            if isinstance(feat, PharmArrowFeats):
                pos_arw.append(feat.pos)
                rad_arw.append(feat.radius)
                dir_arw.append(feat.vec)
                ty = pharm.ftypes[i]
                tyid = self.types_arw.index(ty)
                tys_arw.extend([tyid]*feat.pos.shape[0])
            elif isinstance(feat, PharmSphereFeats):
                pos_sph.append(feat.pos)
                rad_sph.append(feat.radius)
                ty = pharm.ftypes[i]
                tyid = self.types_sph.index(ty) + len(self.types_arw)
                tys_sph.extend([tyid]*feat.pos.shape[0])

        pos_arw = np.concatenate(pos_arw, axis=0)
        pos_sph = np.concatenate(pos_sph, axis=0)
        rad_arw = np.concatenate(rad_arw, axis=0)
        rad_sph = np.concatenate(rad_sph, axis=0)
        dir_arw = np.concatenate(dir_arw, axis=0)

        idx = np.argsort(tys_arw)
        pos_arw = pos_arw[idx]
        rad_arw = rad_arw[idx]
        dir_arw = dir_arw[idx]
        tys_arw = np.array(tys_arw)[idx]

        idx = np.argsort(tys_sph)
        pos_sph = pos_sph[idx]
        rad_sph = rad_sph[idx]
        tys_sph = np.array(tys_sph)[idx]

        return (pos_arw, pos_sph, 
                rad_arw, rad_sph, 
                dir_arw, 
                tys_arw, tys_sph)
    
    def _combination_variables(self,
                               pos_arw, pos_sph, 
                               tys_arw, tys_sph,
                               dir_arw):
        combs_arw = np.array(
            list(itertools.combinations(range(pos_arw.shape[0]), 
                                        r=self.narrows))
        )
        combs_sph = np.array(
            list(itertools.combinations(range(pos_sph.shape[0]), 
                                        r=4-self.narrows))
        )

        combtys_arw = tys_arw[combs_arw]
        combtys_sph = tys_sph[combs_sph]

        combpos_arw = pos_arw[combs_arw]
        combpos_sph = pos_sph[combs_sph]

        combdir_arw = dir_arw[combs_arw]

        idx = np.array(list(
            itertools.product(range(combpos_arw.shape[0]), 
                              range(combpos_sph.shape[0]))
        )).T

        combpos = np.concatenate([combpos_arw[idx[0]], combpos_sph[idx[1]]], 
                                 axis=1)
        combtys = np.concatenate([combtys_arw[idx[0]], combtys_sph[idx[1]]], 
                                 axis=1)
        combtyids = np.where(
            (combtys[:, None] == self.possible_combtys[None, :]).all(axis=-1)
        )[1]

        combdirs = combdir_arw[idx[0]]

        return (combpos, combtys, combtyids, combdirs)
    
    def _combination_distances(self,
                               possible_pairs,
                               combpos, combtys):
        combdvs = combpos[:, :, None, :] - combpos[:, None, :, :]

        combdists = np.linalg.norm(combdvs, axis=-1)
        combdists = combdists[:, *np.triu_indices(combpos.shape[1], k=1)]

        combpairtys = combtys[:, np.triu_indices(combpos.shape[1], k=1)]
        combpairtyids = np.where(
            (combpairtys.transpose(0, 2, 1)[:, :, None] 
             == possible_pairs).all(axis=-1)
        )[2].reshape((-1, 6))

        idx = np.lexsort([combdists, combpairtyids])
        rowidx = np.arange(combdists.shape[0])[:, None]
        combdists = combdists[rowidx, idx]
        combpairtys = combpairtys[rowidx, :, idx]
        combpairtyids = combpairtyids[rowidx, idx]

        return (combdists, combpairtys, combpairtyids)
    
    def _combination_angles(self,
                            possible_afpairs,
                            combpos, combtys, combdirs):
        idx = np.array([x 
                        for x in itertools.product(range(self.narrows), 
                                                   range(4))
                        if x[0] != x[1]])
        idx = idx.reshape(int(idx.shape[0]**0.5), -1, 2)

        combafdvs = (combpos[:, idx[..., 0]] - combpos[:, idx[..., 1]])
        combafdists = np.linalg.norm(combafdvs, axis=-1)

        combafcos = (combdirs[:, :, None] * combafdvs).sum(axis=-1)
        combafcos = np.divide(
            combafcos,
            combafdists,
            out=np.ones_like(combafcos) * 3,
            where=combafdists!=0
        )
        combafcos = combafcos.reshape((combafcos.shape[0], -1))

        combafpairtys = combtys[:, idx]
        combafpairtyids = np.where(
            (combafpairtys[:, :, :, None] == possible_afpairs).all(
                axis=-1
            )
        )[3].reshape((combafpairtys.shape[0], -1))

        idx = np.lexsort([combafcos, combafpairtyids], axis=-1)
        rowidx = np.arange(combafcos.shape[0])[:, None]
        combafcos = combafcos[rowidx, idx]
        combafpairtys = \
            combafpairtys.reshape(combafpairtys.shape[0], 
                                  -1, 
                                  2)[rowidx, idx, :]
        combafpairtyids = combafpairtyids[rowidx, idx]

        return (combafcos, combafpairtys, combafpairtyids)
    
    def profile(self, pharm: Pharmacophore | PharmacophoreList):
        if isinstance(pharm, PharmacophoreList):
            profiles = [self.profile(p) for p in pharm.pharms]
            return PharmProfile.merge(profiles)
            

        (pos_arw, pos_sph, 
         rad_arw, rad_sph, 
         dir_arw, 
         tys_arw, tys_sph) = self._analyze(pharm)
        
        combpos, combtys, combtyids, combdirs = self._combination_variables(
             pos_arw, pos_sph, 
             tys_arw, tys_sph,
             dir_arw
         )
        
        combdists, combpairtys, combpairtyids = self._combination_distances(
            self.possible_pairs, combpos, combtys
        )

        combafcos, combafpairtys, combafpairtyids = self._combination_angles(
            self.possible_afpairs, combpos, combtys, combdirs
        )
        
        return PharmProfile(
            pharm=pharm,
            subborderids=[combtyids.shape[0]],
            tys=combtys, 
            tyids=combtyids, 
            n_tyids=self.possible_combtys.shape[0],
            pair1tys=combpairtys, 
            pair1tyids=combpairtyids,
            pair1vals=combdists,
            n_pair1tyids=self.possible_pairs.shape[0],
            pair2tys=combafpairtys, 
            pair2tyids=combafpairtyids,
            pair2vals=combafcos,
            n_pair2tyids=self.possible_afpairs.shape[0])