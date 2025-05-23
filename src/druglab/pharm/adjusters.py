from typing import Dict

import numpy as np
from scipy.spatial.distance import cdist

from rdkit import Chem

from .pharmacophore import Pharmacophore
from .ftypes import PharmArrowType
from .features import PharmArrowFeats

ELEM_TABLE = Chem.GetPeriodicTable()

class InternalStericAdjuster:

    def __init__(self, tolerance: float = 0.2):
        self.tolerance = tolerance

    def adjust(self, 
               pharmacophore: Pharmacophore,
               conformer: Chem.Conformer):
        atcoords = conformer.GetPositions()
        atradii = np.array([
            ELEM_TABLE.GetRvdw(at.GetAtomicNum())
            for at in conformer.GetOwningMol().GetAtoms()
        ])

        arrow_ft_ids = [i for i, ft in enumerate(pharmacophore.ftypes)
                        if isinstance(ft, PharmArrowType)]
        for ftid in arrow_ft_ids:
            arrow_feats: PharmArrowFeats = pharmacophore.feats[ftid]
            compr = pharmacophore.ftypes[ftid].compradii

            tip_pos = arrow_feats.pos + \
                arrow_feats.vec * (arrow_feats.radius \
                                   + compr)[:, None]
            dists = cdist(tip_pos, atcoords)
            
            mindists = atradii + compr
            dists -= mindists
            dists = dists.min(axis=1)

            arrow_feats.remove_idx(np.where(dists+self.tolerance<0)[0])