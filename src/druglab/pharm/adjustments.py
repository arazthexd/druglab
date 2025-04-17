from rdkit import Chem

import numpy as np
from scipy.spatial.distance import cdist

from .ftypes import PharmSingleType
from .features import (
    PharmArrowSingles, PharmSphereSingles, PharmDistancePairs,
)
from .pharmacophore import Pharmacophore

class BasePharmAdjuster:
    def __init__(self):
        self.name: str = "unnamed"

    def adjust(self, 
               pharmacophore: Pharmacophore, 
               conformer: Chem.Conformer) -> None:
        pass

class PharmStericAdjuster(BasePharmAdjuster):
    def __init__(self):
        super().__init__()
        self.name = "steric"
        self.default_dist = 3.0
        self.default_radii = 1.2
    
    def adjust(self, 
               pharmacophore: Pharmacophore, 
               conformer: Chem.Conformer) -> None:
        
        atcoords = conformer.GetPositions()
        periodic: Chem.PeriodicTable = Chem.GetPeriodicTable()
        atradii = np.array([
            periodic.GetRvdw(at.GetAtomicNum())
            for at in conformer.GetOwningMol().GetAtoms()
        ])

        # Arrow Features
        arrowfs = pharmacophore.arrows
        ftdists = np.array([
            arrowfs.types.types[idx].adjopts\
                .get(self.name, {})\
                    .get("dist", self.default_dist) 
            for idx in arrowfs.tyidx 
        ])
        ftposradii = np.array([
            arrowfs.types.types[idx].adjopts\
                .get(self.name, {})\
                    .get("radii", self.default_radii)
            for idx in arrowfs.tyidx 
        ])
        refpos = arrowfs.pos + arrowfs.vec * ftdists[:, None]
        rp_at_dists = cdist(refpos, atcoords)
        arrow_clash_idx = np.argwhere(
            rp_at_dists < atradii[None, :] + ftposradii[:, None]
        )[:, 0]
        pharmacophore.remove_arrows(arrow_clash_idx)

        # Sphere Features
        # ...

        

    

