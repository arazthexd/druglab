from __future__ import annotations
from typing import List, Literal

import numpy as np

from rdkit import Chem

from ..storage import BaseStorage
from .pprofile import PharmProfile, PharmProfileList
from .profiler import PharmProfiler
from .generator import PharmGenerator
from .pharmacophore import Pharmacophore, PharmacophoreList

class PharmStorage(BaseStorage):
    objects: List[Pharmacophore | PharmacophoreList]

    def __init__(self, 
                 objects = None, 
                 generator: PharmGenerator = None,
                 fdtype = np.float32, 
                 feats = None, 
                 fnames = None, 
                 featurizers = None):
        super().__init__(objects, fdtype, feats, fnames, featurizers)
        self.generator = generator

    def __repr__(self):
        if isinstance(self.feats, np.ndarray):
            return self.__class__.__name__ + \
                "({} objects, featurized: {})".format(len(self), "no")
        elif isinstance(self.feats[0], (PharmProfile, PharmProfileList)):
            return self.__class__.__name__ + \
                "({} objects, featurized: {})".format(len(self), "yes")
        else:
            raise ValueError()

    @classmethod
    def from_mols(cls, 
                  mols: List[Chem.Mol],
                  generator: PharmGenerator,
                  confid: int | Literal["all"] = -1) -> PharmStorage:
        pharms = [generator.generate(mol, confid=confid) for mol in mols]
        return PharmStorage(
            objects=pharms,
            generator=generator
        )
    
    def set_generator(self, generator: PharmGenerator):
        self.generator = generator

    def add_mols(self, 
                 mols: List[Chem.Mol], 
                 confid: int | Literal["all"] = -1):
        assert not isinstance(self.feats, PharmProfile) # For later...
        pharms = [self.generator.generate(mol, confid=confid) for mol in mols]
        self.objects.extend(pharms)

    def featurize(self, profiler: PharmProfiler):
        profiles = [profiler.profile(pharm) for pharm in self.objects]
        self.feats = profiles

    def nearest(self, profile: PharmProfile | PharmProfileList, k=3):
        out = profile.screen(self.profiles)
        scs = out[-1]
        idx = [x[:k] for x in out[:-1]]
        return *idx[::-1], scs[:k]
    
    @property
    def profiles(self) -> List[PharmProfile | PharmProfileList]:
        return self.feats
    