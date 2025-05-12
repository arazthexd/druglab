from typing import List

import numpy as np

from rdkit import Chem

from ..featurize import BaseFeaturizer
from .generator import PharmGenerator
from .profiler import PharmProfiler
from .fingerprint import PharmFingerprinter

class PharmFeaturizer(BaseFeaturizer):
    def __init__(self, 
                 generator: PharmGenerator, 
                 profiler: PharmProfiler,
                 fingerprinter: PharmFingerprinter):
        super().__init__()
        self.generator = generator
        self.profiler = profiler
        self.fingerprinter = fingerprinter
        self.confid_overwrite = None
    
    def featurize(self, object: Chem.Mol | Chem.Conformer) -> np.ndarray:
        if isinstance(object, Chem.Mol):
            if object.GetNumConformers() > 1:
                if self.confid_overwrite is not None:
                    confid = self.confid_overwrite
                else:
                    confid = "all"
            else:
                confid = 0
            return self._featurize_mol(object, confid=confid)
        elif isinstance(object, Chem.Conformer):
            return self._featurize_conformer(object)
    
    def _featurize_mol(self, mol: Chem.Mol, confid: int) -> np.ndarray:
        pharm = self.generator.generate(mol, confid=confid)
        profile = self.profiler.profile(pharm)
        fp = self.fingerprinter.fingerprint(profile, merge_confs=True)
        return fp
    
    def _featurize_conformer(self, conformer: Chem.Conformer) -> np.ndarray:
        mol = conformer.GetOwningMol()
        cid = conformer.GetId()
        return self._featurize_mol(mol, confid=cid)
