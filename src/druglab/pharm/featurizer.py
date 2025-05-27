from typing import List, Tuple

import numpy as np

from rdkit import Chem

from ..featurize import BaseFeaturizer
from .generator import PharmGenerator, BASE_DEFINITIONS_PATH
from .profiler import PharmProfiler
from .fingerprint import PharmFingerprinter
from .adjusters import PharmAdjuster

class PharmFeaturizer(BaseFeaturizer):
    def __init__(self, 
                 generator: PharmGenerator = None, 
                 adjuster: PharmAdjuster = None,
                 profiler: PharmProfiler = None,
                 fingerprinter: PharmFingerprinter = None):
        if generator is None:
            generator = PharmGenerator()
            generator.load_file(BASE_DEFINITIONS_PATH)
        if adjuster is None:
            adjuster = PharmAdjuster()
        if profiler is None:
            profiler = PharmProfiler(generator.ftypes)
        if fingerprinter is None:
            fingerprinter = PharmFingerprinter(fpsize=7000)

        super().__init__()
        self.generator = generator
        self.adjuster = adjuster
        self.profiler = profiler
        self.fingerprinter = fingerprinter
        self.confid_overwrite = None

        self._fnames = [
            f"pharm_{i}" for i in range(self.fingerprinter.fpsize)
        ]
    
    def featurize(self, object: Chem.Mol | Tuple[Chem.Mol, int]) -> np.ndarray:
        if isinstance(object, Chem.Mol):
            if object.GetNumConformers() > 1:
                if self.confid_overwrite is not None:
                    confid = self.confid_overwrite
                else:
                    confid = "all"
            else:
                confid = 0
            return self._featurize_mol(object, confid=confid)
        elif isinstance(object, tuple):
            mol, confid = object
            return self._featurize_mol(mol, confid=confid)
    
    def _featurize_mol(self, mol: Chem.Mol, confid: int) -> np.ndarray:
        pharm = self.generator.generate(mol, confid=confid)
        self.adjuster.adjust(pharm)
        profile = self.profiler.profile(pharm)
        fp = self.fingerprinter.fingerprint(profile, merge_confs=True)
        return fp