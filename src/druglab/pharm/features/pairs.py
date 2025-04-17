from __future__ import annotations
from typing import List, Dict, Union, Tuple, Set, Any, Callable
from dataclasses import dataclass, field
import yaml

import numpy as np

from rdkit import Chem

try:
    import nglview as nv
except:
    pass

from ..ftypes import (
    PharmSingleType, PharmSingleTypes, 
    PharmPairType, PharmPairTypes,
    DrawOptions
)
from .base import BasePharmFeatures, BasePharmAIOFeatures

class BasePharmPairs(BasePharmFeatures):
    _feature_names = ["diff", "memidx"]

    def __init__(self, 
                 types: PharmPairTypes, 
                 tyidx: np.ndarray = None,
                 memidx: np.ndarray = None,
                 diff: np.ndarray = None):
        super().__init__(types=types, tyidx=tyidx)
        
        if diff is None:
            diff = np.empty(self.tyidx.size)
        self.diff: np.ndarray = diff

        if memidx is None:
            memidx = np.empty((self.tyidx.size, 2), dtype=int)
        memidx = memidx.reshape(-1, 2)
        self.memidx: np.ndarray = memidx

class PharmDistancePairs(BasePharmPairs):
    def __init__(self, 
                 types: PharmPairTypes, 
                 tyidx: np.ndarray = None,
                 memidx: np.ndarray = None,
                 diff: np.ndarray = None):
        assert all(ty.subtype == "distance" for ty in types.types)
        super().__init__(types=types, tyidx=tyidx, memidx=memidx, diff=diff)
    
    @property
    def dist(self) -> np.ndarray:
        return self.diff
    
