from __future__ import annotations
from typing import List, Dict, Union, Tuple, Set, Any, Callable
from dataclasses import dataclass, field
import yaml

import numpy as np

from rdkit import Chem

from .drawopts import DrawOptions
from .base import BasePharmType, BasePharmTypes
            
class PharmSingleType(BasePharmType):
    _default_subtype = "arrow"

class PharmSingleTypes(BasePharmTypes):

    @property
    def arrows(self) -> PharmSingleTypes:
        return self.subtypename2types("arrow")
    
    @property
    def arrows_idx(self) -> List[int]:
        return self.subtypename2idx("arrow")
    
    @property
    def spheres(self) -> PharmSingleTypes:
        return self.subtypename2types("sphere")
    
    @property
    def spheres_idx(self) -> List[int]:
        return self.subtypename2idx("sphere")
    