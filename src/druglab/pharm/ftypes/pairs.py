from __future__ import annotations
from typing import List, Dict, Union, Tuple, Set, Any, Callable
from dataclasses import dataclass, field
import yaml

import numpy as np

from rdkit import Chem

from .drawopts import DrawOptions
from .base import BasePharmType, BasePharmTypes
from .singles import PharmSingleTypes, PharmSingleType

class PharmPairType(BasePharmType):
    _default_subtype = "distance"

    @classmethod
    def from_members(cls, 
                     members: List[PharmSingleType], 
                     subtype: str = None,
                     drawopts: DrawOptions = None,
                     name: str = None):

        if name is None:
            name = f"{members[0].name}|{subtype}|{members[1].name}"
        
        return cls(
            name=name,
            subtype=subtype,
            members=PharmSingleTypes(members),
            drawopts=drawopts
        )

    @classmethod
    def from_dict(cls, ftdef):
        raise NotImplementedError()

    @classmethod
    def from_singles(cls, 
                     st1: PharmSingleType, 
                     st2: PharmSingleType,
                     name: str = None,
                     subtype: str = None,
                     drawopts: DrawOptions = None,
                     adjopts: Dict = None) -> PharmPairType:

        stypes = PharmSingleTypes([st1, st2])
        return cls(
            name=name or f"{st1.name}|{type}|{st2.name}",
            stypes=stypes,
            subtype=subtype,
            drawopts=drawopts,
            adjopts=adjopts
        )

class PharmPairTypes(BasePharmTypes):

    @property
    def memnames1(self) -> List[str]:
        return [ty.members[0].name for ty in self.types]
    
    @property
    def memnames2(self) -> List[str]:
        return [ty.members[1].name for ty in self.types]
    
    @property
    def distances(self) -> PharmPairTypes:
        return self.subtypename2types("distance")
    
    @property
    def distances_idx(self) -> List[int]:
        return self.subtypename2idx("distance")