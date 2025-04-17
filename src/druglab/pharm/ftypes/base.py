from __future__ import annotations
from typing import List, Dict, Union, Tuple, Set, Any, Callable, ClassVar
from dataclasses import dataclass, field
import yaml

import numpy as np

from rdkit import Chem

from .drawopts import DrawOptions

@dataclass
class BasePharmType:
    name: str
    subtype: str = None
    members: List[BasePharmType] = field(default_factory=list, repr=False)
    drawopts: DrawOptions = field(default_factory=DrawOptions, repr=False)
    adjopts: Dict[Dict] = field(default_factory=dict, repr=False)
    _default_subtype: ClassVar[str] = field(repr=False)

    def __post_init__(self):
        if self.subtype is None:
            self.subtype = self._default_subtype

        if self.drawopts is None:
            self.drawopts = DrawOptions()

        if self.adjopts is None:
            self.adjopts = dict()

        if self.members is None:
            self.members = list()

    @classmethod
    def from_members(cls, members: List[BasePharmType]) -> BasePharmType:
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, ftdef: dict) -> BasePharmType:
        return cls(
            name=ftdef["name"],
            subtype=ftdef.get("type", cls._default_subtype),
            drawopts=DrawOptions.from_dict(ftdef.get("drawopts", {})),
            adjopts=ftdef.get("adjopts", {}),
        )
    
    def membernames2name(self, membernames: List[str]) -> str:
        raise NotImplementedError()

class BasePharmTypes(list):
    def __init__(self, types: List[BasePharmType] = None):
        if types is None:
            types = []
        super().__init__(types)

    def typenames2onehot(self, names: List[str]) -> np.ndarray:
        onehot = np.zeros((len(names), len(self)), dtype=np.int8)
        for i, name in enumerate(names):
            if name in self.names:
                onehot[i, self.names.index(name)] = True
            else:
                print("Warning: Name not found in feature types...")
        return onehot
    
    def typenames2idx(self, names: List[str]) -> np.ndarray:
        return np.array([self.names.index(name) for name in names], dtype=int)
    
    def typename2type(self, name: str) -> BasePharmType:
        return self[self.names.index(name)]
    
    def subtypename2types(self, name: str) -> BasePharmTypes:
        return self.__class__([ty for ty in self.types if ty.subtype == name])
    
    def subtypename2idx(self, name: str) -> List[int]:
        return [i for i, ty in enumerate(self.types) if ty.subtype == name]
    
    def __add__(self, other: BasePharmTypes):
        combined_list = list(self) + [
            ty for ty in other.types
            if ty.name not in self.names
        ]
        return self.__class__(combined_list)

    
    @property
    def types(self) -> List[BasePharmType]: # for the sake of typing
        return self
    
    @property
    def names(self) -> List[str]:
        return [ty.name for ty in self.types]
    
    @property
    def subtypes(self) -> List[str]:
        return [ty.subtype for ty in self.types]
    
    @property
    def members(self) -> List[BasePharmTypes]:
        return [ty.members for ty in self.types]
    