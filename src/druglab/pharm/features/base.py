from __future__ import annotations
from typing import List, Dict, Union, Tuple, Set, Any, Callable, Type
from dataclasses import dataclass, field
import yaml

import numpy as np

from rdkit import Chem

try:
    import nglview as nv
except:
    pass

from ..ftypes import BasePharmType, BasePharmTypes

def tyidx2subtyidx(arr: np.ndarray, possible):
    if arr.shape[0] == 0:
        return arr
    
    possible = np.asarray(possible)
    lookup = np.zeros(possible.max()+1, dtype=int)
    lookup[possible] = np.arange(len(possible))
    return lookup[arr]

class BasePharmFeatures:
    _feature_names = []

    def __init__(self, 
                 types: BasePharmTypes,
                 tyidx: np.ndarray = None):
        if tyidx is None:
            tyidx = np.zeros((0,), dtype=int)
        self.types: BasePharmTypes = types
        self.tyidx: np.ndarray = tyidx
        
    def add_features(self,
                     names: List[str],
                     **features: Dict[str, np.ndarray]):
        tyidx = self.types.typenames2idx(names)
        self.tyidx = np.append(self.tyidx, tyidx, axis=0)
        for attr in self._feature_names:
            setattr(self, attr, np.append(getattr(self, attr), 
                                          features[attr], axis=0))
            
    def draw(self, 
             view: nv.NGLWidget = None, 
             idx: int = None):
        raise NotImplementedError()
    
    def draw_all(self, view: nv.NGLWidget = None):
        for i in range(len(self)):
            self.draw(view, i)
            
    @property
    def fnames(self) -> List[str]:
        return [self.types.names[i] for i in self.tyidx]
    
    def __len__(self):
        return self.tyidx.size
    
    def __getitem__(self, idx):
        init_dict = {
            "types": self.types,
            "tyidx": self.tyidx[idx],
        }
        for attr in self._feature_names:
            init_dict[attr] = getattr(self, attr)[idx]
        
        return self.__class__(**init_dict)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)})"
    
    def _get_combined_types(self, other: BasePharmFeatures):
        if self.types.names == other.types.names:
            return self.types, other.tyidx
        
        types = self.types + other.types
        other_tyidx = types.typenames2idx(other.fnames)
        return types, other_tyidx
    
    def __add__(self, other: BasePharmFeatures):

        newtypes, other_tyidx = self._get_combined_types(other)
        newtypes: BasePharmTypes
        other_tyidx: np.ndarray

        init_dict = {
            "types": newtypes,
            "tyidx": np.concatenate([self.tyidx, other_tyidx], axis=0),
        }
        for attr in self._feature_names:
            init_dict[attr] = np.concatenate([getattr(self, attr), 
                                              getattr(other, attr)], axis=0)

        return self.__class__(**init_dict)
    
class BasePharmAIOFeatures:
    _feature_names: List[str] = []
    _group_names: List[str] = []
    _group_subtypes: List[str] = []
    _group_classes: List[Type[BasePharmFeatures]] = []
    _types_class: Type[BasePharmTypes] = BasePharmTypes

    def __init__(self,
                 types: BasePharmTypes = None,
                 tyidx: np.ndarray = None,
                 ignore: bool = False):

        if types is None:
            if not ignore:
                raise ValueError()
            types = self._types_class()
        
        if tyidx is None:
            tyidx = np.zeros((0,), dtype=int)
        
        for gname, gclass, gsub in zip(self._group_names, 
                                       self._group_classes, 
                                       self._group_subtypes):
            gtypes = types.subtypename2types(gsub)
            gtypes_idx = types.subtypename2idx(gsub)
            gtyidx = tyidx[np.isin(tyidx, gtypes_idx)]
            group = gclass(
                types=gtypes,
                tyidx=tyidx2subtyidx(gtyidx, possible=gtypes_idx)
            )
            setattr(self, gname, group)

    @classmethod
    def from_groups(cls, groups: List[BasePharmFeatures]):
        out = cls(ignore=True)
        for gi, group in enumerate(groups):
            setattr(out, cls._group_names[gi], group)
        return out

    def draw(self,
             view: nv.NGLWidget,
             idx: int = -1,
             group: str = None):
        if group is None:
            group = self._group_names[0]
            for gn in self._group_names[1:]:
                if idx < len(getattr(self, gn)):
                    break
                group = gn
                idx -= len(getattr(self, gn))
        group: BasePharmFeatures = getattr(self, group)
        group.draw(view, idx)

    def draw_all(self, view: nv.NGLWidget):
        for gname in self._group_names:
            getattr(self, gname).draw_all(view)

    def __len__(self):
        return sum([len(getattr(self, gname)) for gname in self._group_names])
    
    def __getitem__(self, idx):
        for gname in self._group_names:
            if idx < len(getattr(self, gname)):
                return getattr(self, gname)[idx]
            idx -= len(getattr(self, gname))
        raise IndexError()
    
    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)})"
    
    def __add__(self, other: BasePharmAIOFeatures):
        # types = self.types + [
        #     ty for ty in other.types.types
        #     if ty.name not in self.types.names
        # ]
        # types = self._types_class(types)
        # fnames = self.fnames + other.fnames
        # tyidx = types.typenames2idx(fnames)

        assert self._group_names == other._group_names # TODO: If not?
        
        groups = []
        for gname in self._group_names:
            self_group: BasePharmFeatures = getattr(self, gname)
            other_group: BasePharmFeatures = getattr(other, gname)
            group = self_group + other_group
            groups.append(group)
        
        return self.__class__.from_groups(groups)
        
    @property
    def fnames(self) -> List[str]:
        return [self.types.names[i] for i in self.tyidx]
    
    @property
    def types(self) -> BasePharmTypes:
        types = list()
        for gname in self._group_names:
            group: BasePharmFeatures = getattr(self, gname)
            types.extend(group.types.types)
        return self._types_class(types)
    
    @property
    def tyidx(self) -> np.ndarray:
        group: BasePharmFeatures = getattr(self, self._group_names[0])
        tyidx = group.tyidx
        prev_n_ty = len(group.types)
        for gname in self._group_names[1:]:
            group: BasePharmFeatures = getattr(self, gname)
            tyidx = np.append(tyidx, 
                              group.tyidx + prev_n_ty, 
                              axis=0)
            prev_n_ty += len(group.types)
        return tyidx
        
            


    

    

        