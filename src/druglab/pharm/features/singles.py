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

from ..ftypes import PharmSingleType, PharmSingleTypes, DrawOptions
from .base import BasePharmFeatures, BasePharmAIOFeatures

class BasePharmSingles(BasePharmFeatures):
    _feature_names = ["pos"]
    
    def __init__(self, 
                 types: PharmSingleTypes, 
                 tyidx: np.ndarray = None,
                 pos: np.ndarray = None):
        super().__init__(types=types, tyidx=tyidx)
        
        if pos is None:
            pos = np.zeros((0, 3))
        self.pos: np.ndarray = pos

    def draw(self, view: nv.NGLWidget, idx: int):
        drawopts: DrawOptions = self.types.types[self.tyidx[idx].item()].drawopts
        view.shape.add_sphere(
            self.pos[idx],
            drawopts.color,
            drawopts.radius
        )
    
class PharmSphereSingles(BasePharmSingles):
    _feature_names = BasePharmSingles._feature_names + ["radius"]

    def __init__(self,
                 types: PharmSingleTypes,
                 tyidx: np.ndarray = None,
                 pos: np.ndarray = None,
                 radius: np.ndarray = None):
        assert all(ty.subtype == "sphere" for ty in types.types)
        super().__init__(types=types, tyidx=tyidx, pos=pos)

        if radius is None:
            radius = np.zeros(self.tyidx.size)
        self.radius: np.ndarray = radius

    def draw(self, view: nv.NGLWidget, idx: int):
        drawopts: DrawOptions = self.types.types[self.tyidx[idx].item()].drawopts
        radius = max(self.radius[idx], drawopts.radius)
        view.shape.add_sphere(
            self.pos[idx],
            drawopts.color,
            radius,
        )

class PharmArrowSingles(BasePharmSingles):
    _feature_names = BasePharmSingles._feature_names + ["vec"]

    def __init__(self,
                 types: PharmSingleTypes,
                 tyidx: np.ndarray = None,
                 pos: np.ndarray = None,
                 vec: np.ndarray = None):
        assert all(ty.subtype == "arrow" for ty in types.types)
        super().__init__(types=types, tyidx=tyidx, pos=pos)

        if vec is None:
            vec = np.zeros((0, 3))
        self.vec: np.ndarray = vec

    def draw(self, view: nv.NGLWidget, idx: int):
        drawopts: DrawOptions = self.types.types[self.tyidx[idx].item()].drawopts
        view.shape.add_arrow(
            self.pos[idx],
            self.pos[idx] + self.vec[idx] * drawopts.length,
            drawopts.color,
            drawopts.radius,
        )
    
class PharmAIOSingles(BasePharmAIOFeatures): # All-In-One
    _feature_names = ["pos", "vec", "radius"]
    _group_names = ["arrows", "spheres"]
    _group_subtypes = ["arrow", "sphere"]
    _group_classes = [PharmArrowSingles, PharmSphereSingles]
    _types_class = PharmSingleTypes

    def __init__(self,
                 types: PharmSingleTypes = None,
                 tyidx: np.ndarray = None,
                 pos: np.ndarray = None,
                 vec: np.ndarray = None,
                 radius: np.ndarray = None,
                 ignore: bool = False,
                 debug: bool = False):
        
        self.debug = debug
        
        if pos is None:
            pos = np.zeros((0, 3))
        if vec is None:
            vec = np.zeros((0, 3))
        if radius is None:
            radius = np.zeros(0)
        
        super().__init__(types=types, tyidx=tyidx, ignore=ignore)
        self.arrows: PharmArrowSingles
        self.spheres: PharmSphereSingles
        self.arrows.vec = vec
        self.spheres.radius = radius

        if self.debug:
            print("=== inside pharmaiosingle init ===")
            print("tyidx", tyidx)
            print("self.tyidx", self.tyidx)
            print("arrow type idx", self.types.subtypename2idx("arrow"))
            print("sphere type idx", self.types.subtypename2idx("sphere"))
            print("arrow pos", pos[np.isin(
                self.tyidx, self.types.subtypename2idx("arrow")
            )])
            print("sphere pos", pos[np.isin(
                self.tyidx, self.types.subtypename2idx("sphere")
            )])
            print()

        self.arrows.pos = pos[np.isin(self.tyidx, 
                                      self.types.subtypename2idx("arrow"))]
        self.spheres.pos = pos[np.isin(self.tyidx, 
                                       self.types.subtypename2idx("sphere"))]
    
    @property
    def pos(self) -> np.ndarray:
        return np.append(self.arrows.pos, self.spheres.pos, axis=0)
    
    @property
    def vec(self) -> np.ndarray:
        return self.arrows.vec
    
    @property
    def radius(self) -> np.ndarray:
        return self.spheres.radius

