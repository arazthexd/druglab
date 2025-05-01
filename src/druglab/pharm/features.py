from typing import List, Tuple

import numpy as np

from .drawopts import DrawOptions

class PharmFeatures:
    def __init__(self):
        self.pos: np.ndarray = np.zeros((0, 3))
        self.origin_atidx: List[Tuple[int]] = []

    def add_features(self, pos: np.ndarray, atidx: List[Tuple[int]] = None):
        if atidx is None:
            atidx = [None] * pos.shape[0]
        pos = pos.reshape(-1, 3)
        assert pos.shape[0] == len(atidx)

        self.pos = np.append(self.pos, pos, axis=0)
        self.origin_atidx.extend(atidx)

    def tuple(self) -> tuple:
        return (self.pos, )
    
    def draw(self, view, drawopts: DrawOptions):
        pass

class PharmArrowFeats(PharmFeatures):
    def __init__(self):
        super().__init__()
        self.vec: np.ndarray = np.zeros((0, 3))
        self.radius: np.ndarray = np.zeros((0, ))

    def add_features(self, 
                     pos: np.ndarray, 
                     vec: np.ndarray, 
                     radius: np.ndarray | float = 0.0,
                     atidx: List[Tuple[int]] = None):
        
        super().add_features(pos, atidx)
        vec = vec.reshape(-1, 3)
        self.vec = np.append(self.vec, vec, axis=0)

        if isinstance(radius, float):
            radius = np.array([radius]*vec.shape[0])

        assert radius.shape[0] == vec.shape[0]
        assert radius.ndim == 1
        self.radius = np.append(self.radius, radius, axis=0)

    def tuple(self):
        tup = super().tuple()
        return (*tup, self.vec, self.radius)
    
    def draw(self, view, drawopts):
        super().draw(view, drawopts)
        for idx in range(self.pos.shape[0]):
            radius = max(drawopts.radius, self.radius[idx]/2)
            length = drawopts.length + radius
            view.shape.add_arrow(
                self.pos[idx],
                self.pos[idx] + self.vec[idx] * length,
                drawopts.color,
                drawopts.radius,
            )
            view.shape.add_sphere(
                self.pos[idx],
                drawopts.color,
                radius,
            )

    def remove_idx(self, idx: int | List[int]):
        if isinstance(idx, int):
            idx = [idx]
        idx = list(idx)
        
        keepidx = [i for i in range(self.pos.shape[0]) if i not in idx]
        self.pos = self.pos[keepidx]
        self.vec = self.vec[keepidx]
        self.radius = self.radius[keepidx]
        self.origin_atidx = [self.origin_atidx[i] for i in keepidx]

class PharmSphereFeats(PharmFeatures):
    def __init__(self):
        super().__init__()
        self.radius: np.ndarray = np.zeros((0, ))
    
    def add_features(self, 
                     pos: np.ndarray, 
                     radius: np.ndarray | float = 0.0,
                     atidx: List[Tuple[int]] = None):
        
        super().add_features(pos, atidx)

        if isinstance(radius, float):
            radius = np.array([radius]*pos.shape[0])

        assert radius.shape[0] == pos.shape[0]
        assert radius.ndim == 1
        self.radius = np.append(self.radius, radius, axis=0)

    def tuple(self):
        tup = super().tuple()
        return (*tup, self.radius)
    
    def draw(self, view, drawopts):
        super().draw(view, drawopts)
        for idx in range(self.pos.shape[0]):
            radius = max(drawopts.radius, self.radius[idx]/2)
            view.shape.add_sphere(
                self.pos[idx],
                drawopts.color,
                radius,
            )
