from __future__ import annotations
from typing import Dict, List, Callable, Any

import numpy as np

from rdkit import Chem

from .utilities import parse_varname

class PharmCalculation:
    def __init__(self,
                 inputs: List[str],
                 function: Callable,
                 outkeys: List[str]):
        self.inputs = inputs
        self.function = function
        self.outkeys = outkeys
    
    def __call__(self,
                 vars: Dict[str, Any], 
                 match: List[int],
                 mol: Chem.Mol, 
                 confid: int = -1) -> None:
        inps = [
            parse_varname(
                varname=inp, vars=vars, match=match,
                mol=mol, confid=confid) 
            for inp in self.inputs
        ]
        outs = self.function(*inps)
        assert len(outs) == len(self.outkeys)
        for outkey, out in zip(self.outkeys, outs):
            vars[outkey] = out

def direction(xstart, xend, normalize: bool = True):
    v = xend - xstart
    if normalize:
        v = v / np.linalg.norm(v)
    return v, 

def perpendicular(v1, v2, normalize: bool = True):
    perp = np.cross(v1, v2)
    if normalize:
        perp = perp / np.linalg.norm(perp)
    return perp,

def mean2(v1, v2, normalize: bool = True):
    vm = (v1 + v2) / 2
    if normalize:
        vm = vm / np.linalg.norm(vm)
    return vm, 

def pmean(*vs):
    return (sum(vs) / len(vs)), 

def norm(v):
    return (v / np.linalg.norm(v)), 

def tetrahedral3(xcenter, xnei1, xnei2, theta: float = None):
    v1 = xnei1 - xcenter
    v1 = v1 / np.linalg.norm(v1)

    v2 = xnei2 - xcenter
    v2 = v2 / np.linalg.norm(v2)

    u1 = perpendicular(v1, v2, normalize=True)[0]
    u2 = -mean2(v1, v2, normalize=True)[0]

    if theta is None:
        alpha = np.rad2deg(np.arccos(np.dot(v1, v2))) - 109.5
        alpha = alpha / (120 - 109.5)
        alpha = alpha * 109.5
        theta = 109.5 - alpha
    
    theta = theta / 2
    theta = np.deg2rad(theta)

    h1 = u2 * np.cos(theta) + u1 * np.sin(theta)
    h2 = u2 * np.cos(theta) - u1 * np.sin(theta)

    return h1, h2

def tetrahedral4(xcenter, xnei1, xnei2, xnei3):
    h = pmean(xcenter, xnei1, xnei2, xnei3)[0]
    h = h / np.linalg.norm(h)
    return h,

def eplane3(xadd, xnei, xnon, theta: float = 120.0):
    v1 = xnei - xadd
    v1 = v1 / np.linalg.norm(v1)

    v2 = xnon - xnei
    v2 = v2 - np.dot(v2, v1) * v1
    v2 = v2 / np.linalg.norm(v2)

    theta = np.deg2rad(theta)
    h1 = v1 * np.cos(theta) + v2 * np.cos(theta)
    h2 = v1 * np.cos(theta) - v2 * np.cos(theta)
    h1 = h1 / np.linalg.norm(h1)
    h2 = h2 / np.linalg.norm(h2)
    return h1, h2

def plane3(xcenter, xnei1, xnei2):
    v1 = xcenter - xnei1
    v2 = xcenter - xnei2
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    h = v1 + v2
    h = h / np.linalg.norm(h)
    return h, 