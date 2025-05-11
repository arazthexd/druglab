from __future__ import annotations
from typing import List
from dataclasses import dataclass
import itertools

import numpy as np

from .pharmacophore import Pharmacophore, PharmacophoreList

@dataclass(repr=False)
class PharmProfile:
    pharm: Pharmacophore | PharmacophoreList = None
    subborderids: List[int] = None

    tys: np.ndarray = None
    tyids: np.ndarray = None
    n_tyids: int = None

    pair1tys: np.ndarray = None
    pair1tyids: np.ndarray = None
    pair1vals: np.ndarray = None
    n_pair1tyids: int = None

    pair2tys: np.ndarray = None
    pair2tyids: np.ndarray = None
    pair2vals: np.ndarray = None
    n_pair2tyids: int = None

    def __post_init__(self):
        assert self.subborderids is not None or self.tyids is not None
        if self.subborderids is not None and self.tyids is not None:
            assert self.subborderids[-1] == self.tyids.shape[0]
        
        if self.subborderids is not None:
            ntotal = self.subborderids[-1]
        else:
            ntotal = self.tyids.shape[0]
        
        if self.tys is None:
            self.tys = np.zeros((ntotal, 0))
        
        if self.tyids is None:
            self.tyids = np.zeros((ntotal, ))
        
        if self.n_tyids is None:
            if self.tyids is not None:
                raise ValueError("tyids is defined without defining its n")
            self.n_tyids = 1
        
        if self.pair1tys is None:
            self.pair1tys = np.zeros((ntotal, 0, 2))
        
        if self.pair1tyids is None:
            self.pair1tyids = np.zeros((ntotal, 0))
        
        if self.pair1vals is None:
            self.pair1vals = np.zeros((ntotal, 0))
        
        if self.n_pair1tyids is None:
            if self.pair1tyids is not None:
                raise ValueError("pair1tyids is defined without defining its n")
            self.n_pair1tyids = 0
        
        if self.pair2tys is None:
            self.pair2tys = np.zeros((ntotal, 0, 2))
        
        if self.pair2tyids is None:
            self.pair2tyids = np.zeros((ntotal, 0))
        
        if self.pair2vals is None:
            self.pair2vals = np.zeros((ntotal, 0))
        
        if self.n_pair2tyids is None:
            if self.pair2tyids is not None:
                raise ValueError("pair2tyids is defined without defining its n")
            self.n_pair2tyids = 0

    @classmethod
    def merge(cls, profiles: List[PharmProfile]):
        subborderids_list = [prof.subborderids.copy() for prof in profiles]
        addid = [sbi[-1] for sbi in subborderids_list]
        addid = [0] + list(itertools.accumulate(addid))[:-1]
        for i, toadd in enumerate(addid):
            subborderids_list[i] = [sbi+toadd for sbi in subborderids_list[i]]

        assert all(prof.n_tyids == profiles[0].n_tyids 
                   for prof in profiles)
        assert all(prof.n_pair1tyids == profiles[0].n_pair1tyids 
                   for prof in profiles)
        assert all(prof.n_pair2tyids == profiles[0].n_pair2tyids 
                   for prof in profiles)

        return cls(
            subborderids=sum(subborderids_list, []),
            tys=np.concatenate([prof.tys for prof in profiles]),
            tyids=np.concatenate([prof.tyids for prof in profiles]),
            n_tyids=profiles[0].n_tyids,
            pair1tys=np.concatenate([prof.pair1tys for prof in profiles]),
            pair1tyids=np.concatenate([prof.pair1tyids for prof in profiles]),
            pair1vals=np.concatenate([prof.pair1vals for prof in profiles]),
            n_pair1tyids=profiles[0].n_pair1tyids,
            pair2tys=np.concatenate([prof.pair2tys for prof in profiles]),
            pair2tyids=np.concatenate([prof.pair2tyids for prof in profiles]),
            pair2vals=np.concatenate([prof.pair2vals for prof in profiles]),
            n_pair2tyids=profiles[0].n_pair2tyids
        )
            

            
# TODO: Managing "None"s and replacing them with empty numpy arrays