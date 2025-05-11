from typing import List, Tuple

import numpy as np

class PharmBittifier:
    def __init__(self, maxbit: int = 1e9):
        self.maxbit = maxbit

    def bittify(self, 
                tys: np.ndarray,
                tyids: np.ndarray, 
                n_tyids: int, 
                vals: np.ndarray = None) -> Tuple[np.ndarray, int]:
        return tyids % self.maxbit, min(n_tyids, self.maxbit)
    
class PharmValBinBittifier(PharmBittifier):
    def __init__(self, 
                 bins: Tuple[float], 
                 maxbit: int = 1e9):
        self.bins = bins
        super().__init__(maxbit)
    
    def bittify(self, tys, tyids, n_tyids, vals = None):

        bits = np.digitize(vals, self.bins)
        l = len(self.bins) + 1
        
        indexer = np.ones(bits.shape[1]) * l
        indexer = indexer.cumprod().astype(np.uint64)
        indexer = np.flip(indexer)
        indexer[:-1] = indexer[1:]
        indexer[-1] = 1

        bits = (bits * indexer).sum(axis=-1)
        maxbit = min(l ** indexer.shape[0], self.maxbit)
        bits = bits % maxbit
        return bits, maxbit
        
class PharmCompositeBittifier(PharmBittifier):
    def __init__(self, bittifiers: List[PharmBittifier], maxbit: int = 1e9):
        self.bittifiers = bittifiers
        super().__init__(maxbit)
    
    def bittify(self,
                tys: np.ndarray,
                tyids: np.ndarray,
                n_tyids: int,
                vals: np.ndarray = None) -> np.ndarray:
        bits_list = []
        maxbit_list = []
        for bittifier in self.bittifiers:
            bits, maxbit = bittifier.bittify(tys, tyids, n_tyids, vals)
            bits = bits % self.maxbit
            maxbit = min(self.maxbit, maxbit)
            bits_list.append(bits)
            maxbit_list.append(maxbit)
        
        finalbits = np.zeros_like(bits_list[0])
        cmaxbit = 1
        for bits, maxbit in reversed(zip(bits_list, maxbit_list)):
            finalbits += cmaxbit * bits
            cmaxbit *= maxbit
        
        return finalbits % self.maxbit, min(cmaxbit, self.maxbit)