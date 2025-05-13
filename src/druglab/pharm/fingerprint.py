from typing import List, Tuple, Literal
import itertools

import numpy as np

from druglab.pharm import PharmProfile

def _get_indexer(maxbits: np.ndarray) -> np.ndarray:
    indexer = np.flip(np.flip(maxbits).cumprod())
    indexer[:-1] = indexer[1:]
    indexer[-1] = 1
    return indexer

class PharmFingerprinter:
    def __init__(self,
                 fpsize: int = 1024):
        self.fpsize = fpsize

    def bits(self,
             profile: PharmProfile) -> Tuple[np.ndarray, float]:
        raise NotImplementedError()

    def fingerprint(self,
                    profile: PharmProfile,
                    merge_confs: bool = False) -> np.ndarray:
        
        bits, maxbit = self.bits(profile)
        bits = bits % self.fpsize
        
        if merge_confs:
            fp = np.zeros((1, self.fpsize), dtype=np.int8)
            if bits.ndim == 2:
                bits = bits.flatten()
            fp[0, bits] = 1

        else:
            fp = np.zeros((len(profile.subids), self.fpsize), dtype=np.int8)
            for i, subids in enumerate(profile.subids):
                subids: List[int]
                fp[i, bits[subids].flatten()] = 1

        return fp
    
class PharmCompositeFingerprinter(PharmFingerprinter):
    def __init__(self, 
                 fpers: List[PharmFingerprinter],
                 mode: Literal["sum", "prod"] = "prod",
                 fpsize: int = 1024):
        super().__init__(fpsize)
        self.fpers = fpers
        self.mode = mode
    
    def bits(self, 
             profile: PharmProfile) -> Tuple[np.ndarray, float]:
        
        bits_list, maxbit_list = \
            tuple(zip(*[fper.bits(profile) for fper in self.fpers]))
        
        if self.mode == "sum":
            c = 0
            for bits, maxbit in zip(bits_list, maxbit_list):
                bits += c
                c += maxbit
            return np.concatenate(bits_list, axis=-1), c
        
        bit_stack_ls = [bits.shape[1] for bits in bits_list]

        maxbits = np.array(maxbit_list)
        indexer = _get_indexer(maxbits)

        bits = []
        for idcombo in itertools.product(*[range(i) for i in bit_stack_ls]):
            b = [bm[:, idcombo[i]] for i, bm in enumerate(bits_list)]
            b = np.stack(b, axis=-1)
            b = np.sum(b * indexer, axis=-1, keepdims=True)
            bits.append(b)
        
        bits = np.concatenate(bits, axis=-1)

        return (bits, np.prod(maxbits))
    
class PharmTypeIDFingerprinter(PharmFingerprinter):
    def __init__(self, 
                 fpsize: int = 1024):
        super().__init__(fpsize)
    
    def bits(self,
             profile: PharmProfile) -> Tuple[np.ndarray, float]:
        tyids = profile.tyids
        return (
            tyids.reshape(-1, 1),
            profile.n_tyids
        )
    
class PharmDistFingerprinter(PharmFingerprinter):
    def __init__(self, 
                 bins: Tuple[int] = (1, 2, 3, 4, 5, 6, 7, 8), 
                 fpsize: int = 1024):
        super().__init__(fpsize)
        self.bins = bins
    
    def bits(self, 
             profile: PharmProfile) -> Tuple[np.ndarray, float]:
        dists = profile.dists
        distids = np.digitize(dists, self.bins)
        maxbits = np.array([len(self.bins)+1] * dists.shape[1])
        indexer = _get_indexer(maxbits)

        return (
            np.sum(distids * indexer, axis=-1, keepdims=True), 
            np.prod(maxbits)
        )
    
class PharmCosineFingerprinter(PharmFingerprinter):
    def __init__(self, 
                 bins: Tuple[float] = (-0.8, -0.6, -0.4, -0.2, 
                                       0, 0.2, 0.4, 0.6, 0.8, 1.0), 
                 nanval: float = 1.5,
                 fpsize: int = 1024):
        super().__init__(fpsize)
        self.bins = bins
        self.nanval = nanval
    
    def bits(self, 
             profile: PharmProfile) -> Tuple[np.ndarray, float]:
        cos = np.nan_to_num(profile.cos[:, :, 0], 
                            posinf=self.nanval, 
                            nan=self.nanval)
        cosids = np.digitize(cos, self.bins)
        maxbits = np.array([len(self.bins)+1] * cos.shape[1])
        indexer = _get_indexer(maxbits)

        return (
            np.sum(cosids * indexer, axis=-1, keepdims=True), 
            np.prod(maxbits)
        )