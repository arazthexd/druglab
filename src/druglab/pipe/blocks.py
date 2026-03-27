"""
druglab.pipe.blocks
~~~~~~~~~~~~~~~~~~~
Example concrete implementations of pipeline blocks.
"""

import numpy as np
from typing import Optional

from druglab.db.base import BaseTable
from druglab.pipe.archetypes import BaseFeaturizer, BaseFilter, BasePreparation, IOBlock


class MorganFeaturizer(BaseFeaturizer):
    """Calculates Morgan Fingerprints as bit vectors (numpy arrays)."""
    
    def __init__(self, radius: int = 2, n_bits: int = 2048, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius
        self.n_bits = n_bits

    def get_config(self):
        """Include the specific mathematical parameters in the config."""
        config = super().get_config()
        config.update({
            "radius": self.radius,
            "n_bits": self.n_bits
        })
        return config
        
    def _process_item(self, item):
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
        
        if item is None:
            return np.zeros(self.n_bits, dtype=np.int8)
            
        fp = AllChem.GetMorganFingerprintAsBitVect(item, self.radius, nBits=self.n_bits)
        arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr


class MWFilter(BaseFilter):
    """Filters molecules keeping only those strictly under a specified Molecular Weight."""
    
    def __init__(self, max_mw: float = 500.0, **kwargs):
        super().__init__(**kwargs)
        self.max_mw = max_mw
        
    def _process_item(self, item):
        from rdkit.Chem import Descriptors
        if item is None:
            return False
        return Descriptors.MolWt(item) <= self.max_mw


class KekulizePreparation(BasePreparation):
    """In-place standardizer that Kekulizes the RDKit Mol."""
    
    def _process_item(self, item):
        from rdkit import Chem
        if item is not None:
            Chem.Kekulize(item, clearAromaticFlags=True)
        return item


class MemoryIOBlock(IOBlock):
    """
    A testing/utility block that doesn't read from disk, but yields chunks 
    from an already loaded BaseTable. Perfect for triggering pipeline batch mode.
    """
    
    def __init__(self, table: BaseTable, batch_size: int = 1000, **kwargs):
        super().__init__(batch_size=batch_size, **kwargs)
        self.table = table
        
    def yield_batches(self):
        n = len(self.table)
        for i in range(0, n, self.batch_size):
            yield self.table[i : i + self.batch_size]
            
    def _load_table(self):
        return self.table