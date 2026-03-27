import numpy as np
from typing import Optional, List, Tuple

from druglab.db.base import BaseTable
from druglab.pipe.archetypes import BaseFeaturizer

# ---------------------------------------------------------------------------
# Featurizers
# ---------------------------------------------------------------------------

class MACCSFeaturizer(BaseFeaturizer):
    """Calculates the 166-bit MACCS structural keys (returned as 167-length array)."""
    
    def _process_item(self, item):
        if item is None:
            return np.zeros(167, dtype=np.int8)
            
        from rdkit.Chem import MACCSkeys
        from rdkit import DataStructs
        
        fp = MACCSkeys.GenMACCSKeys(item)
        arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    
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