import numpy as np
from typing import Optional, List, Tuple

from druglab.db.base import BaseTable
from druglab.pipe.archetypes import BaseFeaturizer

# ---------------------------------------------------------------------------
# Fingerprints
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
    
# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------
    
class RDKit2DFeaturizer(BaseFeaturizer):
    """
    Calculates the standard suite of RDKit 2D descriptors and returns them
    as a float32 feature vector.
 
    By default, all descriptors in ``rdkit.Chem.Descriptors.descList`` are
    computed (~200+ descriptors).  Pass a custom list of descriptor names via
    ``descriptor_names`` to restrict the output to a specific subset.
 
    Invalid molecules (None) receive a zero-filled vector of the correct
    length.  Individual descriptor errors are silenced and produce a NaN
    value for that position so the rest of the vector is still usable.
 
    Parameters
    ----------
    descriptor_names : list[str] or None
        Names of RDKit descriptors to compute.  When *None* (default) all
        descriptors from ``Descriptors.descList`` are used.
    """
 
    def __init__(self, descriptor_names: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.descriptor_names = descriptor_names
        self._desc_fns: Optional[List] = None  # populated lazily
 
    def _get_desc_fns(self):
        """Lazily build and cache the list of (name, fn) tuples."""
        if self._desc_fns is None:
            from rdkit.Chem import Descriptors
            if self.descriptor_names is None:
                self._desc_fns = [(name, fn) for name, fn in Descriptors.descList]
            else:
                desc_map = dict(Descriptors.descList)
                missing = [
                    name for name in self.descriptor_names if name not in desc_map
                ]
                if missing:
                    raise ValueError(
                        f"Unknown RDKit descriptor(s): {missing}"
                    )
                self._desc_fns = [
                    (name, desc_map[name]) for name in self.descriptor_names
                ]
        return self._desc_fns
 
    def get_config(self):
        config = super().get_config()
        config["descriptor_names"] = self.descriptor_names
        return config
 
    def _process_item(self, item):
        desc_fns = self._get_desc_fns()
        n = len(desc_fns)
 
        if item is None or item.GetNumAtoms() == 0:
            return np.zeros(n, dtype=np.float32)
 
        values = np.empty(n, dtype=np.float32)
        for i, (_, fn) in enumerate(desc_fns):
            try:
                values[i] = float(fn(item))
            except Exception:
                values[i] = np.nan
        return values
    
# ---------------------------------------------------------------------------
# Pharmacophore
# ---------------------------------------------------------------------------

class GobbiFeaturizer(BaseFeaturizer):
    """
    Computes Gobbi 2D pharmacophore fingerprints via RDKit's
    ``rdkit.Chem.Pharm2D`` module.

    The Gobbi fingerprint encodes spatial relationships between pairs of
    pharmacophoric features (hydrogen-bond donors/acceptors, aromatic rings,
    positively/negatively charged groups, and hydrophobic atoms) in a
    2D topological sense.  It is a useful complement to Morgan/MACCS when
    shape or interaction-pattern similarity is of interest.
 
    Parameters
    ----------
    n_bits : int or None
        Length of the output bit vector. If an integer is provided, the 
        vector is folded down to this length. If None, the full native 
        vector is returned unfolded. Defaults to 2048.
    fold_method : str
        The method used to fold the fingerprint. Options are "xor" or "or".
        Defaults to "xor".
    """
 
    def __init__(self, n_bits: Optional[int] = 2048, fold_method: str = "xor", **kwargs):
        super().__init__(**kwargs)
        self.n_bits = n_bits
        
        if fold_method.lower() not in ["xor", "or"]:
            raise ValueError("fold_method must be either 'xor' or 'or'.")
        self.fold_method = fold_method.lower()

        # Dynamically determine the native fingerprint length using a simple dummy molecule
        from rdkit import Chem
        dummy_mol = Chem.MolFromSmiles('CCCOCCCN')
        dummy_arr = self._generate_raw_fp(dummy_mol)
        self._native_len = len(dummy_arr)
        
        # Determine the final output length to use for zero-padding invalid items
        self._output_len = self.n_bits if self.n_bits is not None else self._native_len
 
    def get_config(self):
        config = super().get_config()
        config["n_bits"] = self.n_bits
        config["fold_method"] = self.fold_method
        return config

    def _generate_raw_fp(self, item) -> np.ndarray:
        """Helper method to generate the dense, unfolded native fingerprint."""
        from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
        
        factory = Gobbi_Pharm2D.factory
        fp = Generate.Gen2DFingerprint(item, factory)
        
        fp_len = fp.GetNumBits()
        arr_full = np.zeros(fp_len, dtype=np.int8)
        
        # RDKit's ConvertToNumpyArray does not support SparseBitVect.
        # We extract the indices of the active bits and use NumPy 
        # indexing to set them to 1.
        on_bits = list(fp.GetOnBits())
        if on_bits:
            arr_full[on_bits] = 1
        
        return arr_full
 
    def _process_item(self, item):
        if item is None:
            return np.zeros(self._output_len, dtype=np.int8)
 
        try:
            arr_full = self._generate_raw_fp(item)
            
            # If n_bits is None, return the raw unfolded fingerprint
            if self.n_bits is None:
                return arr_full
 
            # Compress (Fold)
            out = np.zeros(self.n_bits, dtype=np.int8)
            fp_len = len(arr_full)
            for i in range(0, fp_len, self.n_bits):
                chunk = arr_full[i : i + self.n_bits]
                
                # Apply the chosen folding method
                if self.fold_method == "or":
                    out[:len(chunk)] |= chunk
                else:
                    out[:len(chunk)] ^= chunk
                
            return out
        except Exception:
            return np.zeros(self._output_len, dtype=np.int8)