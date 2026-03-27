import numpy as np
from typing import Optional, List, Tuple

from druglab.db.base import BaseTable
from druglab.pipe.archetypes import BasePreparation

# ---------------------------------------------------------------------------
# Preparations (Modifiers)
# ---------------------------------------------------------------------------

class MoleculeKekulizer(BasePreparation):
    """Standardizer that Kekulizes the RDKit Mol."""
    
    def _process_item(self, item):
        from rdkit import Chem
        if item is not None:
            item = Chem.Mol(item)
            Chem.Kekulize(item, clearAromaticFlags=True)
        return item

class MoleculeDesalter(BasePreparation):
    """Removes counterions and solvents, keeping only the largest organic fragment."""
    
    def _process_item(self, item):
        if item is None:
            return None
        from rdkit.Chem.MolStandardize import rdMolStandardize
        try:
            chooser = rdMolStandardize.LargestFragmentChooser()
            return chooser.choose(item)
        except Exception:
            return item

class TautomerCanonicalizer(BasePreparation):
    """Enumerates tautomers and returns a standardized, canonical tautomer."""
    
    def _process_item(self, item):
        if item is None:
            return None
        from rdkit.Chem.MolStandardize import rdMolStandardize
        try:
            enumerator = rdMolStandardize.TautomerEnumerator()
            return enumerator.Canonicalize(item)
        except Exception:
            return item

class HydrogenModifier(BasePreparation):
    """Adds or removes explicit hydrogens from the molecule."""
    
    def __init__(self, add_hs: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.add_hs = add_hs

    def get_config(self):
        config = super().get_config()
        config["add_hs"] = self.add_hs
        return config

    def _process_item(self, item):
        if item is None:
            return None
        from rdkit import Chem
        try:
            return Chem.AddHs(item) if self.add_hs else Chem.RemoveHs(item)
        except Exception:
            return item

class MoleculeSanitizer(BasePreparation):
    """Forces RDKit sanitization. Returns None if the molecule is structurally invalid."""
    
    def _process_item(self, item):
        if item is None:
            return None
        from rdkit import Chem
        try:
            # Create a copy to avoid mutating the original if it fails mid-sanitization
            mol_copy = Chem.Mol(item) 
            Chem.SanitizeMol(mol_copy)
            return mol_copy
        except Exception:
            return None

