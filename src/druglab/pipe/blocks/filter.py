import numpy as np
from typing import Optional, List, Tuple

from druglab.db.base import BaseTable
from druglab.pipe.archetypes import BaseFilter

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

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

class PropertyFilter(BaseFilter):
    """Filters molecules based on standard molecular descriptors."""
    
    def __init__(self, min_mw: Optional[float] = None, max_mw: Optional[float] = None, 
                 min_logp: Optional[float] = None, max_logp: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.min_mw = min_mw
        self.max_mw = max_mw
        self.min_logp = min_logp
        self.max_logp = max_logp

    def get_config(self):
        config = super().get_config()
        config.update({
            "min_mw": self.min_mw, "max_mw": self.max_mw,
            "min_logp": self.min_logp, "max_logp": self.max_logp
        })
        return config

    def _process_item(self, item):
        if item is None:
            return False
        from rdkit.Chem import Descriptors
        
        if self.min_mw is not None and Descriptors.MolWt(item) < self.min_mw: return False
        if self.max_mw is not None and Descriptors.MolWt(item) > self.max_mw: return False
        if self.min_logp is not None and Descriptors.MolLogP(item) < self.min_logp: return False
        if self.max_logp is not None and Descriptors.MolLogP(item) > self.max_logp: return False
        
        return True

class SMARTSFilter(BaseFilter):
    """Filters molecules based on the presence (or absence) of a SMARTS pattern."""
    
    def __init__(self, smarts: str, exclude: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.smarts = smarts
        self.exclude = exclude

    def get_config(self):
        config = super().get_config()
        config.update({"smarts": self.smarts, "exclude": self.exclude})
        return config

    def _process_item(self, item):
        if item is None:
            return False
        from rdkit import Chem
        patt = Chem.MolFromSmarts(self.smarts)
        if patt is None:
            return False
            
        has_match = item.HasSubstructMatch(patt)
        return not has_match if self.exclude else has_match

class ElementFilter(BaseFilter):
    """Filters out molecules containing elements outside of the allowed list."""
    
    def __init__(self, allowed_elements: Tuple[int, ...] = (6, 1, 7, 8, 9, 15, 16, 17, 35, 53), **kwargs):
        super().__init__(**kwargs)
        self.allowed_elements = set(allowed_elements)

    def _process_item(self, item):
        if item is None:
            return False
        for atom in item.GetAtoms():
            if atom.GetAtomicNum() not in self.allowed_elements:
                return False
        return True

class ValidityFilter(BaseFilter):
    """Drops any rows where the parsed object is None or contains 0 atoms."""
    
    def _process_item(self, item):
        if item is None:
            return False
        try:
            return item.GetNumAtoms() > 0
        except Exception:
            return False