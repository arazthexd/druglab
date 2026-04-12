import numpy as np
from typing import Optional, List, Tuple, Set

from druglab.db.base import BaseTable
from druglab.pipe.archetypes import BaseFilter

# ---------------------------------------------------------------------------
# Simple Filters
# ---------------------------------------------------------------------------

class MWFilter(BaseFilter):
    """Filters molecules keeping only those at or below a specified Molecular Weight."""
    
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
        
        if self.min_mw is not None and Descriptors.MolWt(item) < self.min_mw:
            return False
        if self.max_mw is not None and Descriptors.MolWt(item) > self.max_mw:
            return False
        if self.min_logp is not None and Descriptors.MolLogP(item) < self.min_logp:
            return False
        if self.max_logp is not None and Descriptors.MolLogP(item) > self.max_logp:
            return False
        
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
        
class UniqueFilter(BaseFilter):
    """
    Stateful filter that drops duplicate molecules across the pipeline run.
 
    Uniqueness is determined by canonical SMILES (default) or InChIKey.
    The internal seen-set is shared across all calls to ``run()`` on the
    **same block instance**, so duplicates are detected even when a pipeline
    runs in batch mode.  Call :meth:`reset` between independent runs if you
    want a clean slate.

    NOTE: this block is stateful and the state is shared across all calls to
    ``run()`` on the **same block instance**. As a result, multiprocessing
    is not supported.
 
    Parameters
    ----------
    key : {"smiles", "inchikey"}
        Hashing strategy.  ``"smiles"`` (default) uses RDKit canonical SMILES
        which is fast.  ``"inchikey"`` is more robust to different input
        representations at the cost of extra computation.
    """
 
    def __init__(self, key: str = "smiles", **kwargs):
        super().__init__(**kwargs)
        if key not in ("smiles", "inchikey"):
            raise ValueError("key must be 'smiles' or 'inchikey'")
        self.key = key
        self._seen: Set[str] = set()

        if self.n_workers > 1:
            print("WARNING: UniqueFilter does not support multiprocessing. Setting n_workers=1.")
            self.n_workers = 1
 
    def get_config(self):
        config = super().get_config()
        config["key"] = self.key
        return config
 
    def reset(self) -> None:
        """Clear the internal seen-set so the filter starts fresh."""
        self._seen.clear()
 
    def _get_key(self, item) -> Optional[str]:
        try:
            from rdkit.Chem import MolToSmiles
            from rdkit.Chem.inchi import MolToInchiKey
 
            if self.key == "smiles":
                return MolToSmiles(item)
            else:
                return MolToInchiKey(item)
        except Exception:
            return None
 
    def _process_item(self, item):
        if item is None:
            return False
 
        k = self._get_key(item)
        if k is None:
            return False  # can't fingerprint → treat as invalid
 
        if k in self._seen:
            return False
 
        self._seen.add(k)
        return True
        
# ---------------------------------------------------------------------------
# Drug Likeliness Filters
# ---------------------------------------------------------------------------

class RuleOfFiveFilter(BaseFilter):
    """
    Convenience filter enforcing Lipinski's Rule of Five (Ro5).
 
    A molecule passes if it violates **at most** ``max_violations`` of the
    four classic Lipinski criteria:
 
    1. Molecular weight ≤ 500 Da
    2. LogP ≤ 5
    3. Hydrogen-bond donors ≤ 5
    4. Hydrogen-bond acceptors ≤ 10
 
    Parameters
    ----------
    max_violations : int
        Maximum number of rule violations allowed (default 0 → strict Ro5).
        Set to 1 to allow the common "one-violation" relaxation used in
        fragment-based and natural-product drug discovery.
    """
 
    def __init__(self, max_violations: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.max_violations = max_violations
 
    def get_config(self):
        config = super().get_config()
        config["max_violations"] = self.max_violations
        return config
 
    def _process_item(self, item):
        if item is None or item.GetNumAtoms() == 0:
            return False
 
        from rdkit.Chem import rdMolDescriptors, Descriptors
 
        violations = 0
        if rdMolDescriptors.CalcExactMolWt(item) > 500:
            violations += 1
        if Descriptors.MolLogP(item) > 5:
            violations += 1
        if rdMolDescriptors.CalcNumHBD(item) > 5:
            violations += 1
        if rdMolDescriptors.CalcNumHBA(item) > 10:
            violations += 1
 
        return violations <= self.max_violations
    
class CatalogFilter(BaseFilter):
    """
    Filters molecules against RDKit's built-in structural alert catalogs.
 
    A molecule is **dropped** (returns False) when it matches one or more
    entries in any of the requested catalogs.  Use ``exclude=False`` to
    *keep* only the flagged molecules (e.g. for auditing).
 
    Supported catalog names (case-insensitive):
 
    * ``"PAINS"`` - Pan-assay interference compounds (Baell & Holloway 2010)
    * ``"PAINS_A"`` / ``"PAINS_B"`` / ``"PAINS_C"`` - individual PAINS subsets
    * ``"BRENK"`` - Brenk et al. (2008) unwanted fragments
    * ``"NIH"`` - NIH structural alert catalog
    * ``"ZINC"`` - ZINC drug-like filter
    * ``"CHEMBL23_DUNDEE"`` / ``"CHEMBL23_BMS"`` / ``"CHEMBL23_GLAXO"``
      / ``"CHEMBL23_INPHARMATICA"`` / ``"CHEMBL23_LINT"``
      / ``"CHEMBL23_MLSMR"`` / ``"CHEMBL23_SureChEMBL"``
    * ``"CHEMBL_Ro3"`` - ChEMBL Rule of Three
    * ``"CHEMBL_Ro5"`` - ChEMBL Rule of Five
 
    Parameters
    ----------
    catalogs : list[str]
        One or more catalog names.  Defaults to ``["PAINS"]``.
    exclude : bool
        When *True* (default), molecules that **match** the catalog are
        dropped.  When *False*, only matched molecules are kept.
    """
 
    # Map human-friendly names to RDKit's FilterCatalogParams enum values.
    _CATALOG_MAP = {
        "PAINS":                   "PAINS",
        "PAINS_A":                 "PAINS_A",
        "PAINS_B":                 "PAINS_B",
        "PAINS_C":                 "PAINS_C",
        "BRENK":                   "BRENK",
        "NIH":                     "NIH",
        "ZINC":                    "ZINC",
        "CHEMBL23_DUNDEE":         "CHEMBL23_Dundee",
        "CHEMBL23_BMS":            "CHEMBL23_BMS",
        "CHEMBL23_GLAXO":          "CHEMBL23_Glaxo",
        "CHEMBL23_INPHARMATICA":   "CHEMBL23_Inpharmatica",
        "CHEMBL23_LINT":           "CHEMBL23_LINT",
        "CHEMBL23_MLSMR":          "CHEMBL23_MLSMR",
        "CHEMBL23_SURECHEMBL":     "CHEMBL23_SureChEMBL",
        "CHEMBL_RO3":              "CHEMBL_Ro3",
        "CHEMBL_RO5":              "CHEMBL_Ro5",
    }
 
    def __init__(
        self,
        catalogs: Optional[List[str]] = None,
        exclude: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.catalogs = [c.upper() for c in (catalogs or ["PAINS"])]
        self.exclude = exclude
        self._catalog_obj = None  # built lazily
 
    def get_config(self):
        config = super().get_config()
        config.update({"catalogs": self.catalogs, "exclude": self.exclude})
        return config
 
    def _build_catalog(self):
        """Lazily construct and cache the RDKit FilterCatalog object."""
        if self._catalog_obj is not None:
            return self._catalog_obj
 
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
 
        params = FilterCatalogParams()
        for name in self.catalogs:
            rdkit_name = self._CATALOG_MAP.get(name)
            if rdkit_name is None:
                raise ValueError(
                    f"Unknown catalog: '{name}'. "
                    f"Supported catalogs: {list(self._CATALOG_MAP.keys())}"
                )
            catalog_enum = getattr(FilterCatalogParams.FilterCatalogs, rdkit_name, None)
            if catalog_enum is None:
                raise ValueError(
                    f"RDKit does not expose catalog '{rdkit_name}' via "
                    "FilterCatalogParams.FilterCatalogs in this version."
                )
            params.AddCatalog(catalog_enum)
 
        self._catalog_obj = FilterCatalog(params)
        return self._catalog_obj
 
    def _process_item(self, item):
        if item is None:
            return False
 
        catalog = self._build_catalog()
        has_match = catalog.HasMatch(item)
        return not has_match if self.exclude else has_match