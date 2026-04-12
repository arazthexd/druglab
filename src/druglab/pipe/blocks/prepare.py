from typing import Optional

from druglab.db.base import BaseTable
from druglab.pipe.archetypes import BasePreparation

# ---------------------------------------------------------------------------
# Standardization
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

# ---------------------------------------------------------------------------
# Conformers
# ---------------------------------------------------------------------------


class ConformerGenerator(BasePreparation):
    """
    Embeds a molecule into 3D space using RDKit's ETKDGv3 algorithm and
    optionally optimises the geometry with MMFF94s (default) or UFF.

    By default, only the **lowest-energy conformer** is kept on the output molecule.
    If `keep_all=True`, all generated conformers are retained and sorted by 
    energy (lowest energy becomes conformer ID 0).
    If embedding fails entirely for a molecule, the original (2D) mol is
    returned unchanged so downstream steps can still handle it gracefully.

    Pipeline integration
    --------------------
    This block is a :class:`~druglab.pipe.archetypes.BasePreparation`, so it
    modifies ``table.objects`` in-place (within the copied table).  After
    running, use :meth:`~druglab.db.molecule.MoleculeTable.unroll_conformers`
    to explode multi-conformer molecules into a
    :class:`~druglab.db.conformer.ConformerTable` for per-conformer analysis.

    Parameters
    ----------
    n_confs : int
        Number of conformers to generate per molecule before energy ranking.
        Default 10.
    ff : {"MMFF94s", "MMFF94", "UFF"}
        Force field used for geometry optimisation and energy ranking.
        ``"MMFF94s"`` (the small-molecule MMFF variant with additional
        torsion terms) is the default and generally recommended.
        Pass ``None`` to skip optimisation entirely.
    max_iters : int
        Maximum number of force-field minimisation iterations.  Default 2000.
    random_seed : int
        Random seed for the ETKDG conformer generator.  Default 42.
    add_hs : bool
        Whether to add explicit hydrogens before embedding and strip them
        afterwards.  Highly recommended (default *True*); ETKDG needs Hs to
        generate correct 3D geometry.
    keep_all : bool
        If True, retains all successfully embedded conformers, sorting them 
        by energy (lowest energy first). If False, only the lowest energy 
        conformer is kept. Default False.
    """

    _SUPPORTED_FF = ("MMFF94s", "MMFF94", "UFF")

    def __init__(
        self,
        n_confs: int = 10,
        ff: Optional[str] = "MMFF94s",
        max_iters: int = 2000,
        random_seed: int = 42,
        add_hs: bool = True,
        keep_all: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if ff is not None and ff not in self._SUPPORTED_FF:
            raise ValueError(
                f"Unsupported force field '{ff}'. "
                f"Choose from {self._SUPPORTED_FF} or None."
            )
        self.n_confs = n_confs
        self.ff = ff
        self.max_iters = max_iters
        self.random_seed = random_seed
        self.add_hs = add_hs
        self.keep_all = keep_all

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_confs": self.n_confs,
            "ff": self.ff,
            "max_iters": self.max_iters,
            "random_seed": self.random_seed,
            "add_hs": self.add_hs,
            "keep_all": self.keep_all,
        })
        return config

    def _process_item(self, item):  # noqa: C901  (complexity is inherent here)
        if item is None:
            return None

        from rdkit import Chem
        from rdkit.Chem import AllChem

        # --- add Hs for better embedding geometry ---
        mol = Chem.AddHs(item) if self.add_hs else Chem.Mol(item)

        # --- ETKDG embedding ---
        params = AllChem.ETKDGv3()
        params.randomSeed = self.random_seed
        params.numThreads = 1  # keep deterministic in multiprocessing context

        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=self.n_confs, params=params)

        if len(conf_ids) == 0:
            # Fallback: return the original molecule without 3D coords.
            return item

        energies = {}

        # --- force-field optimisation & energy ranking ---
        if self.ff in ("MMFF94s", "MMFF94"):
            mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=self.ff)
            if mp is not None:
                for cid in conf_ids:
                    ff_obj = AllChem.MMFFGetMoleculeForceField(
                        mol, mp, confId=cid
                    )
                    if ff_obj is None:
                        continue
                    ff_obj.Minimize(maxIts=self.max_iters)
                    energies[cid] = ff_obj.CalcEnergy()

        elif self.ff == "UFF":
            for cid in conf_ids:
                ff_obj = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                if ff_obj is None:
                    continue
                ff_obj.Minimize(maxIts=self.max_iters)
                energies[cid] = ff_obj.CalcEnergy()

        # --- Sort and retain conformers ---
        # Sort conformer IDs by energy (if energy calculation failed/skipped, float('inf') pushes it to the back)
        if energies:
            sorted_cids = sorted(conf_ids, key=lambda cid: energies.get(cid, float('inf')))
        else:
            # Fallback if no force field was applied or parameterisation failed
            sorted_cids = list(conf_ids)

        # Decide which conformers to keep
        cids_to_keep = sorted_cids if self.keep_all else [sorted_cids[0]]
        
        # Extract conformers before clearing
        n_atoms = mol.GetNumAtoms()
        confs_to_keep = [Chem.Conformer(n_atoms) for cid in cids_to_keep]
        [confs_to_keep[i].SetPositions(mol.GetConformer(cid).GetPositions()) for i, cid in enumerate(cids_to_keep)]
        
        # Clear existing conformers and add the kept ones back sequentially
        # assignId=True ensures they are indexed 0, 1, 2... based on energy
        mol.RemoveAllConformers()
        for conf in confs_to_keep:
            mol.AddConformer(conf, assignId=True)

        # --- strip Hs if we added them ---
        if self.add_hs:
            try:
                mol = Chem.RemoveHs(mol)
            except Exception:
                pass  # keep with Hs if stripping fails

        return mol