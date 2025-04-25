from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom

@dataclass
class MOPACConfig:
    keywords: List[str] = field(default_factory=lambda: ["PDBOUT"])
    desc1: str = "MOPACConfig"
    desc2: str = ""
    coordinate_lines: List[str] = field(default_factory=list)
    charge: int = 0

    def add_molecule(self, mol: Chem.Mol):
        self.charge += Chem.GetFormalCharge(mol)
        pdb_block = Chem.MolToPDBBlock(mol)
        self.coordinate_lines.extend([
            line for line in pdb_block.split("\n") 
            if line.startswith("HETATM") or line.startswith("ATOM")
        ])

    def get_config_str(self):
        return f"""{" ".join(self.keywords)} CHARGE={self.charge}
{self.desc1}
{self.desc2}
{"\n".join(self.coordinate_lines)}
"""

@dataclass
class MOPACMozymeConfig(MOPACConfig):
    setpi: List[Tuple[int, int]] = field(default_factory=list)
    neg_cvb: List[Tuple[int, int]] = field(default_factory=list)

    def __post_init__(self):
        self.keywords.extend(["MOZYME", "GEO-OK", "SETPI"])
        self.desc1 = "MOPACMozymeConfig"

    def add_molecule(self, mol: Chem.Mol):
        current_atom_count = len(self.coordinate_lines)
        super().add_molecule(mol)
        new_atoms = range(current_atom_count + 1, 
                        current_atom_count + mol.GetNumAtoms() + 1)
        Chem.Kekulize(mol)
        for bond in mol.GetBonds():
            if bond.GetBondType() in [Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]:
                start = new_atoms[bond.GetBeginAtomIdx()]
                end = new_atoms[bond.GetEndAtomIdx()]
                self.setpi.append((start, end))
                if bond.GetBondType() == Chem.BondType.TRIPLE:
                    self.setpi.append((start, end))

        mol = Chem.AddHs(mol)
        rdDistGeom.EmbedMolecule(mol)
        conf = mol.GetConformer()
        oxygen_atoms = [
            (i, conf.GetAtomPosition(i)) 
            for i in range(mol.GetNumAtoms())
            if mol.GetAtomWithIdx(i).GetSymbol() == "O"
        ]
        
        for i, (idx1, pos1) in enumerate(oxygen_atoms):
            for j, (idx2, pos2) in enumerate(oxygen_atoms[i+1:], i+1):
                if np.linalg.norm(np.array(pos1) - np.array(pos2)) < 3.0:
                    self.neg_cvb.append((
                        new_atoms[idx1], 
                        new_atoms[idx2]
                    ))

    def get_config_str(self):
        base_config = super().get_config_str().strip()
        setpi_lines = "\n".join([f"{a} {b}" for a, b in self.setpi])
        cvb_lines = " ".join([f":-{a}-{b}" for a, b in self.neg_cvb])
        
        return f"""{base_config}

{setpi_lines}
"""
    # TODO implementation of cvb