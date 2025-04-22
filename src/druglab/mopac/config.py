from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom

MOPAC_TEMPLATE = """{keywords}
{desc1}
{desc2}
{coordinates}
"""

@dataclass
class MOPACConfig:
    keywords: List[str] = field(default_factory=lambda: ["PDBOUT"])
    desc1: str = "MOPACConfig "
    desc2: str = ""
    coordinate_lines: List[str] = field(default_factory=list)
    charge: int = 0
    
    def add_molecule(self, mol: Chem.Mol):
        self.charge += Chem.GetFormalCharge(mol)
        mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "Unnamed"
        self.desc2 += f"| {mol_name} "
        pdb_block = Chem.MolToPDBBlock(mol)
        pdb_lines = [line for line in pdb_block.split("\n") 
                    if "ATOM" in line or "HETATM" in line]

        num_current_atoms = len(self.coordinate_lines)
        self.coordinate_lines.extend([
            self._update_atom_numbers(line, num_current_atoms) 
            for line in pdb_lines
        ])
    
    def get_config_str(self) -> str:
        return MOPAC_TEMPLATE.format(
            keywords=" ".join(self.keywords) + f" CHARGE={self.charge}",
            desc1=self.desc1,
            desc2=self.desc2,
            coordinates="\n".join(self.coordinate_lines)
        )
    
    @staticmethod
    def _update_atom_numbers(line: str, num_current_atoms: int) -> str:
        atom_number = int(line[6:11])
        atom_number += num_current_atoms
        return line[:6] + str(atom_number).rjust(5) + line[11:]

@dataclass
class MOPACMozymeConfig(MOPACConfig):
    setpi: str = ""
    neg_cvb: List[Tuple[int, int]] = field(default_factory=list)
    
    def __post_init__(self):
        self.keywords.extend(["MOZYME", "GEO-OK", "SETPI"])
        self.desc1 = "MOPACMozymeConfig "
    
    def add_molecule(self, mol: Chem.Mol):
        num_current_atoms = len(self.coordinate_lines)
        setpi_pairs = self._get_setpi_pairs(mol)
        self.setpi = "\n".join(
            [" ".join([str(p1 + num_current_atoms), str(p2 + num_current_atoms)])
             for p1, p2 in setpi_pairs[:40]]
        )

        neg_cvb_pairs = self._get_neg_cvb_pairs(mol)
        self.neg_cvb.extend([
            (p1 + num_current_atoms, p2 + num_current_atoms) 
            for p1, p2 in neg_cvb_pairs
        ])
        super().add_molecule(mol)
    
    def get_config_str(self) -> str:
        old_keywords = self.keywords.copy()
        cvb_text = self._cvblist_to_cvbtxt(self.neg_cvb, mode="negative")
        if cvb_text:
            self.keywords.append(f"CVB({cvb_text})")
        config = super().get_config_str()
        self.keywords = old_keywords
        return config + "\n" + self.setpi
    
    @staticmethod
    def _get_setpi_pairs(mol: Chem.Mol) -> List[Tuple[int, int]]:
        setpi = []
        Chem.Kekulize(mol)
        for bond in mol.GetBonds():
            if bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
                pair = (bond.GetBeginAtomIdx() + 1, bond.GetEndAtomIdx() + 1)
                setpi.append(pair)
                if bond.GetBondType() == Chem.BondType.TRIPLE:
                    setpi.append(pair)
        return setpi
    
    @staticmethod
    def _get_neg_cvb_pairs(mol: Chem.Mol) -> List[Tuple[int, int]]:
        neg_cvb_list = []
        mol = Chem.AddHs(mol)
        rdDistGeom.EmbedMolecule(mol)
        conf = mol.GetConformer()
        oxygen_atoms = [
            (i, conf.GetAtomPosition(i)) 
            for i in range(mol.GetNumAtoms())
            if mol.GetAtomWithIdx(i).GetSymbol() == "O"
        ]
        for i, (atom1_idx, pos1) in enumerate(oxygen_atoms):
            for atom2_idx, pos2 in oxygen_atoms[i+1:]:
                dist = np.linalg.norm(np.array(pos2) - np.array(pos1))
                if dist < 3.0:
                    neg_cvb_list.append((atom1_idx + 1, atom2_idx + 1))
        
        return neg_cvb_list
    
    @staticmethod
    def _cvblist_to_cvbtxt(
        cvblist: List[Tuple[int, int]], 
        mode: str = "negative", 
        atnum_shift: int = 0
    ) -> str:
        if mode == "negative":
            cvblist = [
                ":-".join((str(p1 + atnum_shift), str(p2 + atnum_shift)))
                for p1, p2 in cvblist
            ]
            return ";".join(cvblist)
        else:
            raise NotImplementedError("Only negative CVB is implemented")