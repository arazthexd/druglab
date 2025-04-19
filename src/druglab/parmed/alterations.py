import sys
import os
from pathlib import Path
current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent

sys.path.append(str(parent_dir))

from typing import Dict, List, Tuple
import numpy as np
import parmed as pmd
from openmm import app, unit
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
DEFAULT_FORCEFIELDS = ['amber14-all.xml', 'amber14/tip3pfb.xml']
from pocketloc import PocketLocation

class ParmedProteinPocketIsolator:
    def __init__(self, debug: bool = False):
        self.debug = debug

    def isolate(self, prot: pmd.Structure, loc: PocketLocation) -> pmd.Structure:
        self.fix_cyx(prot, loc)
        
        for res in prot.residues:
            if self._is_residue_included(res, loc):
                continue
            
            nei_n, nei_c = self._get_neighboring_residues(prot, res)
            is_inc_n = self._is_residue_included(nei_n, loc) if nei_n else False
            is_inc_c = self._is_residue_included(nei_c, loc) if nei_c else False
            
            if is_inc_n and is_inc_c:
                continue
            elif not (is_inc_n or is_inc_c):
                self.delete_residue_atoms(prot, res)
            elif is_inc_n:
                self.create_cterm(prot, res)
            elif is_inc_c:
                self.create_nterm(prot, res)
                
        return self.clean_up(prot)

    def fix_cyx(self, prot: pmd.Structure, loc: PocketLocation):
        for res in prot.residues:
            if res.name.strip() != "CYS":
                continue
                
            s_atom = next((at for at in res.atoms if at.element_name == "S"), None)
            if s_atom and any(at.element_name == "S" for at in s_atom.bond_partners):
                res.name = "CYX"

        for cyx_res in [r for r in prot.residues if r.name.strip() == "CYX"]:
            if not self._is_residue_included(cyx_res, loc):
                continue
                
            for cyx_at in [a for a in cyx_res.atoms if a.element_name == "S"]:
                for nei_s in [a for a in cyx_at.bond_partners if a.element_name == "S"]:
                    if not self._is_residue_included(nei_s.residue, loc):
                        self._convert_s_to_h(prot, cyx_at, nei_s, cyx_res)
                        cyx_res.name = "CYS"

    def _convert_s_to_h(self, prot: pmd.Structure, cyx_at: pmd.Atom, 
                       nei_s: pmd.Atom, cyx_res: pmd.Residue):

        new_coords = self.shorten_distance(
            np.array([cyx_at.xx, cyx_at.xy, cyx_at.xz]),
            np.array([nei_s.xx, nei_s.xy, nei_s.xz]),
            1.337
        )
        
        h_atom = pmd.Atom(atomic_number=1, name="HG", charge=0)
        h_atom.xx, h_atom.xy, h_atom.xz = new_coords
        prot.add_atom_to_residue(h_atom, cyx_res)
        prot.bonds.append(pmd.Bond(h_atom, cyx_at))

    def shorten_distance(self, pos_fixed: np.ndarray, pos_free: np.ndarray, 
                        dist: float) -> np.ndarray:

        diff = pos_free - pos_fixed
        return pos_fixed + (diff / np.linalg.norm(diff)) * dist

    def _is_residue_included(self, residue: pmd.Residue, 
                            loc: PocketLocation) -> bool:
        if residue is None:
            return False
        return any(loc.point_is_included(np.array([a.xx, a.xy, a.xz])) 
               for a in residue.atoms)

    def _get_neighboring_residues(self, structure: pmd.Structure,
                                residue: pmd.Residue) -> Tuple[pmd.Residue, pmd.Residue]:
        nei_n, nei_c = None, None
        for atom in residue.atoms:
            if atom.name == "N":
                nei_n = next((a.residue for a in atom.bond_partners 
                            if a.name == "C"), None)
            elif atom.name == "C":
                nei_c = next((a.residue for a in atom.bond_partners 
                            if a.name == "N"), None)
        return nei_n, nei_c

    def create_nterm(self, structure: pmd.Structure, residue: pmd.Residue):
        if residue.name.strip() == "ACE":
            return
            
        for atom in list(residue.atoms):
            if atom.name.strip() in ["O", "C"]:
                continue
            elif atom.name.strip() == "CA":
                atom.name = "CH3"
            elif atom.name.strip() in ["HA", "HA2", "HA3"]:
                atom.name = {"HA": "HH31", "HA2": "HH31", "HA3": "HH32"}[atom.name]
            elif atom.name.strip() in ["CB", "N"]:
                atom.name = {"CB": "HH32", "N": "HH33"}[atom.name]
                atom.atomic_number = 1
                atom_ca = next(a for a in residue.atoms if a.name.strip() in ["CA", "CH3"])
                atom.xx, atom.xy, atom.xz = self.shorten_distance(
                    [atom_ca.xx, atom_ca.xy, atom_ca.xz],
                    [atom.xx, atom.xy, atom.xz],
                    1.08
                )
            else:
                self.delete_atom(structure, atom)
        
        residue.name = "ACE"

    def create_cterm(self, structure: pmd.Structure, residue: pmd.Residue):
        if residue.name.strip() == "NME":
            return
            
        for atom in list(residue.atoms):
            if atom.name.strip() in ["N", "H"]:
                continue
            elif atom.name.strip() == "CA":
                atom.name = "CH3"
            elif atom.name.strip() in ["HA", "HA2", "HA3"]:
                atom.name = {"HA": "HH31", "HA2": "HH31", "HA3": "HH32"}[atom.name]
            elif atom.name.strip() in ["CB", "C"]:
                atom.name = {"CB": "HH32", "C": "HH33"}[atom.name]
                atom.atomic_number = 1
                atom_ca = next(a for a in residue.atoms if a.name.strip() in ["CA", "CH3"])
                atom.xx, atom.xy, atom.xz = self.shorten_distance(
                    [atom_ca.xx, atom_ca.xy, atom_ca.xz],
                    [atom.xx, atom.xy, atom.xz],
                    1.08
                )
            elif atom.name.strip() == "CD" and residue.name.strip() == "PRO":
                atom.name = "H"
                atom.atomic_number = 1
                atom_n = next(a for a in residue.atoms if a.name.strip() == "N")
                atom.xx, atom.xy, atom.xz = self.shorten_distance(
                    [atom_n.xx, atom_n.xy, atom_n.xz],
                    [atom.xx, atom.xy, atom.xz],
                    1.08
                )
            else:
                self.delete_atom(structure, atom)
        
        residue.name = "NME"

    def delete_atom(self, structure: pmd.Structure, atom: pmd.Atom):
        for bond in list(atom.bonds):
            if bond in structure.bonds:
                structure.bonds.remove(bond)
        if atom in structure.atoms:
            structure.atoms.remove(atom)
        if atom.residue and atom in atom.residue.atoms:
            atom.residue.atoms.remove(atom)

    def delete_residue_atoms(self, structure: pmd.Structure, residue: pmd.Residue):
        
        for atom in list(residue.atoms):
            self.delete_atom(structure, atom)

    def clean_up(self, structure: pmd.Structure) -> pmd.Structure:
        for atom in list(structure.atoms):
            for bond in list(atom.bonds):
                if bond.atom1.residue is None or bond.atom2.residue is None:
                    atom.bonds.remove(bond)
        
        for res in list(structure.residues):
            if len(res.atoms) == 0:
                structure.residues.remove(res)
        
        for residue in structure.residues:
            if not residue.chain:
                residue.chain = "A"
        
        return structure


class ParmedProteinPreparator:
    def __init__(self, remove_water: bool = True, add_H: bool = True,
                 remove_heterogens: bool = False, 
                 replace_nonstandard_residues: bool = True,
                 debug: bool = False):
        self.remove_water = remove_water
        self.add_H = add_H
        self.remove_heterogens = remove_heterogens
        self.replace_nonstandard = replace_nonstandard_residues
        self.debug = debug

    def prepare(self, structure: pmd.Structure) -> pmd.Structure:
        if self.remove_water:
            for res in list(structure.residues):
                if res.name.strip() in ["WAT", "HOH"]:
                    self._delete_residue_atoms(structure, res)
        
        if self.remove_heterogens:
            ff = app.ForceField(*DEFAULT_FORCEFIELDS)
            standard_residues = set(ff._templates.keys())
            for res in list(structure.residues):
                if res.name.strip() not in standard_residues:
                    self._delete_residue_atoms(structure, res)
        
        if self.add_H or self.replace_nonstandard:
            modeller = app.Modeller(structure.topology, 
                                   structure.coordinates * unit.angstrom)
            if self.add_H:
                modeller.addHydrogens()
            
            ff = app.ForceField(*DEFAULT_FORCEFIELDS)
            system = ff.createSystem(modeller.topology, rigidWater=False)
            return pmd.openmm.load_topology(modeller.topology, system, modeller.positions)
        
        return structure

    def _delete_residue_atoms(self, structure: pmd.Structure, residue: pmd.Residue):
        for atom in list(residue.atoms):
            self._delete_atom(structure, atom)

    def _delete_atom(self, structure: pmd.Structure, atom: pmd.Atom):
        for bond in list(atom.bonds):
            if bond in structure.bonds:
                structure.bonds.remove(bond)
        if atom in structure.atoms:
            structure.atoms.remove(atom)
        if atom.residue and atom in atom.residue.atoms:
            atom.residue.atoms.remove(atom)


class ParmedMoleculeCombiner:
    def __init__(self, num_inputs: int, debug: bool = False):
        self.num_inputs = num_inputs
        self.debug = debug

    def combine(self, input_dict: Dict[str, pmd.Structure]) -> pmd.Structure:
        molecules = [input_dict[f"molecule_{i+1}"] for i in range(self.num_inputs)]
        combined = molecules[0].copy()
        for mol in molecules[1:]:
            combined += mol.copy()
        return combined

    @property
    def input_keys(self) -> List[str]:
        return [f"molecule_{i+1}" for i in range(self.num_inputs)]