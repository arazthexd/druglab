from typing import Dict
from openmm import app, unit
import parmed
from openff.toolkit import Molecule
from openff.units.openmm import to_openmm as openff_unit_to_openmm
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
DEFAULT_FORCEFIELDS = ['amber14-all.xml', 'amber14/tip3pfb.xml']

class ProteinParameterizer:
    def __init__(self, forcefield=None):
        self.forcefield = forcefield or app.ForceField(*DEFAULT_FORCEFIELDS)
    
    def parameterize(self, protein: parmed.Structure) -> parmed.Structure:
        for res in protein.residues:
            if res.name == "WAT":
                res.name = "HOH"
    
        system = self.forcefield.createSystem(protein.topology, rigidWater=False)
        
        param_protein = parmed.openmm.load_topology(
            protein.topology,
            system,
            protein.positions
        )
        
        for i, atom in enumerate(param_protein.atoms):
            atom.formal_charge = protein.atoms[i].formal_charge
            
        return param_protein

class SmallMoleculeParameterizer:
    def __init__(self, forcefield=None):
        self.forcefield = forcefield or app.ForceField(*DEFAULT_FORCEFIELDS)
    
    def parameterize(self, molecule: parmed.Structure) -> parmed.Structure:
        rdmol = self._parmed_to_rdkit(molecule)
        offmol = Molecule.from_rdkit(
            rdmol, 
            hydrogens_are_explicit=True,
            allow_undefined_stereo=True
        )
        
        offmol.assign_partial_charges('mmff94')
        offtop = offmol.to_topology()

        smirnoff = SMIRNOFFTemplateGenerator(molecules=offmol)
        self.forcefield.registerTemplateGenerator(smirnoff.generator)
        ommsys = self.forcefield.createSystem(offtop.to_openmm())
        param_mol = parmed.openmm.load_topology(
            molecule.topology,
            ommsys,
            molecule.positions
        )

        for i, atom in enumerate(param_mol.atoms):
            atom.formal_charge = molecule.atoms[i].formal_charge
            
        return param_mol
    
    def _parmed_to_rdkit(self, struct: parmed.Structure):
        from rdkit import Chem
        return Chem.MolFromPDBBlock("\n".join(
            f"ATOM  {i+1:5} {atom.name:4} {res.name:3} {res.number:4}    "
            f"{atom.xx:8.3f}{atom.xy:8.3f}{atom.xz:8.3f}"
            for i, (atom, res) in enumerate(zip(struct.atoms, struct.residues))
        ))