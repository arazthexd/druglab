from typing import List, Dict, Any
import h5py

from rdkit import Chem
from rdkit.Chem import rdChemReactions

def load_rxns_file(filename: str) -> List[rdChemReactions.ChemicalReaction]:
    suffix = filename.split(".")[-1]
    if suffix == "txt":
        with open(filename, "r") as f:
            rxns = [rdChemReactions.ReactionFromSmarts(smarts.strip()) 
                    for smarts in f.readlines()]
        return rxns
    else:
        raise NotImplementedError()

def load_mols_file(filename: str, **kwargs: Dict[str, Dict[str, Any]]) \
    -> List[Chem.Mol]:
    suffix = filename.split(".")[-1]
    if suffix == "sdf":
        suppl = Chem.SDMolSupplier(filename, **kwargs.get('sdf', dict()))
        return [mol for mol in suppl if mol is not None]
    elif suffix == "smi":
        suppl = Chem.SmilesMolSupplier(filename, **kwargs.get('smi', dict()))
        return [mol for mol in suppl if mol is not None]
    elif suffix == "h5":
        with h5py.File(filename, "r") as f:
            return [Chem.JSONToMols(mol.decode())[0] for mol in f['molecules']]
    else:
        raise NotImplementedError()