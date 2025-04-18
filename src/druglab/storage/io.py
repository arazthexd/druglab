from typing import List

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

def load_mols_file(filename: str) -> List[Chem.Mol]:
    suffix = filename.split(".")[-1]
    if suffix == "sdf":
        suppl = Chem.SDMolSupplier(filename)
        return [mol for mol in suppl if mol is not None]
    elif suffix in ["smi", "txt"]:
        suppl = Chem.SmilesMolSupplier(filename)
        return [mol for mol in suppl if mol is not None]
    else:
        raise NotImplementedError()