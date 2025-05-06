
from typing import Any, List

from rdkit import Chem

ELEM_TABLE = Chem.GetPeriodicTable()

def parse_varname(varname: str, 
                  mol: Chem.Mol, 
                  match: List[int],
                  vars: dict = None,
                  confid: int = -1) -> Any:
    
    if vars is None:
        vars = dict()
    
    try:
        idx = int(varname)
        return mol.GetConformer(confid).GetPositions()[match[idx]]
    except:
        pass

    try:
        num = float(varname)
        return num
    except:
        pass

    if varname[:3] in ["VOL", "NEI", "NON"]:
        args = varname.split("(")[1].split(")")[0].split(",")
        
        if varname.startswith("VOL"):
            assert len(args) == 1
            at = mol.GetAtomWithIdx(match[int(args[0])])
            return ELEM_TABLE.GetRvdw(at.GetAtomicNum())
        
        if varname.startswith("NEI"):
            assert len(args) == 2
            atomidx, neiidx = match[int(args[0])], int(args[1])
            at = mol.GetAtomWithIdx(atomidx)
            nei: Chem.Atom = list(at.GetNeighbors())[neiidx]
            return mol.GetConformer(confid).GetPositions()[nei.GetIdx()]
        
        if varname.startswith("NON"):
            assert len(args) == 3
            atomidx = match[int(args[0])]
            neiidx, nonidx = int(args[1]), int(args[2])
            at = mol.GetAtomWithIdx(atomidx)
            nei: Chem.Atom = list(at.GetNeighbors())[neiidx]

            if nei.GetNeighbors()[0].GetIdx() == at.GetIdx():
                nonidx += 1
            non: Chem.Atom = list(nei.GetNeighbors())[nonidx]
            return mol.GetConformer(confid).GetPositions()[non.GetIdx()]
    
    return vars[varname]