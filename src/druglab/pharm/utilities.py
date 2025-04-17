from typing import List

from rdkit import Chem

def parse_neighboring_key(key: str):
    if key.startswith("nn"):
        ntype, nidx, nnidx = key.split("-")
    elif key.startswith("n"):
        ntype, nidx = key.split("-")
        nnidx = 0
    else:
        raise NotImplementedError()
    
    if nnidx == "a":
        nnidx = -1
    if nidx == "a":
        nidx = -1
    nnidx, nidx = int(nnidx), int(nidx)
    
    atomidx = int(ntype.strip("n()"))
    ntype = ntype.split("(")[0]
    return atomidx, ntype, nidx, nnidx

def get_neighboring(mol: Chem.Mol, 
                    atomidx: int, ntype: str, 
                    nidx: int = -1, nnidx: int = -1):
    
    atom = mol.GetAtomWithIdx(atomidx)
    neis: List[int] = [nei.GetIdx() for nei in atom.GetNeighbors()]
    
    if ntype == "n":
        if nidx == -1:
            return neis
        return neis[nidx]
    
    if ntype == "nn":
        nns = []
        for nei in atom.GetNeighbors():
            nei: Chem.Atom
            nneis = get_neighboring(mol,
                                    atomidx=nei.GetIdx(),
                                    ntype="n",
                                    nidx=-1)
            nneis.remove(atom.GetIdx())
            nns.append(nneis)
        
        if nidx > -1:
            nns = nns[nidx]
            if nnidx > -1:
                return nns[nnidx]
            return nns
        
        if nnidx > -1:
            return [ns[nnidx] for ns in nns]
        return nns