from typing import List, Callable, Dict, Any
from dataclasses import dataclass, field

from rdkit import Chem

from .ftypes import PharmFeatureType
from .features import PharmFeatures
from .pharmacophore import Pharmacophore
from .utilities import parse_varname
from .calculations import PharmCalculation

@dataclass
class PharmGroup:
    name: str
    query: Chem.Mol
    calcs: List[PharmCalculation] = field(repr=False)
    ftypes: List[PharmFeatureType] = field(repr=False, default_factory=list)
    fargs: List[tuple] = field(repr=False, default_factory=list)

    def generate(self, 
                 mol: Chem.Mol, 
                 confid: int = -1) -> Pharmacophore:

        pcores = []
        
        matches = mol.GetSubstructMatches(self.query)
        for match in matches:

            pcore = Pharmacophore()
            
            vars = {}
            for calc in self.calcs:
                calc(vars=vars, match=match, mol=mol, confid=confid)
            
            for ftype, args in zip(self.ftypes, self.fargs):
                args = [
                    parse_varname(varname=arg, mol=mol, match=match,
                                  confid=confid, vars=vars)
                    for arg in args
                ]
                pcore.add_single(ftype, *args, atidx=match)
            
            pcores.append(pcore)
        
        return sum(pcores, start=Pharmacophore())
            
            
