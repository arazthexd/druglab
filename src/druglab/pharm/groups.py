from typing import List, Literal
from dataclasses import dataclass, field

from rdkit import Chem

from .ftypes import PharmFeatureType
from .pharmacophore import Pharmacophore, PharmacophoreList
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
                 confid: int | Literal["all"] = -1) \
                    -> Pharmacophore | PharmacophoreList:
        
        if confid == "all":
            pharms = [self.generate(mol, confid=i) 
                      for i in range(mol.GetNumConformers())]
            return PharmacophoreList(pharms)

        pharms = []
        matches = mol.GetSubstructMatches(self.query)
        for match in matches:

            pharm = Pharmacophore()
            
            vars = {}
            for calc in self.calcs:
                calc(vars=vars, match=match, mol=mol, confid=confid)
            
            for ftype, args in zip(self.ftypes, self.fargs):
                args = [
                    parse_varname(varname=arg, mol=mol, match=match,
                                  confid=confid, vars=vars)
                    for arg in args
                ]
                pharm.add_feature(ftype, *args, atidx=match)
            
            pharms.append(pharm)
        
        return sum(pharms, start=Pharmacophore())
            
            
