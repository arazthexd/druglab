from typing import List, Dict, Any, Tuple
import os
from collections import OrderedDict

from rdkit import Chem

from .groups import PharmGroup
from .drawopts import DrawOptions
from .ftypes import PharmFeatureType
from .parser import PharmDefaultParser, PharmParser, PharmDefinitions
from .pharmacophore import Pharmacophore

BASE_DEFINITIONS_PATH = os.path.abspath(__file__).replace("generator.py", 
                                                          "definitions.pharm")

class PharmGenerator:
    def __init__(self):
        self.groups: List[PharmGroup] = []
        self.drawopts: Dict[str, DrawOptions] = {}
        self.patterns: Dict[str, str] = {}
        self.ftypes: Dict[str, PharmFeatureType] = OrderedDict()

        self._loaded = False
    
    def load_file(self, path: str, parser: PharmParser = None):
        parser = parser or PharmDefaultParser()
        definitions: PharmDefinitions = parser.parse(path)
        self.load_definitions(definitions)

        self._loaded = True
    
    def load_definitions(self, definitions: PharmDefinitions):

        for group in definitions.groups:
            if group not in self.groups:
                self.groups.append(group)

        self.drawopts.update(definitions.drawopts)
        self.patterns.update(definitions.patterns)

        for ftname, ftype in definitions.ftypes.items():
            if ftname in self.ftype_names:
                print((f"WARNING: Duplicate FeatureType: {ftname} "
                       "Overwriting..."))
            self.ftypes[ftname] = ftype
    
    def generate(self, mol: Chem.Mol, confid: int = -1) -> Pharmacophore:
        if self._loaded == False:
            self.load_file(BASE_DEFINITIONS_PATH)
        
        pcore = Pharmacophore()
        for group in self.groups:
            pcore += group.generate(mol, confid)
        return pcore
    
    @property
    def ftype_names(self) -> List[str]:
        return [ftype.name for ftype in self.ftypes.values()]
    
    @property
    def defined(self) -> Dict[str, Any]:
        out = {}
        out.update({
            k: v.strip("[]") for k, v in self.patterns.items()
        })
        out.update(self.drawopts)
        out.update(dict(self.ftypes))
        return out

