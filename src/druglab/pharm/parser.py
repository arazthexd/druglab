from typing import List, Dict, Any, Tuple, Callable
import string
from collections import OrderedDict

from rdkit import Chem

from .groups import PharmGroup
from .drawopts import DrawOptions
from .ftypes import PharmFeatureType, PharmArrowType, PharmSphereType
from .utilities import parse_varname
from .calculations import *

NAME2FUNC: Dict[str, Callable] = {
    "eplane3": eplane3,
    "tetrahedral3": tetrahedral3,
    "tetrahedral4": tetrahedral4,
    "direction": direction,
    "plane3": plane3,
    "perpendicular": perpendicular,
    "pmean": pmean,
    "sum": lambda *xs: (sum(xs), ),
    "multiply": lambda x, y: (x*y, ),
    # "minus": geom_minus,
    # "mean": geom_mean,
    "norm": norm
}

ELEM_TABLE = Chem.GetPeriodicTable()

def _string2dict(s: str) -> Dict[str, str]:
    args = s.split()
    arg_keys = [arg.split("=")[0] for arg in args]
    arg_vals = [arg.split("=")[1] for arg in args]
    return dict(zip(arg_keys, arg_vals))

class PharmDefinitions:
    def __init__(self):
        self.groups: List[PharmGroup] = []
        self.drawopts: Dict[str, DrawOptions] = {}
        self.patterns: Dict[str, str] = {}
        self.ftypes: OrderedDict[str, PharmFeatureType] = OrderedDict()
    
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

class PharmParser:
    def parse(self, path: str) -> PharmDefinitions:
        raise NotImplementedError()

class PharmDefaultParser:
    def __init__(self):
        pass

    def parse(self, path: str) -> PharmDefinitions:

        definitions = PharmDefinitions()
        
        with open(path, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        
        counter = -1
        while counter+1 < len(lines):
            
            counter += 1
            line = lines[counter]

            if line.startswith("#") or len(line) == 0:
                continue

            if line.startswith("DRAWOPTS"):
                _, drawopt_name, drawopt_args = line.split(maxsplit=2)
                drawopt_kwargs = _string2dict(drawopt_args)
                
                for argkey, argval in dict(drawopt_kwargs).items():
                    try: 
                        argval = float(argval)
                        drawopt_kwargs[argkey] = argval
                        continue
                    except:
                        pass

                    try:
                        argval = argval.split(",")
                        argval = [float(a.strip(" []")) for a in argval]
                        drawopt_kwargs[argkey] = argval
                        continue
                    except:
                        pass

                    try:
                        argval = argval.strip("{}")
                        argval = definitions.defined[argval]
                        drawopt_kwargs[argkey] = argval
                        continue
                    except:
                        pass

                    raise ValueError()

                definitions.drawopts[drawopt_name] = \
                    DrawOptions(**drawopt_kwargs)
                continue

            if line.startswith("PATTERN"):
                _, pattern_name, pattern_smarts = line.split()
                if pattern_name not in definitions.patterns:
                    definitions.patterns[pattern_name] = pattern_smarts
                else:
                    raise NotImplementedError()
                continue

            if line.startswith("FEATURETYPE"):
                _, ftype_name, ftype_args = line.split(maxsplit=2)
                ftype_kwargs = _string2dict(ftype_args)
                ftype_kwargs = {
                    k: definitions.defined[v.strip("{}")] 
                    if v.strip("{}") in definitions.defined else v
                    for k, v in ftype_kwargs.items()
                }
                definitions.ftypes[ftype_name] = \
                    self._get_feature_type(name=ftype_name, **ftype_kwargs)
                continue

            if line.startswith("GROUP"):
                self._parse_group(lines, counter, definitions)
                continue
                
        return definitions
            

    def _parse_group(self, 
                     lines: List[str], 
                     counter: int, 
                     definitions: PharmDefinitions):
        line = lines[counter]
        tag, group_name, group_smarts = line.split()
        assert tag == "GROUP"
        
        group_smarts = group_smarts.format(**definitions.defined)
        ftypes = []
        fargs = []
        
        nextline = lines[counter+1]
        while not nextline.startswith("ENDGROUP"):
            counter += 1
            line = nextline
            nextline = lines[counter+1]

            if line.startswith("CALCULATE"):
                calcs = []
                while not nextline.startswith("ENDCALC"):
                    counter += 1
                    line = nextline
                    nextline = lines[counter+1]

                    outkeys, opstr = line.split("<<")
                    outkeys = outkeys.strip()
                    opstr = opstr.strip()
                    
                    outkeys = outkeys.split(",")
                    outkeys = [key.strip() for key in outkeys]

                    opfunc, *opinps = opstr.split()
                    opfunc = opfunc.strip()
                    opinps = [inp.strip() for inp in opinps]

                    opfunc = NAME2FUNC[opfunc]
                    
                    calc = PharmCalculation(
                        inputs=opinps,
                        function=opfunc,
                        outkeys=outkeys
                    )
                    
                    calcs.append(calc)
                
            if line.startswith("FEATURE"):
                _, ftype_name, *feature_args = line.split()
                ftype = definitions.ftypes[ftype_name]
                ftypes.append(ftype)
                fargs.append(feature_args)
        
        definitions.groups.append(
            PharmGroup(
                name=group_name,
                query=Chem.MolFromSmarts(group_smarts),
                calcs=calcs,
                ftypes=ftypes,
                fargs=fargs
            )
        )
    
    def _get_feature_type(self, 
                          name: str,
                          type: str = "SPHERE", 
                          drawopts: DrawOptions = None) -> PharmFeatureType:
        
        if drawopts is None:
            drawopts = DrawOptions()
        
        if type == "ARROW":
            return PharmArrowType(name=name,
                                  drawopts=drawopts)

        if type == "SPHERE":
            return PharmSphereType(name=name,
                                   drawopts=drawopts)



        