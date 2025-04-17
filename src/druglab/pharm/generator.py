from __future__ import annotations
from typing import List, Dict, Union, Tuple, Set, Any, Callable
from dataclasses import dataclass, field
import yaml

import numpy as np

from rdkit import Chem

from .utilities import get_neighboring, parse_neighboring_key
from .ftypes import (
    PharmSingleType, PharmSingleTypes,
    PharmPairType, PharmPairTypes,
    DrawOptions
)
from .features import (
    BasePharmSingles, PharmSphereSingles, PharmArrowSingles, PharmAIOSingles
)
from.pharmacophore import Pharmacophore
from .functions import *
NAME2FUNC: Dict[str, Callable] = {
    "eplane3": geom_extended_plane_3to2,
    "tetrahedral3": geom_tetrahedral_3to2_new,
    "angle2-point": geom_angle_2to1_point,
    "direction": geom_direction,
    "plane3": geom_plane_3to1,
    "perpendicular3": geom_perpendicular3to1,
    "minus": geom_minus,
    "mean": geom_mean
}

BASEDEF_PATH = __file__.replace("generator.py", "base.yaml")

class PharmPattern:
    def __init__(self, 
                 definition: dict, 
                 stypes: PharmSingleTypes = None,
                 debug: bool = False) -> None:
        
        self.debug = debug
        if debug:
            print("=== initiating pattern ===")
            print("definition = ", definition)
            print()
        
        # Save name...
        self.name: str = definition["name"]
        
        # Save smarts and queries list...
        if isinstance(definition["smarts"], list):
            self.smarts = definition["smarts"]
        else:
            self.smarts = [definition["smarts"]]
        self.queries: List[Chem.Mol] = [Chem.MolFromSmarts(sma) 
                                        for sma in self.smarts]
        
        # Save feature types if given...
        self.stypes = stypes

        # Set up needed functions for new variables...
        if "variables" not in definition:
            definition["variables"] = []
        self.vargens: List[Dict[str, Any]] = definition["variables"]
        for vargen in self.vargens:
            if "values" not in vargen:
                vargen["values"] = []
            if "extra" not in vargen:
                vargen["extra"] = dict()
            assert "func" in vargen
            assert "keys" in vargen

        # Set up feature definitions...
        self.features: List[Dict[str, Union[str, int, float]]] = []
        for feature in definition["features"]:
            if feature["type"] not in self.stypes.names:
                continue
            if "direction" not in feature:
                feature["direction"] = 0
            if "radius" not in feature:
                feature["radius"] = 0
            self.features.append(feature)
        
    def match(self, mol: Chem.Mol) -> List[Tuple[int]]:
        matched: List[Tuple[int]] = []
        firsts: List[int] = []
        for query in self.queries:
            for match in mol.GetSubstructMatches(query):
                if match[0] not in firsts:
                    firsts.append(match[0])
                    matched.append(match)
        return matched
    
    def _temp_neiboring_vars(self,
                             conformer: Chem.Conformer, 
                             keys: List[str], 
                             variables: Dict[Union[str, int], Any], 
                             match: Tuple[int]) -> List[str]:
        temp_keys = [key for key in keys 
                     if key not in variables 
                     and not isinstance(key, int)
                     and key.startswith("n")]
        
        if self.debug:
            print("=== inside pattern's _temp_neiboring_vars ===", self.name)

        for key in temp_keys:
            atomidx, ntype, nidx, nnidx = parse_neighboring_key(key)
            atomidx = match[atomidx]
            nei_idx = get_neighboring(conformer.GetOwningMol(), 
                                      atomidx, ntype, nidx, nnidx)
            if self.debug:
                print("nei of atom = ", nei_idx, " of ", atomidx, ":", ntype)
            variables[key] = conformer.GetPositions()[nei_idx]
        
        if self.debug:
            print()

        return temp_keys


    def update(self, 
               variables: Dict[Union[int, str], Any],
               conformer: Chem.Conformer,
               match: Tuple[int]) -> None:
        
        if self.debug:
            print("=== inside pattern's update ===", self.name)
            print("variables = ", variables)
        
        for settings in self.vargens:
            extra: Dict = settings["extra"]
            func = NAME2FUNC[settings["func"]]
            temp_keys = self._temp_neiboring_vars(conformer,
                                                  settings["values"],
                                                  variables,
                                                  match)
            if self.debug:
                print("temp keys = ", temp_keys)
                print("settings = ", settings)

            func(variables=variables, 
                 output_keys=settings["keys"], 
                 extra=extra, 
                 input_keys=settings["values"])
            
            if self.debug:
                print("temp variables = ", variables)
            
            [variables.pop(key) for key in temp_keys]
        
        if self.debug:
            print("new variables = ", variables)
            print()
            
        return

    def vars2arrows(self, 
                    variables: Dict[Union[int, str], Any]) -> PharmArrowSingles:
        ftns, ps, ds = [], [], []
        for feature in self.features:
            ftname = feature["type"]
            if ftname not in self.stypes.arrows.names:
                continue

            point = feature["point"]
            direction = feature["direction"]

            l = self._get_len(point=point, direction=direction)

            points = point \
                if isinstance(point, list) else [point]*l
            directions = direction \
                if isinstance(direction, list) else [direction]*l

            assert len(points) == len(directions)

            for p, d in zip(points, directions):
                ftns.append(ftname)
                ps.append(variables[p])
                ds.append(variables[d])

        return PharmArrowSingles(
            types=self.stypes.arrows,
            tyidx=np.array([self.stypes.arrows.names.index(name) 
                            for name in ftns], dtype=int).flatten(),
            pos=np.array(ps).reshape(-1, 3),
            vec=np.array(ds).reshape(-1, 3),
        )
    
    def vars2spheres(self, 
                     variables: Dict[Union[int, str], Any]) -> PharmSphereSingles:
        ftns, ps, rs = [], [], []
        for feature in self.features:
            ftname = feature["type"]
            if ftname not in self.stypes.spheres.names:
                continue

            point = feature["point"]
            radius = feature["radius"]

            l = self._get_len(point=point, radius=radius)

            points = point \
                if isinstance(point, list) else [point]*l
            radiuses = radius \
                if isinstance(radius, list) else [radius]*l

            assert len(points) == len(radiuses)

            for p, r in zip(points, radiuses):
                ftns.append(ftname)
                ps.append(variables[p])
                rs.append(r)

        return PharmSphereSingles(
            types=self.stypes.spheres,
            tyidx=np.array([self.stypes.spheres.names.index(name) 
                            for name in ftns], dtype=int).flatten(),
            pos=np.array(ps).reshape(-1, 3),
            radius=np.array(rs)
        )
            
    def vars2feats(self, 
                   variables: Dict[Union[int, str], Any]) -> Pharmacophore:
        pharmacophore = Pharmacophore.empty(
            stypes=self.stypes,
            ptypes=None
        )
        arrows = self.vars2arrows(variables)
        spheres = self.vars2spheres(variables)
        pharmacophore.add_arrows(arrows)
        pharmacophore.add_spheres(spheres)
        return pharmacophore
    
    @staticmethod
    def _get_len(point=None, direction=None, radius=None):
        if isinstance(point, list):
            return len(point)
        if isinstance(direction, list):
            return len(direction)
        if isinstance(radius, list):
            return len(radius)
        return 1
            
    def generate(self, 
                 mol: Chem.Mol, 
                 conf_id: int = -1,
                 stypes: PharmSingleTypes = None) -> Pharmacophore:
        stypes = stypes or self.stypes
        assert stypes is not None

        matches = self.match(mol)
        conformer = mol.GetConformer(conf_id)

        pharmacophore = Pharmacophore.empty(
            stypes=stypes,
            ptypes=None
        )
        for match in matches:
            variables: Dict[Union[int, str], Any] = {
                i: np.array(conformer.GetAtomPosition(idx)) 
                for i, idx in enumerate(match)
            }
            self.update(variables, conformer, match)
            pcore = self.vars2feats(variables)
            pharmacophore += pcore

        return pharmacophore

class PharmGenerator:
    def __init__(self):
        self.stypes: PharmSingleTypes = PharmSingleTypes()
        self.ptypes: PharmPairTypes = PharmPairTypes()
        self.patterns: List[PharmPattern] = []
        self.debug = False

    def read_yaml(self, filename: str):
        with open(filename, "r") as f:
            config = yaml.safe_load(f)

        self._parse_features(config["types"]["features"])
        self._parse_pairs(config["types"]["pairs"])
        self._parse_patterns(config["patterns"])

    def generate(self, mol: Chem.Mol, conf_id: int = -1) -> Pharmacophore:
        
        pharmacophore = Pharmacophore.empty(
            stypes=self.stypes,
            ptypes=self.ptypes
        )
        
        for pattern in self.patterns:
            pcore = pattern.generate(mol, conf_id, self.stypes)
            pharmacophore += pcore

        return pharmacophore
    
    def set_debug(self, debug: bool = False):
        self.debug = debug
        for pattern in self.patterns:
            pattern.debug = debug
    
    @property
    def pattern_names(self):
        return [pattern.name for pattern in self.patterns]
    
    def _parse_features(self, features_dict: List[Dict]):
        for ftdef in features_dict:
            if ftdef["name"] not in self.stypes.names:
                self.stypes.append(PharmSingleType.from_dict(ftdef))

    def _parse_pairs(self, pairs_dict: Dict[str, Any]):
        manual: List[Dict[str, Any]] = pairs_dict.get("manual", [])
        distance: Dict[str, bool] = pairs_dict.get("distance", {})
        
        for manual_def in manual:
            pt = PharmPairType.from_members(
                members=[
                    self.stypes.typename2type(manual_def["fts"][i])
                    for i in range(2)
                ],
                subtype=manual_def["type"],
                name=manual_def["name"]
            )
            if pt.name not in self.ptypes.names:
                self.ptypes.append(pt)

        if distance.get("auto", False):
            disabled_dists = distance.get("disable", [])
            for i in range(len(self.stypes.names)):
                for j in range(i+1, len(self.stypes.names)):
                    pt = PharmPairType.from_members(
                        members=[
                            self.stypes.typename2type(self.stypes.names[i]),
                            self.stypes.typename2type(self.stypes.names[j])
                        ],
                        subtype="distance",
                        name=f"{self.stypes.names[i]}|distance|{self.stypes.names[j]}"
                    )
                    if pt.name not in self.ptypes.names + disabled_dists:
                        self.ptypes.append(pt)
    
    def _parse_patterns(self, patterns: List[Dict]):
        for pattern in patterns:
            if pattern["name"] not in self.pattern_names:
                self.patterns.append(PharmPattern(definition=pattern, 
                                                  stypes=self.stypes,
                                                  debug=self.debug))
