from typing import Dict, Any, Union, List
import numpy as np
from dataclasses import dataclass
from copy import deepcopy
from rdkit import Chem
from interface import MOPACInterface

@dataclass
class MOPACOptimizer(MOPACInterface):
    def run_optimization(self, 
                        molecules: Union[Chem.Mol, List[Chem.Mol]], 
                        debug: bool = False) -> Dict[str, Any]:
        self.reset_config()
        self.update_config(molecules)
        
        if not any(kw in self.config.keywords for kw in ["EF", "TS", "PRECISE"]):
            self.config.keywords.append("EF")
            
        out_path, _ = self.run_job(base_dir=".", debug=debug)
        
        with open(out_path, "r") as f:
            out_str = f.read()

        pre_energy = float(out_str.split("CYCLE:     1")[1].split()[8])
        post_energy = float(out_str.split("FINAL HEAT OF FORMATION =")[1].split()[0])

        coord_section = out_str.split(" CARTESIAN COORDINATES\n\n")[1] \
                          .split("\n\n           Empirical Formula")[0]
        coordinates = np.array([
            [float(coord) for coord in line.split()[2:5]] 
            for line in coord_section.splitlines() if line.strip()
        ])

        return {
            "out_path": out_path,
            "pre_energy": pre_energy,
            "post_energy": post_energy,
            "coordinates": coordinates
        }