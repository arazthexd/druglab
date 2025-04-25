from typing import Dict, Any, Union, List
from dataclasses import dataclass
from copy import deepcopy
from rdkit import Chem
from .interface import MOPACInterface

@dataclass
class MOPACEnergyCalculator(MOPACInterface):
    def calculate_energy(self, 
                        debug: bool = False) -> Dict[str, Any]:
        if not any(kw in self.config.keywords for kw in ["1SCF", "SINGLE"]):
            self.config.keywords.append("1SCF")
        out_path, _ = self.run_job(base_dir=self.output_dir, debug=debug)
        with open(out_path, "r") as f:
            out_str = f.read()

        energy = float(out_str.split("HEAT OF FORMATION =")[1].split()[0])

        return {
            "out_path": out_path,
            "energy": energy
        }