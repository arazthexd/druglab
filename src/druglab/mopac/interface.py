from typing import List, Tuple, Optional
import os
import pathlib
import glob
import subprocess
from copy import deepcopy
from dataclasses import dataclass, field

from rdkit import Chem

DATA_UNIQUE_CODE_LEN = 6
import random, string
from config import MOPACConfig, MOPACMozymeConfig

def generate_random_str(n: int):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


@dataclass
class MOPACInterface:
    config: MOPACConfig = field(default_factory=MOPACConfig)
    
    def __post_init__(self):
        self.init_config = deepcopy(self.config)
    
    def reset_config(self) -> None:
        self.config = deepcopy(self.init_config)
    
    def update_config(self, 
                     molecules: Chem.Mol | List[Chem.Mol]) -> None:
        if isinstance(molecules, Chem.Mol):
            self.config.add_molecule(molecules)
        elif isinstance(molecules, list):
            for mol in molecules:
                self.config.add_molecule(mol)
    
    def run_job(self, 
               base_dir: str, 
               debug: bool = False) -> Tuple[str, str]:
        path = self.generate_random_input_file(
            base_dir=base_dir, 
            key_length=DATA_UNIQUE_CODE_LEN
        )

        if debug:
            print("MOPAC input file written:", path)

        self.write_and_run_mopac(path, self.config, debug)

        out_path = pathlib.Path(path).with_suffix('.out').as_posix()
        arc_path = pathlib.Path(path).with_suffix('.arc').as_posix()
        
        return out_path, arc_path

    @staticmethod
    def write_and_run_mopac(path: str | os.PathLike, 
                           mopac_config: MOPACConfig, 
                           debug: bool = False) -> None:

        path = pathlib.Path(path)
        cur_dir = pathlib.Path.cwd()
        with open(path, "w") as f:
            f.write(mopac_config.get_config_str())

        os.chdir(path.parent)
        try:
            subprocess.run(
                ["mopac", path.name], 
                capture_output=not debug,
                check=True
            )
        finally:
            os.chdir(cur_dir)
    
    @staticmethod
    def generate_random_input_file(base_dir: str, 
                                 key_length: int) -> str:
        base_dir = pathlib.Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        
        while True:
            random_key = generate_random_str(key_length)
            candidate = base_dir / f"{random_key}.mop"
            
            if not candidate.exists():
                return candidate.as_posix()