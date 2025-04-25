from typing import List, Tuple, Optional
import os
import pathlib
import glob
import subprocess
from copy import deepcopy
from dataclasses import dataclass, field
import shutil

from rdkit import Chem

DATA_UNIQUE_CODE_LEN = 6
import random, string
from .config import MOPACConfig, MOPACMozymeConfig

def generate_random_str(n: int):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))

@dataclass
class MOPACInterface:
    mopac_path: str = field(default="auto")
    output_dir: str = field(default="auto")
    config: MOPACConfig = field(default_factory=MOPACConfig)

    def __post_init__(self):
        self._resolve_paths()
        self.init_config = deepcopy(self.config)

    def _resolve_paths(self):
        """Resolve MOPAC path and output directory"""
        if self.mopac_path == "auto":
            self.mopac_path = self._find_mopac_executable()
        if self.output_dir == "auto":
            self.output_dir = os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)

    def _find_mopac_executable(self) -> str:
        exe_names = ['mopac', 'MOPAC', 'mopac.exe', 'MOPAC.exe']
        for name in exe_names:
            path = shutil.which(name)
            if path: return path
        raise FileNotFoundError("MOPAC executable not found in PATH")

    def run_job(self, config: Optional[MOPACConfig] = None, 
               base_dir: Optional[str] = None, debug: bool = False) -> Tuple[str, str]:
        config = config or self.config
        base_dir = base_dir or self.output_dir
        path = self._generate_input_path(base_dir)

        with open(path, "w") as f:
            f.write(config.get_config_str())

        self._execute_mopac(path, debug)
        return (
            str(path.with_suffix('.out')),
            str(path.with_suffix('.arc'))
        )

    def _generate_input_path(self, base_dir: str) -> pathlib.Path:
        base_path = pathlib.Path(base_dir)
        while True:
            random_name = generate_random_str(8) + ".mop"
            candidate = base_path / random_name
            if not candidate.exists():
                return candidate

    def _execute_mopac(self, path: pathlib.Path, debug: bool):
        try:
            subprocess.run(
                [self.mopac_path, str(path.name)],
                cwd=str(path.parent),
                check=True,
                capture_output=not debug
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"MOPAC calculation failed: {e.stderr}")