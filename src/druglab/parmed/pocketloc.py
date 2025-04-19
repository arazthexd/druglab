import numpy as np
from rdkit import Chem
from typing import Dict, Any, Union, List

class PocketLocation:
    def __init__(self, method: str, **kwargs):
        self.params: Dict[str, Any] = {}
        self.loc_type = method
        
        init_methods = {
            "point": self._init_point,
            "box2p": self._init_box2p,
            "boxcxyz": self._init_boxcxyz,
            "ligand": self._init_ligand,
            "residues": self._init_residues
        }
        
        if method not in init_methods:
            raise ValueError("Method must be one of: 'point', 'box2p', 'boxcxyz', 'ligand', 'residues'")
        
        init_methods[method](**kwargs)

    def point_is_included(self, point: Union[np.ndarray, List[float]]) -> bool:
        point = np.asarray(point)
        check_methods = {
            "point": self._point_is_included_point,
            "ligand": self._point_is_included_ligand,
        }
        
        if self.loc_type not in check_methods:
            raise NotImplementedError(f"Point inclusion check not implemented for {self.loc_type}")
        
        return check_methods[self.loc_type](point)

    def get_params(self, param_type: str) -> Dict[str, Any]:
        if param_type in self.params:
            return self.params[param_type]

        for source_format in self.params:
            try:
                self._convert_format(source_format, param_type)
                return self.params[param_type]
            except NotImplementedError:
                continue
        
        raise ValueError(f"Could not convert to {param_type} format")

    def _convert_format(self, source_format: str, target_format: str):
        if source_format == "ligand" and target_format == "boxcxyz":
            coords = self.params["ligand"]["c"]
            radius = self.params["ligand"]["r"]
            lig_center = (np.max(coords, axis=0) + np.min(coords, axis=0)) / 2
            size = np.max(coords, axis=0) - np.min(coords, axis=0) + radius
            self.params["boxcxyz"] = {"c": lig_center, "size": size}
        elif source_format == "point" and target_format == "boxcxyz":
            point = self.params["point"]["p"]
            radius = self.params["point"]["r"]
            self.params["boxcxyz"] = {"c": point, "size": np.array([radius]*3)}
        else:
            raise NotImplementedError(f"Conversion from {source_format} to {target_format} not implemented")

    def _init_point(self, point: Union[np.ndarray, List[float]], radius: float = 10):
        self.params["point"] = {
            "p": np.asarray(point),
            "r": float(radius)
        }

    def _init_box2p(self, point1: Union[np.ndarray, List[float]], 
                   point2: Union[np.ndarray, List[float]]):
        self.params["box2p"] = {
            "p1": np.asarray(point1),
            "p2": np.asarray(point2)
        }

    def _init_boxcxyz(self, center: Union[np.ndarray, List[float]], 
                     xyz_size: Union[np.ndarray, List[float]]):
        self.params["boxcxyz"] = {
            "c": np.asarray(center),
            "size": np.asarray(xyz_size)
        }

    def _init_ligand(self, ligand: Union[str, Chem.Mol], radius: float = 10):
        if isinstance(ligand, str):
            if ligand.endswith(".pdb"):
                mol = Chem.MolFromPDBFile(ligand)
            elif ligand.endswith(".mol2"):
                mol = Chem.MolFromMol2File(ligand)
            elif ligand.endswith(".sdf"):
                mol = next(Chem.SDMolSupplier(ligand))
            else:
                raise ValueError("Ligand file must be .pdb, .mol2 or .sdf")
        elif isinstance(ligand, Chem.Mol):
            mol = ligand
        else:
            raise ValueError("Ligand must be file path or RDKit Mol object")
        
        coords = mol.GetConformer().GetPositions()
        self.params["ligand"] = {
            "c": coords,
            "r": float(radius),
            "l": mol
        }

    def _init_residues(self, protein_pdb: str, res_ids: List[int]):
        self.params["residues"] = {
            "pdb": protein_pdb,
            "ids": res_ids
        }

    # Point inclusion check methods
    def _point_is_included_point(self, point: np.ndarray) -> bool:
        return np.linalg.norm(point - self.params["point"]["p"]) <= self.params["point"]["r"]

    def _point_is_included_ligand(self, point: np.ndarray) -> bool:
        distances = np.linalg.norm(self.params["ligand"]["c"] - point, axis=1)
        return np.min(distances) <= self.params["ligand"]["r"]
