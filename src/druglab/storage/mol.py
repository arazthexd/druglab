from __future__ import annotations
from typing import List, Any, Tuple, Type, Dict
import mpire

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

from ..io import load_mols_file
from ..featurize import BaseFeaturizer
from .base import BaseStorage

class ConformerStorage(BaseStorage):
    pass

class MolStorage(BaseStorage):
    def __init__(self, 
                 mols: List[Chem.Mol] = None, 
                 fdtype: Type[np.dtype] = np.float32,
                 feats: np.ndarray = None,
                 fnames: List[str] = None,
                 featurizers: List[BaseFeaturizer | dict] = None):
        super().__init__(mols, 
                         fdtype=fdtype, 
                         feats=feats, 
                         fnames=fnames, 
                         featurizers=featurizers)
        self.objects: List[Chem.Mol]
        
        self.cstores: List[ConformerStorage] = [ConformerStorage() 
                                                for _ in self]

    def load_mols(self, filename: str):
        mols = load_mols_file(filename)

        newstore = MolStorage(mols)

        for featurizer in self.featurizers:
            newstore.featurize(featurizer)

        self.extend(newstore)
    
    def extend(self, mols: MolStorage):
        super().extend(mols)
        self.cstores.extend(mols.cstores)

    def subset(self, idx, inplace = False):
        out: MolStorage | None = super().subset(idx, inplace)
        cstores = [self.cstores[i] for i in idx]
        if inplace:
            self.cstores = cstores
            return

        out.cstores = cstores
        return out
    
    def clean(self):
        smiles = []
        idx_keep = []
        for i, mol in enumerate(self):
            smi = Chem.MolToSmiles(mol)
            if smi in smiles:
                continue
            smiles.append(Chem.MolToSmiles(mol))
            idx_keep.append(i)
        
        self.feats = self.feats[idx_keep]

        counter = 0
        for i in range(len(self)):
            if i not in idx_keep:
                del self[i-counter]
                counter += 1

    def initiate_conformers(self):
        self.cstores = [ConformerStorage([
            conf for conf in mol.GetConformers()]) for mol in self.objects]

    def generate_conformers(self, 
                            nconfs: int = 1, 
                            nworkers: int = 8,
                            params: rdDistGeom.EmbedParameters = None,
                            optimize: bool = False,
                            cluster: bool = False,
                            cluster_tol: float = 0.4):

        if params is None:
            params = rdDistGeom.ETKDGv3()

        if cluster:
            raise NotImplementedError()

        def create_confs(mol: Chem.Mol):
            confs = rdDistGeom.EmbedMultipleConfs(mol, nconfs, params)
            if optimize:
                rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol)
            return mol
        
        with mpire.WorkerPool(nworkers) as pool:
            mols = pool.map(create_confs, self.objects, progress_bar=True)
        
        self.objects = mols
        self.initiate_conformers()
        

    def featurize_conformers(self, featurizer: BaseFeaturizer):
        cstore = ConformerStorage()
        [cstore.extend(cs) for cs in self.cstores]
        cstore.featurize(featurizer)
        
        newcs = []
        id1, id2 = 0, len(self.cstores[0])
        for cs in self.cstores:
            newcs.append(cs[id1:id2])
            id1 = id2
            id2 += len(cs)
        self.cstores = newcs

