from typing import List

from rdkit import Chem
from rdkit.Chem import (
    rdDistGeom, rdForceFieldHelpers, 
    TorsionFingerprints,
    rdMolAlign
)
from rdkit.ML.Cluster import Butina # type: ignore

from ..featurize import BaseFeaturizer, CompositeFeaturizer
from .abstract import BasePreparation

class MoleculePreparation(BasePreparation):
    def __init__(self,
                 addhs: bool = False,
                 removehs: bool = False,
                 cgen: bool = False, 
                 cgen_n: int = 1, 
                 cgen_maxatts: int = None,
                 cgen_params: rdDistGeom.EmbedParameters = None,
                 copt: bool = False,
                 copt_nthreads: int = 1,
                 copt_maxits: int = 200,
                 cclust: bool = False,
                 cclust_tol = 0.3,
                 cclust_afteropt: bool = True,
                 calign: bool = False):

        if cgen_params is None:
            cgen_params = rdDistGeom.ETKDGv3()

        if cgen_maxatts is None:
            cgen_maxatts = cgen_n * 4

        self.addhs = addhs
        self.removehs = removehs
        self.cgen = cgen
        self.cgen_n = cgen_n
        self.cgen_params = cgen_params
        self.cgen_params.maxIterations = cgen_maxatts
        self.copt = copt
        self.copt_nthreads = copt_nthreads
        self.copt_maxits = copt_maxits
        self.cclust = cclust
        self.cclust_tol = cclust_tol
        self.cclust_afteropt = cclust_afteropt
        self.calign = calign

    def _cluster(self, mol: Chem.Mol) -> Chem.Mol:
        tfds = TorsionFingerprints.GetTFDMatrix(mol)
        clusts = Butina.ClusterData(tfds, 
                                    mol.GetNumConformers(), 
                                    self.cclust_tol, 
                                    isDistData=True, 
                                    reordering=True)
        clustcs = [clust[0] for clust in clusts]
        for cid in range(mol.GetNumConformers()):
            if cid not in clustcs:
                mol.RemoveConformer(cid)
        return mol

    def prepare(self, mol: Chem.Mol) -> Chem.Mol:
        if self.addhs:
            addcs = False
            if mol.GetNumConformers() > 0:
                if mol.GetConformer().Is3D():
                    addcs = True
            mol = Chem.AddHs(mol, addCoords=addcs)

        if self.cgen:
            rdDistGeom.EmbedMultipleConfs(mol, self.cgen_n, self.cgen_params)

        if self.cclust and not self.cclust_afteropt:
            mol = self._cluster(mol)

        if self.copt:
            rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, 
                                                          self.copt_nthreads, 
                                                          self.copt_maxits)
        
        if self.cclust and self.cclust_afteropt:
            mol = self._cluster(mol)

        if self.calign:
            rdMolAlign.AlignMolConformers(mol)

        if self.removehs: 
            mol = Chem.RemoveHs(mol)

        confs = mol.GetConformers()
        for i in range(mol.GetNumConformers()):
            confs[i].SetId(i)
        
        return mol
        