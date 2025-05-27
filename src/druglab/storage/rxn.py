from __future__ import annotations
from typing import List, Any, Tuple, Type, Dict

from collections import defaultdict
from tqdm import tqdm
import functools
import h5py

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdChemReactions as rdRxn
from rdkit.Chem.rdChemReactions import ChemicalReaction as Rxn

from ..io import load_rxns_file
from ..featurize import BaseFeaturizer
from .base import BaseStorage
from .mol import MolStorage

class RxnStorage(BaseStorage):
    def __init__(self, 
                 rxns: List[Rxn] = None, 
                 fdtype: Type[np.dtype] = np.float32,
                 feats: np.ndarray = None,
                 fnames: List[str] = None,
                 featurizers: List[BaseFeaturizer | dict] = None):
        super().__init__(rxns, 
                         fdtype=fdtype, 
                         feats=feats, 
                         fnames=fnames, 
                         featurizers=featurizers)

        self.mstores: List[List[MolStorage]] = [
            [MolStorage() for _ in rxn.GetReactants()] for rxn in self
        ]

    def load_rxns(self, filename: str):
        rxns = load_rxns_file(filename)

        newstore = RxnStorage(rxns)

        for featurizer in self.featurizers:
            newstore.featurize(featurizer)

        self.extend(newstore)

    def extend(self, rxns: RxnStorage):
        super().extend(rxns)
        self.mstores.extend(rxns.mstores)

    def subset(self, idx, inplace = False):
        out: RxnStorage | None = super().subset(idx, inplace)
        mstores = [self.mstores[i] for i in idx]
        if inplace:
            self.mstores = mstores
            return

        out.mstores = mstores
        return out

    def add_mols(self, 
                 mols: List[Chem.Mol] | MolStorage,
                 overwrite: bool = False) -> Dict[int, List[Tuple[int, int]]]:
        
        mols = MolStorage(mols) if isinstance(mols, list) else mols
        
        molstore_init = functools.partial(MolStorage, 
                                          fnames=mols.fnames, 
                                          featurizers=mols.featurizers)
        
        if overwrite:
            self.mstores: List[List[MolStorage]] = [
                [molstore_init() for _ in rxn.GetReactants()] for rxn in self
            ]

        mol_idx = [
            idx 
            for idx, mol in enumerate(tqdm(mols, desc="adding mols to rxns")) 
            if any(
                mol.HasSubstructMatch(rt) 
                for rxn in self 
                for rt in rxn.GetReactants()
            )
        ]

        idx2rxnr: Dict[int, List] = defaultdict(list)
        for rxnid, (rxn, stores) in enumerate(
            zip(self, tqdm(self.mstores, desc="matching mols to rxns"))
        ):
            rxn: Rxn
            for rid, (reactant_template, 
                      store) in enumerate(zip(rxn.GetReactants(), stores)):
                store: MolStorage
                matched_idx = [
                    idx for idx in mol_idx
                    if mols[idx].HasSubstructMatch(reactant_template)
                ]
                store.extend(mols.subset(matched_idx))
                [idx2rxnr[idx].append((rxnid, rid)) for idx in matched_idx]

        return dict(idx2rxnr)
    
    def clean(self) -> None:
        
        rxn_remove_idx = []
        rxn_smas = []
        for i, (rxn, stores) in enumerate(zip(self, self.mstores)):
            sma = rdRxn.ReactionToSmarts(rxn)
            if sma in rxn_smas:
                rxn_remove_idx.append(i)
                continue
            rxn_smas.append(sma)

            for store in stores:
                if len(store) == 0:
                    rxn_remove_idx.append(i)
                    break
        
        rxn_keep_idx = [i for i in range(len(self)) if i not in rxn_remove_idx]
        self.subset(rxn_keep_idx, inplace=True)

    def match_mols(self, 
                   mols: List[Chem.Mol] | MolStorage) \
                    -> Dict[int, List[Tuple[int, int]]]:
        
        mols = MolStorage(mols) if isinstance(mols, list) else mols

        idx2rxnr = defaultdict(list)
        for rxnid, rxn in enumerate(tqdm(self, desc="matching mols to rxns")):
            for rid, reactant in enumerate(rxn.GetReactants()):
                mol_idx = []
                for mid, mol in enumerate(mols):
                    mol: Chem.Mol
                    if mol.HasSubstructMatch(reactant):
                        mol_idx.append(mid)
                        idx2rxnr[mid].append((rxnid, rid))

        return dict(idx2rxnr)   

    def _serialize_object(self, obj):
        return rdRxn.ReactionToSmarts(obj)

    def _unserialize_object(self, obj):
        return rdRxn.ReactionFromSmarts(obj.decode())
    
    def save_dict(self, save_mols: bool = False, save_confs: bool = False):
        d = super().save_dict()
        if save_mols:
            for i, rxn in enumerate(self.objects):
                rxn: Rxn
                for j in range(rxn.GetNumReactantTemplates()):
                    predix = f"mstores/rxn{i}/r{j}/"
                    for k, v in self.mstores[i][j].save_dict(save_confs).items():
                        d[predix + k] = v
        return d

    def save(self, 
             dst: str | h5py.Dataset, 
             close: bool = True,
             save_mols: bool = False,
             save_confs: bool = False):
        d = self.save_dict(save_mols, save_confs)
        f = self._save(d, dst, close=False)
        if save_mols:
            for i, rxn in enumerate(self.objects):
                rxn: Rxn
                for j in range(rxn.GetNumReactantTemplates()):
                    grp = f[f"mstores/rxn{i}/r{j}"]
                    for k, fzer in enumerate(self.mstores[i][j].featurizers):
                        grp[f"featurizer{k}"].attrs["_name_"] = fzer.__class__.__name__
        if close:
            f.close()
        else:
            return f
    
    def _load(self, d: dict | Any):
        super()._load(d)
        if "mstores" in d:
            self.mstores = []
            for i, rxn in enumerate(self.objects):
                rxn: Rxn
                rxngrp = d[f"mstores/rxn{i}"]
                mstores = []
                for j in range(rxn.GetNumReactantTemplates()):
                    grp = rxngrp[f"r{j}"]
                    mstore = MolStorage()
                    mstore._load(grp)
                    mstores.append(mstore)
                self.mstores.append(mstores)
    
    # def load(self, path, close = True):
    #     f = super().load(path, close=False)

    #     if "mstores" in f:
    #         self.mstores = []
    #         for i, rxn in enumerate(self.objects):
    #             rxn: Rxn
    #             rxngrp = f[f"mstores/rxn{i}"]
    #             mstores = []
    #             for j in range(rxn.GetNumReactantTemplates()):
    #                 grp = rxngrp[f"r{j}"]
    #                 mstore = MolStorage()
    #                 mstore.objects = [Chem.JSONToMols(obj)[0] 
    #                                   for obj in grp["mols"]]
    #                 mstore.fnames = grp["fnames"][:]
    #                 mstore.feats = grp["feats"][:]

    #                 if "cstore" in grp:
    #                     mstore: MolStorage
    #                     mstore.initiate_cstores()
    #                     for cstore in mstore.cstores:
    #                         cstore.fnames = grp["cstore"]["cfnames"][:]
    #                         cstore.feats = grp["cstore"]["cfeats"][:]
    #                 mstores.append(mstore)
                
    #             self.mstores.append(mstores)

    #     if close:
    #         f.close()
    #     else:
            # return f