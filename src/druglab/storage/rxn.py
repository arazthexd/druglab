from __future__ import annotations
from typing import List, Any, Tuple, Type, Dict

from collections import defaultdict
from tqdm import tqdm

import numpy as np

from rdkit import Chem
from rdkit.Chem.rdChemReactions import ChemicalReaction as Rxn

from .io import load_rxns_file
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
        
        if overwrite:
            self.mstores: List[List[MolStorage]] = [
                [MolStorage() for _ in rxn.GetReactants()] for rxn in self
            ]

        mols = MolStorage(mols) if isinstance(mols, list) else mols

        mol_idx = [
            idx for idx, mol in enumerate(tqdm(mols)) if any(
                mol.HasSubstructMatch(rt)
                for rxn in self 
                for rt in rxn.GetReactants()
            )
        ]

        idx2rxnr: Dict[int, List] = defaultdict(list)
        for rxnid, (rxn, stores) in enumerate(zip(self, tqdm(self.mstores))):
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
        for i, (rxn, stores) in enumerate(zip(self, self.mstores)):
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