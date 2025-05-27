from typing import Type, List
import h5py

import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdChemReactions as rdRxn
from rdkit.Chem.rdChemReactions import ChemicalReaction as Rxn

from ..storage import BaseStorage, MolStorage, RxnStorage
from ..featurize import BaseFeaturizer
from .route import SynthesisRoute, ActionTypes, _SequenceMember
from .featurize import SynRouteFeaturizer
    
class SynRouteStorage(BaseStorage):
    def __init__(self, 
                 routes: List[SynthesisRoute] = None, 
                 fdtype: Type[np.dtype] = np.float32,
                 feats: np.ndarray = None,
                 fnames: List[str] = None,
                 featurizers: List[BaseFeaturizer | dict] = None):
        
        if routes is None:
            routes = []

        self._psmi: List[str] = []
        self._rsmi: List[str] = []
        self._rxnsmi: List[str] = []
        self._ps: List[Chem.Mol] = []
        self._rs: List[Chem.Mol] = []
        self._rxns: List[Rxn] = []

        self.seqs: List[List[_SequenceMember]] = []

        for route in routes:
            seq: List[_SequenceMember] = []
            _cpids: List[Chem.Mol] = []
            for mem in route.seq:
                if mem.type == ActionTypes.START:
                    seq.append(_SequenceMember(ActionTypes.START))
                
                elif mem.type == ActionTypes.REACTANT:
                    smi = Chem.MolToSmiles(route.reactants[mem.idx])
                    try:
                        idx = self._rsmi.index(smi)
                    except ValueError:
                        idx = len(self._rsmi)
                        self._rsmi.append(smi)
                        self._rs.append(route.reactants[mem.idx])
                    seq.append(_SequenceMember(
                        type=ActionTypes.REACTANT, 
                        idx=idx
                    ))
                
                elif mem.type == ActionTypes.REACTION:
                    smi = rdRxn.ReactionToSmiles(route.reactions[mem.idx])
                    try:
                        idx = self._rxnsmi.index(smi)
                    except ValueError:
                        idx = len(self._rxnsmi)
                        self._rxnsmi.append(smi)
                        self._rxns.append(route.reactions[mem.idx])
                    seq.append(_SequenceMember(
                        type=ActionTypes.REACTION, 
                        idx=idx
                    ))
                
                elif mem.type == ActionTypes.PRODUCT:
                    smi = Chem.MolToSmiles(route.products[mem.idx])
                    try:
                        idx = self._psmi.index(smi)
                    except ValueError:
                        idx = len(self._psmi)
                        self._psmi.append(smi)
                        self._ps.append(route.products[mem.idx])
                    seq.append(_SequenceMember(
                        type=ActionTypes.PRODUCT, 
                        idx=idx
                    ))
                    _cpids.append(idx)
                
                elif mem.type == ActionTypes.USEPROD:
                    seq.append(_SequenceMember(
                        type=ActionTypes.USEPROD,
                        idx=_cpids.pop()
                    ))
                
                elif mem.type == ActionTypes.END:
                    seq.append(_SequenceMember(
                        type=ActionTypes.END
                    ))

            self.seqs.append(seq)

        self.rstore = MolStorage(self._rs)
        self.pstore = MolStorage(self._ps)
        self.rxnstore = RxnStorage(self._rxns)

        super().__init__(routes, 
                         fdtype=fdtype, 
                         feats=feats, 
                         fnames=fnames, 
                         featurizers=featurizers)

    def featurize(self, 
                  featurizer: SynRouteFeaturizer, 
                  overwrite: bool = False, 
                  n_workers: int = 1):
        super().featurize(featurizer, overwrite, n_workers)
        self.pstore.featurize(featurizer.pfeaturizer, overwrite, n_workers)
        self.rstore.featurize(featurizer.rfeaturizer, overwrite, n_workers)
        self.rxnstore.featurize(featurizer.rxnfeaturizer, overwrite, n_workers)

    def subset(self, 
               idx: List[int] | np.ndarray, 
               inplace: bool = False) -> BaseStorage | None:
        new: None | SynRouteStorage = super().subset(idx, inplace)
        if inplace:
            self.seqs = [self.seqs[i] for i in idx]
        return new

    def visualize(self, idx: int):
        vislist = []
        vl = []
        rxns: List[Rxn] = []
        for mem in self.seqs[idx]:
            if mem.type == ActionTypes.REACTANT:
                vl.append(self.rstore[mem.idx])
            elif mem.type in [ActionTypes.PRODUCT, ActionTypes.USEPROD]:
                vl.append(self.pstore[mem.idx])
            elif mem.type == ActionTypes.REACTION:
                rxns.append(self.rxnstore[mem.idx])
            
            if mem.type == ActionTypes.PRODUCT: 
                vislist.append(vl)
                vl = []
        
        l = max([rxn.GetNumReactantTemplates() for rxn in rxns]) + 1
        vislist = [vl+[None]*(l-len(vl)) for vl in vislist]
        view = Draw.MolsToGridImage(
            mols=[v for vl in vislist for v in vl],
            molsPerRow=l,
            subImgSize=(300, 150)
        )
        return view
    
    def save(self, 
             dst: str | h5py.Dataset | h5py.Group, 
             close: bool = True):
        if isinstance(dst, str):
            dst = h5py.File(dst, "w")
        dst.create_group("rstore")
        dst.create_group("pstore")
        dst.create_group("rxnstore")
        dst.create_group("seqs")
        dst.attrs["_name_"] = self.__class__.__name__

        self.rstore.save(dst["rstore"], close=False)
        self.pstore.save(dst["pstore"], close=False)
        self.rxnstore.save(dst["rxnstore"], close=False)

        for i, seq in enumerate(self.seqs):
            arr = np.array([[m.type, m.idx] if m.idx is not None
                            else [m.type, -1] for m in seq])
            dst[f"seqs/{i}"] = arr
        
        if close:
            dst.close()
        else:
            return dst
        
    def _load(self, grp: h5py.Group):
        self.rstore = MolStorage()
        self.rstore._load(grp["rstore"])
        self.pstore = MolStorage()
        self.pstore._load(grp["pstore"])
        self.rxnstore = RxnStorage()
        self.rxnstore._load(grp["rxnstore"])

        self.seqs = [None] * len(grp["seqs"].keys())
        for i in grp["seqs"].keys():
            i = int(i)
            arr = grp[f"seqs/{i}"]
            seq = []
            for j in range(arr.shape[0]):
                seq.append(_SequenceMember(arr[j, 0], arr[j, 1]))
            self.seqs[i] = seq
        
        for i, seq in enumerate(self.seqs):
            route = SynthesisRoute(
                seq=seq,
                reactants=[self.rstore[j] for j in range(len(seq)) 
                           if seq[j].type == ActionTypes.REACTANT],
                products=[self.pstore[j] for j in range(len(seq)) 
                          if seq[j].type == ActionTypes.PRODUCT],
                reactions=[self.rxnstore[j] for j in range(len(seq)) 
                           if seq[j].type == ActionTypes.REACTION]
            )
            self.objects.append(route)
        
    @property
    def routes(self) -> List[SynthesisRoute]:
        return self.objects

        
