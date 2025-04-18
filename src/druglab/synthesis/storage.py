from typing import Type, List

import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw

from ..storage import BaseStorage, MolStorage, RxnStorage
from ..featurize import BaseFeaturizer
from .route import SynthesisRoute, ActionTypes
from .featurize import SynRouteFeaturizer
    
class SynRouteStorage(BaseStorage):
    def __init__(self, 
                 routes: List[SynthesisRoute] = None, 
                 fdtype: Type[np.dtype] = np.float32,
                 feats: np.ndarray = None,
                 fnames: List[str] = None,
                 featurizers: List[BaseFeaturizer | dict] = None):
        super().__init__(routes, 
                         fdtype=fdtype, 
                         feats=feats, 
                         fnames=fnames, 
                         featurizers=featurizers)
        
        self.pstore: MolStorage = MolStorage()
        self.rstore: MolStorage = MolStorage()
        self.rxnstore: RxnStorage = RxnStorage()

        for route in self.routes:
            
            for mem in route.seq:
                if mem.type == ActionTypes.REACTION:
                    mem.idx += len(self.rxnstore)
                elif mem.type == ActionTypes.REACTANT:
                    mem.idx += len(self.rstore)
                elif mem.type in [ActionTypes.PRODUCT, ActionTypes.USEPROD]:
                    mem.idx += len(self.pstore)

            self.pstore.extend(MolStorage(route.products))
            self.rstore.extend(MolStorage(route.reactants))
            self.rxnstore.extend(RxnStorage(route.reactions))

    def featurize(self, 
                  featurizer: SynRouteFeaturizer, 
                  overwrite: bool = False, 
                  n_workers: int = 1):
        super().featurize(featurizer, overwrite, n_workers)
        self.pstore.featurize(featurizer.pfeaturizer, overwrite, n_workers)
        self.rstore.featurize(featurizer.rfeaturizer, overwrite, n_workers)
        self.rxnstore.featurize(featurizer.rxnfeaturizer, overwrite, n_workers)

    def visualize(self, idx: int):
        vislist = []
        vl = []
        for mem in self.routes[idx].seq:
            if mem.type == ActionTypes.REACTANT:
                vl.append(self.rstore[mem.idx])
            elif mem.type in [ActionTypes.PRODUCT, ActionTypes.USEPROD]:
                vl.append(self.pstore[mem.idx])
            
            if mem.type == ActionTypes.PRODUCT: 
                vislist.append(vl)
                vl = []
        
        l = max([rxn.GetNumReactantTemplates() 
                 for rxn in self.routes[idx].reactions])+1
        vislist = [vl+[None]*(l-len(vl)) for vl in vislist]
        view = Draw.MolsToGridImage(
            mols=[v for vl in vislist for v in vl],
            molsPerRow=l,
            subImgSize=(300, 150)
        )
        return view
    
    @property
    def routes(self) -> List[SynthesisRoute]:
        return self.objects

        
