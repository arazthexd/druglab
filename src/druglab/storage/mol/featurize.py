from typing import Optional, Any, List

import numpy as np
from rdkit import Chem

from ...featurize import BaseFeaturizer
from ..featurize import BasicStorageFeaturizerWrapper
from .mol import MolStorage

class BasicMoleculeFeaturizerWrapper(BasicStorageFeaturizerWrapper):
    def __init__(self, 
                 featurizer: BaseFeaturizer,
                 feature_key: Optional[str] = None,
                 n_processes = 1):
        super().__init__(featurizer, ['molecules'], feature_key, n_processes)

class BasicConformerFeaturizerWrapper(BasicStorageFeaturizerWrapper):
    def __init__(self, 
                 featurizer: BaseFeaturizer,
                 feature_key: Optional[str] = None,
                 n_processes = 1):
        super().__init__(featurizer, ['molecules'], feature_key, n_processes)

    def compute_features(self, object_dict, context_data):
        mol: Chem.Mol = object_dict['molecules']
        confmols = [Chem.Mol(mol, confId=conf.GetId()) 
                    for conf in mol.GetConformers()]
        conffeats = [
            super().compute_features({'molecules': confmol}, context_data)
            for confmol in confmols
        ]
        return np.vstack(conffeats)
    
    def apply_featurization(self, 
                            storage: MolStorage,
                            features_array: np.ndarray,
                            context_data: Any,
                            success_indices: Optional[List[int]] = None) \
                                -> None:
        confstorage = storage.get_conformers_as_storage()
        super().apply_featurization(confstorage, features_array, 
                                    context_data, success_indices)
        storage._conformer_features = confstorage.features
