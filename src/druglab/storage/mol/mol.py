from __future__ import annotations
from typing import List, Any, Dict, Optional, Union
import logging

import numpy as np
import h5py
from rdkit import Chem

from ..base import BaseStorage, StorageFeatures, StorageMetadata

logger = logging.getLogger(__name__)


class MolStorage(BaseStorage):
    """Storage for RDKit molecules with conformer support."""
    
    @property
    def required_object_keys(self) -> List[str]:
        return ['molecules']
    
    @property
    def save_dtypes(self) -> Dict[str, type]:
        return {'molecules': h5py.string_dtype()}
    
    def __init__(self,
                 molecules: Optional[List[Chem.Mol]] = None,
                 features: Optional[StorageFeatures] = None,
                 metadata: Optional[StorageMetadata] = None,
                 conformer_features: Optional[StorageFeatures] = None):
        """
        Initialize MolStorage.
        
        Args:
            molecules: List of RDKit molecule objects
            features: StorageFeatures container for molecules
            metadata: Storage metadata
            conformer_features: StorageFeatures container for conformers
        """
        # Convert molecules to proper format for BaseStorage
        objects = None
        if molecules is not None:
            objects = {'molecules': molecules}
            
        super().__init__(
            objects=objects,
            features=features,
            metadata=metadata
        )
        
        # Conformer-specific features
        self._conformer_features = conformer_features or StorageFeatures()
    
    @property
    def molecules(self) -> List[Chem.Mol]:
        """Get molecules (alias for objects['molecules'])."""
        return self._objects['molecules']
    
    @property 
    def conformer_features(self) -> StorageFeatures:
        """Get conformer features container."""
        return self._conformer_features
    
    @property
    def num_conformers(self) -> int:
        """Get total number of conformers across all molecules."""
        return sum(self.num_conformers_per_molecule)
    
    @property
    def num_conformers_per_molecule(self) -> List[int]:
        """Get number of conformers for each molecule."""
        return [mol.GetNumConformers() if mol else 0 
                for mol in self.molecules]
    
    def add_molecule(self, molecule: Chem.Mol) -> None: # TODO: Auto feat
        """Add a single molecule to storage."""
        self._objects['molecules'].append(molecule)
    
    def add_molecules(self, molecules: List[Chem.Mol]) -> None: 
        """Add multiple molecules to storage."""
        self._objects['molecules'].extend(molecules) # TODO: Auto feat

    def get_conformers_as_molecules(self) -> List[Chem.Mol]:
        """Extract all conformers as separate molecule objects."""
        conformer_mols = []
        for mol in self.molecules:
            if mol is None:
                continue
            for conf_id in range(mol.GetNumConformers()):
                # Create new molecule with single conformer
                conf_mol = Chem.Mol(mol)
                conf_mol.RemoveAllConformers()
                conf_mol.AddConformer(mol.GetConformer(conf_id), assignId=True)
                conformer_mols.append(conf_mol)
        return conformer_mols
    
    def get_conformers_as_storage(self) -> MolStorage:
        """Create a new MolStorage with each conformer as a separate molecule."""
        conformer_mols = self.get_conformers_as_molecules()
        
        # Create new storage with conformer features as main features
        conformer_storage = self.__class__(
            molecules=conformer_mols,
            features=self._conformer_features,
            metadata=self._metadata.copy()
        )
        
        return conformer_storage
    
    def _split_conformer_features_by_molecule(self, 
                                              features: np.ndarray) \
                                                -> List[np.ndarray]:
        """Split conformer features back into per-molecule groups."""
        if len(features) != self.num_conformers:
            raise ValueError(f"Feature array length {len(features)} doesn't "
                             f"match total conformers {self.num_conformers}")
        
        split_features = []
        start_idx = 0
        
        for n_conf in self.num_conformers_per_molecule:
            if n_conf == 0:
                split_features.append(np.empty((0, features.shape[1]), 
                                                dtype=features.dtype))
            else:
                end_idx = start_idx + n_conf
                split_features.append(features[start_idx:end_idx])
                start_idx = end_idx
        
        return split_features
    
    def clean_molecules(self,
                        remove_none: bool = True,
                        remove_duplicates: bool = False,
                        sanitize: bool = True,
                        remove_conformers: bool = False) -> List[int]:
        """
        Clean molecules and return indices of kept molecules.
        
        Args:
            remove_none: Remove None molecules
            remove_duplicates: Remove duplicate molecules (by SMILES)
            sanitize: Sanitize molecules
            
        Returns:
            List of indices of molecules that were kept
        """
        keep_indices = []
        seen_smiles = set() if remove_duplicates else None
        
        for i, mol in enumerate(self.molecules):
            # Remove None molecules
            if remove_none and mol is None:
                logger.debug(f"Removing None molecule at index {i}")
                continue

            if remove_none and mol.GetNumAtoms() == 0:
                logger.debug(f"Removing empty molecule at index {i}")
                continue
            
            # Sanitize if requested
            if sanitize and mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except Exception as e:
                    logger.warning(f"Failed to sanitize molecule at index {i}:" 
                                   f" {e}")
                    if remove_none:
                        continue
            
            # Check for duplicates
            if remove_duplicates and mol is not None:
                try:
                    smiles = Chem.MolToSmiles(mol)
                    if smiles in seen_smiles:
                        logger.debug(f"Removing duplicate molecule at index {i}")
                        continue
                    seen_smiles.add(smiles)
                except Exception as e:
                    logger.warning(f"Could not generate SMILES for molecule "
                                 f"at index {i}: {e}")
                    if remove_none:
                        continue

            if remove_conformers:
                mol.RemoveAllConformers()
            
            keep_indices.append(i)
        
        # Apply cleaning
        if len(keep_indices) != len(self):
            cleaned_storage = self.subset(keep_indices)
            self._objects = cleaned_storage._objects
            self._features = cleaned_storage._features
            self._conformer_features = cleaned_storage._conformer_features
            
            removed_count = len(self.molecules) + len(keep_indices) - len(self)
            logger.info(f"Cleaned {removed_count} molecules, kept {len(self)}")
        
        return keep_indices
    
    def extend(self, other: MolStorage) -> None:
        """Extend with another MolStorage."""
        if not isinstance(other, MolStorage):
            raise TypeError(f"Cannot extend MolStorage with {type(other)}")
        
        super().extend(other)
        
        # Extend conformer features
        self._conformer_features.extend(other._conformer_features)

    def subset(self, indices: Union[List[int], np.ndarray]) -> MolStorage:
        """Create a subset of this storage."""
        base_subset = super().subset(indices)
        
        # Handle conformer features - need to map molecule idx to conf idx
        conformer_indices = []
        conformer_start = 0
        
        for mol_idx in range(len(self)):
            n_conf = self.num_conformers_per_molecule[mol_idx]
            if mol_idx in indices:
                conformer_indices.extend(range(conformer_start, 
                                               conformer_start + n_conf))
            conformer_start += n_conf
        
        subset_conformer_features = \
            self._conformer_features.subset(conformer_indices)
        
        return self.__class__(
            molecules=base_subset._objects['molecules'],
            features=base_subset._features,
            metadata=base_subset._metadata,
            conformer_features=subset_conformer_features
        )
    
    def get_save_ready_objects(self) -> Dict[str, List[Any]]: 
        """Convert molecules to JSON strings for saving."""
        # TODO: opt to change serialization of mols

        json_mols = []
        for mol in self.molecules:
            if mol is None:
                json_mols.append("")
            else:
                try:
                    json_mols.append(Chem.MolToJSON(mol))
                except Exception as e:
                    logger.warning(f"Failed to serialize molecule: {e}")
                    json_mols.append("")
        
        return {'molecules': json_mols}
    
    def get_load_ready_objects(self, 
                               db: h5py.File, 
                               indices: Optional[List[int]] = None) \
                                -> Dict[str, List[Any]]:
        """Load molecules from JSON strings."""
        if indices is None:
            json_data = db['molecules'][:]
        else:
            json_data = db['molecules'][indices]
        
        molecules = []
        for json_str in json_data:
            json_str = json_str.decode() \
                if isinstance(json_str, bytes) else json_str
            if not json_str:
                molecules.append(None)
            else:
                try:
                    mol = Chem.JSONToMols(json_str)[0]
                    assert mol is not None
                    Chem.SanitizeMol(mol)
                    molecules.append(mol)
                except Exception as e:
                    logger.warning(f"Failed to deserialize molecule: {e}")
                    molecules.append(None)
        
        return {'molecules': molecules}
    
    def save_features_to_file(self, db: h5py.File) -> None:
        """Save both molecule and conformer features."""
        # Save molecule features
        super().save_features_to_file(db)
        
        # Save conformer features
        if len(self._conformer_features) > 0:
            if 'conformer_features' not in db:
                conf_group = db.create_group('conformer_features')
            else:
                conf_group = db['conformer_features']
            
            for feat_key in self._conformer_features.keys():
                features = self._conformer_features.get_features(feat_key)
                metadata = self._conformer_features.get_metadata(feat_key)
                
                self._append_to_featdb(conf_group, feat_key, features)
                if isinstance(metadata, StorageMetadata):
                    for key, val in metadata.items():
                        conf_group[feat_key].attrs[key] = val
    
    def load_features_from_file(self,
                                db: h5py.File,
                                indices: Optional[Union[List[int], np.ndarray, slice]] = None,
                                append: bool = False) -> None:
        """Load both molecule and conformer features."""
        # Load molecule features
        super().load_features_from_file(db, indices, append)
        
        # Load conformer features
        if 'conformer_features' not in db:
            return
        
        conformer_features = StorageFeatures()
        conf_grp = db['conformer_features']
        
        for key in conf_grp:
            conformer_features.add_features(
                key=key,
                features=conf_grp[key][indices] \
                    if indices is not None else conf_grp[key][:],
                dtype=np.dtype(conf_grp[key].attrs.get('dtype', 'float64')),
                featurizer=None,
                metadata=dict(conf_grp[key].attrs)
            )
        
        if append:
            self._conformer_features.extend(conformer_features)
        else:
            self._conformer_features = conformer_features
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({len(self)} molecules, "
                f"{self.num_conformers} conformers, "
                f"{self.num_features} mol feats)")