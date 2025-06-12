from __future__ import annotations
from typing import List, Any, Tuple, Type, Dict, Optional, Union
from pathlib import Path
from collections import defaultdict
import logging

import numpy as np
import h5py
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from rdkit import Chem, rdBase
from rdkit.Chem import rdChemReactions as rdRxn
from rdkit.Chem.rdChemReactions import ChemicalReaction as Rxn

from ..base import BaseStorage, StorageFeatures, StorageMetadata
from ..mol import MolStorage

logger = logging.getLogger(__name__)


class ReactantGroup:
    """Manages molecules matching a specific reactant template."""
    
    def __init__(self, 
                 template: Chem.Mol,
                 rxn_id: int,
                 reactant_id: int):
        self.template = template
        self.rxn_id = rxn_id
        self.reactant_id = reactant_id
        self.mol_indices: List[int] = []
        self._knn: Optional[NearestNeighbors] = None
    
    def add_molecule_index(self, mol_idx: int) -> None:
        """Add a molecule index to this reactant group."""
        if mol_idx not in self.mol_indices:
            self.mol_indices.append(mol_idx)
    
    def remove_molecule_index(self, mol_idx: int) -> None:
        """Remove a molecule index from this reactant group."""
        if mol_idx in self.mol_indices:
            self.mol_indices.remove(mol_idx)
    
    def get_molecules(self, mol_storage: MolStorage) -> List[Chem.Mol]:
        """Get all molecules in this reactant group."""
        return [mol_storage.molecules[i] for i in self.mol_indices]
    
    def get_molstore(self, mol_storage: MolStorage) -> MolStorage:
        """Get a MolStorage subset for this reactant group."""
        return mol_storage.subset(self.mol_indices)
    
    def init_knn(self, 
                 mol_storage: MolStorage,
                 feature_key: str = None,
                 knn: Optional[NearestNeighbors] = None) -> NearestNeighbors:
        """Initialize KNN for this reactant group."""
        if not self.mol_indices:
            raise ValueError("Cannot initialize KNN for empty reactant group")
        
        # Get features for molecules in this group
        if feature_key is None:
            # Use concatenated features if no specific key provided
            if len(mol_storage.features) == 0:
                raise ValueError("No features available in molecule storage")
            all_features, _ = mol_storage.features.concatenate_all()
            features = all_features[self.mol_indices]
        else:
            mol_features = mol_storage.features.get_features(feature_key)
            if mol_features is None:
                raise ValueError(f"Feature key '{feature_key}' not found")
            features = mol_features[self.mol_indices]
            
        if knn is None:
            knn = NearestNeighbors()
        
        self._knn = knn
        self._knn.fit(features)
        return self._knn
    
    def nearest(self, 
                features: np.ndarray, 
                k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Find nearest neighbors within this reactant group."""
        if self._knn is None:
            raise ValueError("KNN not initialized. Call init_knn() first.")
        
        return self._knn.kneighbors(features, 
                                    n_neighbors=k, 
                                    return_distance=True)
    
    def __len__(self) -> int:
        return len(self.mol_indices)
    
    def __repr__(self) -> str:
        return (f"ReactantGroup(rxn={self.rxn_id}, "
                f"reactant={self.reactant_id}, "
                f"molecules={len(self.mol_indices)})")


class RxnStorage(BaseStorage):
    """Storage for chemical reactions with utilities for matched molecules."""
    
    @property
    def required_object_keys(self) -> List[str]:
        return ['reactions']
    
    @property
    def save_dtypes(self) -> Dict[str, Type[np.dtype]]:
        return {'reactions': h5py.string_dtype()}
    
    def __init__(self, 
                 reactions: Optional[List[Rxn]] = None,
                 features: Optional[StorageFeatures] = None,
                 metadata: Optional[StorageMetadata] = None):
        """
        Initialize RxnStorage.
        
        Args:
            reactions: List of RDKit reaction objects
            features: StorageFeatures container for reactions
            metadata: Storage metadata
        """
        # Convert reactions to proper format for BaseStorage
        objects = None
        if reactions is not None:
            objects = {'reactions': reactions}
            
        super().__init__(
            objects=objects,
            features=features,
            metadata=metadata
        )
        
        # Initialize reactant groups
        self._reactant_groups: List[ReactantGroup] = []
        self._mol_to_groups: Dict[int, List[ReactantGroup]] = defaultdict(list)
        
        # Build reactant groups from existing reactions
        if reactions:
            self._build_reactant_groups()

    @property
    def reactions(self) -> List[Rxn]:
        """Get reactions (alias for objects['reactions'])."""
        return self._objects['reactions']
    
    @property
    def reactant_groups(self) -> List[ReactantGroup]:
        """Get all reactant groups."""
        return self._reactant_groups.copy()
    
    def _build_reactant_groups(self) -> None:
        """Build reactant groups from current reactions."""
        self._reactant_groups.clear()
        self._mol_to_groups.clear()
        
        for rxn_id, rxn in enumerate(self.reactions):
            if rxn is None:
                continue
            for reactant_id, template in enumerate(rxn.GetReactants()):
                group = ReactantGroup(template, rxn_id, reactant_id)
                self._reactant_groups.append(group)

    def add_reaction(self, reaction: Rxn) -> None:
        """Add a single reaction to storage."""
        self._objects['reactions'].append(reaction)
        # Update reactant groups
        rxn_id = len(self.reactions) - 1
        if reaction is not None:
            for reactant_id, template in enumerate(reaction.GetReactants()):
                group = ReactantGroup(template, rxn_id, reactant_id)
                self._reactant_groups.append(group)
    
    def add_reactions(self, reactions: List[Rxn]) -> None:
        """Add multiple reactions to storage."""
        start_idx = len(self.reactions)
        self._objects['reactions'].extend(reactions)
        
        # Update reactant groups
        for i, reaction in enumerate(reactions):
            rxn_id = start_idx + i
            if reaction is not None:
                for reactant_id, template in enumerate(reaction.GetReactants()):
                    group = ReactantGroup(template, rxn_id, reactant_id)
                    self._reactant_groups.append(group)

    def extend(self, other: 'RxnStorage') -> None:
        """Extend with another RxnStorage."""
        if not isinstance(other, RxnStorage):
            raise TypeError(f"Cannot extend RxnStorage with {type(other)}")
        
        # Store original counts for updating indices
        original_rxn_count = len(self)
        
        # Extend base storage
        super().extend(other)
        
        # Update and extend reactant groups
        for group in other._reactant_groups:
            new_group = ReactantGroup(
                group.template,
                group.rxn_id + original_rxn_count,  # Adjust reaction ID
                group.reactant_id
            )
            # Keep molecule indices as-is (external mol storage management)
            new_group.mol_indices = group.mol_indices.copy()
            self._reactant_groups.append(new_group)
            
            # Update molecule-to-group mappings
            for mol_idx in new_group.mol_indices:
                self._mol_to_groups[mol_idx].append(new_group)

    def subset(self, indices: Union[List[int], np.ndarray]) -> 'RxnStorage':
        """Create a subset of this storage."""
        # Get base subset
        base_subset = super().subset(indices)
        
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        
        # Find reactant groups for subset reactions
        subset_groups: List[ReactantGroup] = []
        
        for new_idx, old_idx in enumerate(indices):
            # Find reactant groups for this reaction
            reaction_groups = [g 
                               for g in self._reactant_groups 
                               if g.rxn_id == old_idx]
            
            for group in reaction_groups:
                new_group = ReactantGroup(group.template, 
                                          new_idx, 
                                          group.reactant_id)
                new_group.mol_indices = group.mol_indices.copy()
                subset_groups.append(new_group)
        
        # Create new RxnStorage with subset data
        subset_storage = RxnStorage(
            reactions=base_subset._objects['reactions'],
            features=base_subset._features,
            metadata=base_subset._metadata
        )
        
        subset_storage._reactant_groups = subset_groups
        subset_storage._rebuild_mol_to_groups_mapping()
        
        return subset_storage
    
    def _rebuild_mol_to_groups_mapping(self) -> None:
        """Rebuild the molecule-to-groups mapping."""
        self._mol_to_groups.clear()
        for group in self._reactant_groups:
            for mol_idx in group.mol_indices:
                self._mol_to_groups[mol_idx].append(group)

    def match_molecules(self, 
                        mol_storage: MolStorage,
                        mol_indices: Optional[List[int]] = None) \
                            -> Dict[int, List[Tuple[int, int]]]:
        """
        Match molecules to reactant templates.
        
        Args:
            mol_storage: External molecule storage to match against
            mol_indices: Specific molecule indices to match. 
                If None, match all molecules.
            
        Returns:
            Dictionary mapping molecule indices to 
                (reaction_id, reactant_id) pairs
        """
        if mol_indices is None:
            mol_indices = list(range(len(mol_storage)))
        
        return self._match_molecules_to_groups(mol_storage, mol_indices)
    
    def _match_molecules_to_groups(self, 
                                   mol_storage: MolStorage,
                                   mol_indices: List[int]) -> Dict[int, List[Tuple[int, int]]]:
        """Internal method to match molecules to reactant groups."""
        matches = defaultdict(list)
        invalid_ids = []
        
        logger.info("Matching molecules to reactants")
        for mol_idx in tqdm(mol_indices):
            if mol_idx >= len(mol_storage):
                invalid_ids.append(mol_idx)
                continue
                
            mol = mol_storage.molecules[mol_idx]
            if mol is None:
                continue
            
            for group in self._reactant_groups:
                try:
                    if mol.HasSubstructMatch(group.template):
                        group.add_molecule_index(mol_idx)
                        self._mol_to_groups[mol_idx].append(group)
                        matches[mol_idx].append((group.rxn_id, group.reactant_id))
                except Exception as e:
                    logger.warning(f"Error matching molecule {mol_idx} to "
                                   f"group {group}: {e}")
        
        if invalid_ids:
            logger.warning(f"Invalid molecule indices when matching "
                          f"molecules to reactant groups: {invalid_ids}")
        
        return dict(matches)
    
    def subset_referenced_mols(self, subset_ids: List[int]) -> None:
        """
        Update molecule indices in reactant groups to reflect a subset of molecules.
        
        This method updates all molecule indices in reactant groups when the external
        molecule storage has been subsetted. It removes indices not in subset_ids
        and remaps remaining indices to their new positions.
        
        Args:
            subset_ids: List of molecule indices that remain after subsetting.
                        These should be the original indices before subsetting.
        """
        # Create mapping from old indices to new indices
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(subset_ids)}
        subset_set = set(subset_ids)
        
        # Update all reactant groups
        groups_to_remove = []
        for group in self._reactant_groups:
            # Filter molecule indices to only those in the subset
            filtered_indices = [idx for idx in group.mol_indices if idx in subset_set]
            
            # Remap to new indices
            group.mol_indices = [old_to_new[idx] for idx in filtered_indices]
            
            # Mark empty groups for removal if desired
            if len(group.mol_indices) == 0:
                groups_to_remove.append(group)
        
        # Optionally remove empty groups (uncomment if desired)
        # for group in groups_to_remove:
        #     self._reactant_groups.remove(group)
        
        # Rebuild molecule-to-groups mapping with new indices
        self._rebuild_mol_to_groups_mapping()
        
        logger.info(f"Updated molecule indices for {len(subset_ids)} molecules. "
                   f"Found {len(groups_to_remove)} empty reactant groups.")
    
    def get_reactant_group(self, 
                           rxn_id: int, 
                           reactant_id: int) -> ReactantGroup:
        """Get a specific reactant group."""
        for group in self._reactant_groups:
            if group.rxn_id == rxn_id and group.reactant_id == reactant_id:
                return group
        raise IndexError(f"Reactant group {rxn_id}, {reactant_id} not found")

    def get_molecules_for_reactant(self, 
                                   mol_storage: MolStorage,
                                   rxn_id: int, 
                                   reactant_id: int) -> List[Chem.Mol]:
        """Get all molecules matching a specific reactant."""
        group = self.get_reactant_group(rxn_id, reactant_id)
        return group.get_molecules(mol_storage)
    
    def clean_reactions(self) -> List[int]:
        """
        Clean the reaction storage.
        
        Args:
            sanitize_reactions: Whether to sanitize reactions and remove failed
            remomve_duplicates: Whether to remove duplicated reactions
            initiate_reactions: Whether to initiate reactions at the end
            
        Returns:
            List of indices of reactions that were kept
        """
        keep_indices = []
        seen_smarts = set()

        blocker = rdBase.BlockLogs()  # noqa: F841
        
        for i, rxn in enumerate(self.reactions):
            if rxn is None:
                continue

            # Check for duplicates
            rxn_sma = rdRxn.ReactionToSmarts(rxn)
            if rxn_sma in seen_smarts:
                continue
            seen_smarts.add(rxn_sma)

            # Sanitize
            try:
                rdRxn.SanitizeRxn(rxn)
            except Exception:
                continue

            # Initiate
            rxn.Initialize()
            
            keep_indices.append(i)
        
        # Apply cleaning
        if len(keep_indices) != len(self):
            cleaned_storage = self.subset(keep_indices)
            self._objects = cleaned_storage._objects
            self._features = cleaned_storage._features
            self._reactant_groups = cleaned_storage._reactant_groups
            self._mol_to_groups = cleaned_storage._mol_to_groups
            
            removed_count = len(self.reactions) + len(keep_indices) - len(self)
            logger.info(f"Cleaned {removed_count} reactions, kept {len(self)}")
        
        return keep_indices
    
    def init_reactant_knn(self, 
                          mol_storage: MolStorage,
                          rxn_id: int, 
                          reactant_id: int,
                          feature_key: str = None,
                          knn: Optional[NearestNeighbors] = None) -> NearestNeighbors:
        """Initialize KNN for a specific reactant group."""
        group = self.get_reactant_group(rxn_id, reactant_id)
        return group.init_knn(mol_storage, feature_key, knn)
    
    def init_all_reactant_knns(self, 
                               mol_storage: MolStorage,
                               feature_key: str = None,
                               knn: Optional[NearestNeighbors] = None) -> None:
        """Initialize KNN for all reactant groups that have molecules."""
        for group in self._reactant_groups:
            if len(group) > 0:
                try:
                    group.init_knn(mol_storage, feature_key, knn)
                except Exception as e:
                    logger.error(f"Failed to initialize KNN for {group}: {e}")

    def nearest_reactants(self, 
                          features: np.ndarray,
                          rxn_id: int,
                          reactant_id: int,
                          k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Find nearest molecules for a specific reactant."""
        group = self.get_reactant_group(rxn_id, reactant_id)
        return group.nearest(features, k)
    
    def get_save_ready_objects(self) -> Dict[str, List[Any]]:
        """Convert reactions to SMARTS strings for saving."""
        smarts_list = []
        for rxn in self.reactions:
            if rxn is None:
                smarts_list.append("")
            else:
                try:
                    smarts_list.append(rdRxn.ReactionToSmarts(rxn))
                except Exception as e:
                    logger.warning(f"Failed to serialize reaction: {e}")
                    smarts_list.append("")
        
        return {'reactions': smarts_list}
    
    def get_load_ready_objects(self, 
                               db: h5py.File, 
                               indices: Optional[List[int]] = None) -> Dict[str, List[Any]]:
        """Load reactions from SMARTS strings."""
        if indices is None:
            smarts_data = db['reactions'][:]
        else:
            smarts_data = db['reactions'][indices]
        
        reactions = []
        for smarts_str in smarts_data:
            smarts_str = smarts_str.decode() \
                if isinstance(smarts_str, bytes) else smarts_str
            if not smarts_str:
                reactions.append(None)
            else:
                try:
                    rxn = rdRxn.ReactionFromSmarts(smarts_str)
                    reactions.append(rxn)
                except Exception as e:
                    logger.warning(f"Failed to deserialize reaction: {e}")
                    reactions.append(None)
        
        return {'reactions': reactions}
    
    def save_features_to_file(self, db: h5py.File) -> None:
        """Save reaction features and reactant groups metadata."""
        # Save reaction features
        super().save_features_to_file(db)
        
        # Save reactant groups metadata
        if self._reactant_groups:
            if 'reactant_groups' not in db:
                rg_group = db.create_group('reactant_groups')
            else:
                rg_group = db['reactant_groups']
            
            group_data = []
            for group in self._reactant_groups:
                group_info = {
                    'template_smarts': Chem.MolToSmarts(group.template) if group.template else "",
                    'rxn_id': group.rxn_id,
                    'reactant_id': group.reactant_id,
                    'mol_indices': group.mol_indices
                }
                group_data.append(str(group_info))  # Store as string for HDF5
            
            if 'group_data' not in rg_group:
                rg_group.create_dataset('group_data', 
                                        data=group_data,
                                        maxshape=(None,),
                                        dtype=h5py.string_dtype())
            else:
                rg_group['group_data'].resize(len(group_data), axis=0)
                rg_group['group_data'][-len(group_data):] = group_data
    
    def load_features_from_file(self,
                                db: h5py.File,
                                indices: Optional[Union[List[int], np.ndarray, slice]] = None,
                                append: bool = False) -> None:
        """Load reaction features and reactant groups metadata."""
        # Load reaction features
        super().load_features_from_file(db, indices, append)
        
        # Load reactant groups
        if 'reactant_groups' in db:
            rg_group = db['reactant_groups']
            if 'group_data' in rg_group:
                if not append:
                    self._reactant_groups = []
                
                # Load all group strings first
                group_strings = rg_group['group_data'][:]
                
                # Convert indices to a set for efficient lookup
                if indices is not None:
                    if isinstance(indices, slice):
                        # Convert slice to list of indices
                        start, stop, step = indices.indices(len(group_strings))
                        target_rxn_ids = set(range(start, stop, step))
                    elif isinstance(indices, np.ndarray):
                        target_rxn_ids = set(indices.tolist())
                    else:
                        target_rxn_ids = set(indices)
                else:
                    target_rxn_ids = None
                
                # Create mapping from original reaction IDs to new reaction IDs
                if target_rxn_ids is not None:
                    sorted_indices = sorted(target_rxn_ids)
                    old_to_new_rxn_id = {
                        old_id: new_id 
                        for new_id, old_id in enumerate(sorted_indices)
                    }
                else:
                    old_to_new_rxn_id = None
                
                for group_str in group_strings:
                    if isinstance(group_str, bytes):
                        group_str = group_str.decode()
                    
                    try:
                        # Parse the string representation back to dict
                        group_info = eval(group_str)  # Note: In production, use json.loads
                        
                        original_rxn_id = group_info['rxn_id']
                        
                        # Skip this group if we're filtering by indices and this reaction isn't included
                        if target_rxn_ids is not None and original_rxn_id not in target_rxn_ids:
                            continue
                        
                        template = Chem.MolFromSmarts(group_info['template_smarts']) \
                            if group_info['template_smarts'] else None
                        
                        # Update reaction ID to match the new index in the subset
                        new_rxn_id = old_to_new_rxn_id[original_rxn_id] \
                            if old_to_new_rxn_id else original_rxn_id
                        
                        group = ReactantGroup(
                            template,
                            new_rxn_id,
                            group_info['reactant_id']
                        )
                        group.mol_indices = group_info['mol_indices']
                        self._reactant_groups.append(group)
                        
                    except Exception as e:
                        logger.warning(f"Failed to load reactant group: {e}")
                
                self._rebuild_mol_to_groups_mapping()
    
    def write(self, path: Union[str, Path], mode: str = 'w') -> None:
        """Write storage to file with reactant groups."""
        super().write(path, mode)
    
    @classmethod
    def load(cls, 
             path: Union[str, Path], 
             indices: Optional[Union[List[int], np.ndarray]] = None) -> RxnStorage:
        """Load storage from file."""
        storage: RxnStorage = super().load(path, indices)
            
        # Rebuild reactant groups after loading
        if not storage._reactant_groups and len(storage.reactions) > 0:
            storage._build_reactant_groups()
                
        return storage
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({len(self)} reactions, "
                f"{len(self._reactant_groups)} reactant groups, "
                f"{self.num_features} reaction features)")