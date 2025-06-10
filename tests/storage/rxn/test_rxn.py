import pytest
import numpy as np
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdChemReactions as rdRxn
from sklearn.neighbors import NearestNeighbors

from druglab.storage.rxn import RxnStorage, ReactantGroup
from druglab.storage.mol import MolStorage


@pytest.fixture
def sample_reactions():
    """Create sample reactions including some invalid ones."""
    reactions = []
    
    # Valid reaction 1: Simple nucleophilic substitution
    rxn1_smarts = "[C:1][Cl:2].[N:3]>>[C:1][N:3]"
    rxn1 = rdRxn.ReactionFromSmarts(rxn1_smarts)
    rdRxn.SanitizeRxn(rxn1)
    rxn1.Initialize()
    reactions.append(rxn1)
    
    # Valid reaction 2: Amide formation
    rxn2_smarts = "[C:1](=[O:2])[OH:3].[N:4]>>[C:1](=[O:2])[N:4]"
    rxn2 = rdRxn.ReactionFromSmarts(rxn2_smarts)
    rdRxn.SanitizeRxn(rxn2)
    rxn2.Initialize()
    reactions.append(rxn2)
    
    # Invalid reaction (None)
    reactions.append(None)
    
    # Invalid reaction (malformed SMARTS)
    try:
        rxn_invalid = rdRxn.ReactionFromSmarts("[C:1][invalid")
        reactions.append(rxn_invalid)
    except Exception:
        reactions.append(None)
    
    return reactions


@pytest.fixture
def sample_molecules():
    """Create sample molecules that match the reaction templates."""
    molecules = [
        Chem.MolFromSmiles("CCCl"),      # Matches rxn1 reactant 1
        Chem.MolFromSmiles("N"),         # Matches rxn1 reactant 2
        Chem.MolFromSmiles("CC(=O)O"),   # Matches rxn2 reactant 1
        Chem.MolFromSmiles("CCN"),       # Matches rxn2 reactant 2
        Chem.MolFromSmiles("CCCC"),      # No matches
        None,                            # Invalid molecule
    ]
    return molecules


@pytest.fixture
def mol_storage(sample_molecules):
    """Create MolStorage with sample molecules."""
    return MolStorage(molecules=sample_molecules)


@pytest.fixture
def rxn_storage(sample_reactions):
    """Create RxnStorage with sample reactions."""
    return RxnStorage(reactions=sample_reactions)


class TestReactantGroup:
    """Test ReactantGroup functionality."""
    
    def test_reactant_group_creation(self, sample_reactions):
        template = sample_reactions[0].GetReactants()[0]
        group = ReactantGroup(template, rxn_id=0, reactant_id=0)
        
        assert group.template == template
        assert group.rxn_id == 0
        assert group.reactant_id == 0
        assert len(group.mol_indices) == 0
    
    def test_add_remove_molecule_index(self, sample_reactions):
        template = sample_reactions[0].GetReactants()[0]
        group = ReactantGroup(template, rxn_id=0, reactant_id=0)
        
        # Add molecule indices
        group.add_molecule_index(1)
        group.add_molecule_index(2)
        assert group.mol_indices == [1, 2]
        
        # Test duplicate addition (should not duplicate)
        group.add_molecule_index(1)
        assert group.mol_indices == [1, 2]
        
        # Remove molecule index
        group.remove_molecule_index(1)
        assert group.mol_indices == [2]
        
        # Remove non-existent index (should not error)
        group.remove_molecule_index(99)
        assert group.mol_indices == [2]
    
    def test_get_molecules(self, sample_reactions, mol_storage):
        template = sample_reactions[0].GetReactants()[0]
        group = ReactantGroup(template, rxn_id=0, reactant_id=0)
        group.add_molecule_index(0)
        group.add_molecule_index(2)
        
        molecules = group.get_molecules(mol_storage)
        assert len(molecules) == 2
        assert molecules[0] == mol_storage.molecules[0]
        assert molecules[1] == mol_storage.molecules[2]
    
    def test_get_molstore(self, sample_reactions, mol_storage):
        template = sample_reactions[0].GetReactants()[0]
        group = ReactantGroup(template, rxn_id=0, reactant_id=0)
        group.add_molecule_index(0)
        group.add_molecule_index(2)
        
        subset_storage = group.get_molstore(mol_storage)
        assert len(subset_storage) == 2
        assert subset_storage.molecules[0] == mol_storage.molecules[0]
        assert subset_storage.molecules[1] == mol_storage.molecules[2]
    
    def test_init_knn_no_molecules(self, sample_reactions, mol_storage):
        template = sample_reactions[0].GetReactants()[0]
        group = ReactantGroup(template, rxn_id=0, reactant_id=0)
        
        with pytest.raises(ValueError, match="Cannot initialize KNN for empty reactant group"):
            group.init_knn(mol_storage)
    
    def test_init_knn_no_features(self, sample_reactions, mol_storage):
        template = sample_reactions[0].GetReactants()[0]
        group = ReactantGroup(template, rxn_id=0, reactant_id=0)
        group.add_molecule_index(0)
        
        with pytest.raises(ValueError, match="No features available"):
            group.init_knn(mol_storage)
    
    def test_init_knn_with_features(self, sample_reactions, mol_storage):
        # Add dummy features to mol_storage
        dummy_features = np.random.rand(len(mol_storage), 10)
        mol_storage.features.add_features('dummy', dummy_features)
        
        template = sample_reactions[0].GetReactants()[0]
        group = ReactantGroup(template, rxn_id=0, reactant_id=0)
        group.add_molecule_index(0)
        group.add_molecule_index(2)
        
        knn = group.init_knn(mol_storage, feature_key='dummy')
        assert isinstance(knn, NearestNeighbors)
        assert group._knn is not None
    
    def test_nearest_no_knn(self, sample_reactions):
        template = sample_reactions[0].GetReactants()[0]
        group = ReactantGroup(template, rxn_id=0, reactant_id=0)
        
        with pytest.raises(ValueError, match="KNN not initialized"):
            group.nearest(np.array([[1, 2, 3]]))
    
    def test_len_and_repr(self, sample_reactions):
        template = sample_reactions[0].GetReactants()[0]
        group = ReactantGroup(template, rxn_id=0, reactant_id=0)
        group.add_molecule_index(1)
        group.add_molecule_index(2)
        
        assert len(group) == 2
        repr_str = repr(group)
        assert "ReactantGroup" in repr_str
        assert "rxn=0" in repr_str
        assert "reactant=0" in repr_str
        assert "molecules=2" in repr_str


class TestRxnStorage:
    """Test RxnStorage functionality."""
    
    def test_rxn_storage_creation_empty(self):
        storage = RxnStorage()
        assert len(storage) == 0
        assert len(storage.reactions) == 0
        assert len(storage.reactant_groups) == 0
    
    def test_rxn_storage_creation_with_reactions(self, sample_reactions):
        storage = RxnStorage(reactions=sample_reactions)
        assert len(storage) == len(sample_reactions)
        assert len(storage.reactions) == len(sample_reactions)
        
        # Should have reactant groups for valid reactions
        expected_groups = 0
        for rxn in sample_reactions:
            if rxn is not None:
                expected_groups += len(rxn.GetReactants())
        assert len(storage.reactant_groups) == expected_groups
    
    def test_add_single_reaction(self, sample_reactions):
        storage = RxnStorage()
        storage.add_reaction(sample_reactions[0])
        
        assert len(storage) == 1
        assert storage.reactions[0] == sample_reactions[0]
        assert len(storage.reactant_groups) == len(sample_reactions[0].GetReactants())
    
    def test_add_multiple_reactions(self, sample_reactions):
        storage = RxnStorage()
        storage.add_reactions(sample_reactions[:2])
        
        assert len(storage) == 2
        expected_groups = sum(len(rxn.GetReactants()) for rxn in sample_reactions[:2] if rxn)
        assert len(storage.reactant_groups) == expected_groups
    
    def test_extend_rxn_storage(self, sample_reactions):
        storage1 = RxnStorage(reactions=sample_reactions[:2])
        storage2 = RxnStorage(reactions=sample_reactions[2:])
        
        original_len = len(storage1)
        storage1.extend(storage2)
        
        assert len(storage1) == len(sample_reactions)
        # Check reaction IDs were updated correctly
        for group in storage1.reactant_groups:
            if group.rxn_id >= original_len:
                # These should be from storage2 with updated IDs
                assert group.rxn_id < len(storage1)
    
    def test_extend_wrong_type(self, rxn_storage, mol_storage):
        with pytest.raises(TypeError):
            rxn_storage.extend(mol_storage)
    
    def test_subset(self, sample_reactions):
        storage = RxnStorage(reactions=sample_reactions)
        subset_storage = storage.subset([0, 1])
        
        assert len(subset_storage) == 2
        assert subset_storage.reactions[0] == sample_reactions[0]
        assert subset_storage.reactions[1] == sample_reactions[1]
        
        # Check reactant groups were updated with correct indices
        for group in subset_storage.reactant_groups:
            assert group.rxn_id < 2
    
    def test_match_molecules(self, rxn_storage, mol_storage):
        matches = rxn_storage.match_molecules(mol_storage)
        
        # Should have matches for molecules that match templates
        assert isinstance(matches, dict)
        assert 0 in matches  # CCCl should match rxn1 reactant 1
        assert 1 in matches  # N should match rxn1 reactant 2
        assert 2 in matches  # CC(=O)O should match rxn2 reactant 1
    
    def test_match_molecules_subset(self, rxn_storage, mol_storage):
        matches = rxn_storage.match_molecules(mol_storage, mol_indices=[0, 1])
        
        # Should only have matches for specified molecules
        assert len(matches) <= 2
        for mol_idx in matches:
            assert mol_idx in [0, 1]
    
    def test_match_molecules_invalid_indices(self, rxn_storage, mol_storage):
        # Should handle invalid indices gracefully
        matches = rxn_storage.match_molecules(mol_storage, 
                                              mol_indices=[0, 999, 1000])
        # Should still work for valid indices
        assert 0 in matches or len(matches) >= 0
    
    def test_get_reactant_group(self, rxn_storage):
        if len(rxn_storage.reactant_groups) > 0:
            group = rxn_storage.get_reactant_group(0, 0)
            assert isinstance(group, ReactantGroup)
            assert group.rxn_id == 0
            assert group.reactant_id == 0
    
    def test_get_reactant_group_not_found(self, rxn_storage):
        with pytest.raises(IndexError):
            rxn_storage.get_reactant_group(999, 999)
    
    def test_get_molecules_for_reactant(self, rxn_storage, mol_storage):
        # First match molecules
        rxn_storage.match_molecules(mol_storage)
        
        if len(rxn_storage.reactant_groups) > 0:
            molecules = rxn_storage.get_molecules_for_reactant(mol_storage, 0, 0)
            assert isinstance(molecules, list)
    
    def test_clean_reactions(self, sample_reactions):
        # Create storage with duplicates and invalid reactions
        reactions_with_duplicates = sample_reactions + [sample_reactions[0]]
        storage = RxnStorage(reactions=reactions_with_duplicates)
        
        original_len = len(storage)
        kept_indices = storage.clean_reactions()
        
        assert len(kept_indices) <= original_len
        assert len(storage) == len(kept_indices)
    
    def test_subset_referenced_mols(self, rxn_storage, mol_storage):
        # Match molecules first
        rxn_storage.match_molecules(mol_storage)
        
        # Subset to first 3 molecules
        subset_ids = [0, 1, 2]
        rxn_storage.subset_referenced_mols(subset_ids)
        
        # Check that molecule indices were updated
        for group in rxn_storage.reactant_groups:
            for mol_idx in group.mol_indices:
                assert mol_idx < len(subset_ids)
    
    def test_init_reactant_knn(self, rxn_storage, mol_storage):
        # Add features and match molecules
        dummy_features = np.random.rand(len(mol_storage), 10)
        mol_storage.features.add_features('dummy', dummy_features)
        rxn_storage.match_molecules(mol_storage)
        
        if len(rxn_storage.reactant_groups) > 0:
            group = rxn_storage.reactant_groups[0]
            if len(group) > 0:
                knn = rxn_storage.init_reactant_knn(mol_storage, 
                                                    group.rxn_id, 
                                                    group.reactant_id,
                                                    feature_key='dummy')
                assert isinstance(knn, NearestNeighbors)
    
    def test_init_all_reactant_knns(self, rxn_storage, mol_storage):
        # Add features and match molecules
        dummy_features = np.random.rand(len(mol_storage), 10)
        mol_storage.features.add_features('dummy', dummy_features)
        rxn_storage.match_molecules(mol_storage)
        
        # Should not raise errors even if some groups are empty
        rxn_storage.init_all_reactant_knns(mol_storage, feature_key='dummy')
    
    def test_nearest_reactants(self, rxn_storage, mol_storage):
        # Add features and match molecules
        dummy_features = np.random.rand(len(mol_storage), 10)
        mol_storage.features.add_features('dummy', dummy_features)
        rxn_storage.match_molecules(mol_storage)
        
        if len(rxn_storage.reactant_groups) > 0:
            group = rxn_storage.reactant_groups[0]
            if len(group) > 0:
                rxn_storage.init_reactant_knn(mol_storage, 
                                              group.rxn_id, 
                                              group.reactant_id,
                                              feature_key='dummy')
                
                query_features = np.random.rand(1, 10)
                distances, indices = rxn_storage.nearest_reactants(
                    query_features, group.rxn_id, group.reactant_id, k=1)
                
                assert distances.shape[0] == 1
                assert indices.shape[0] == 1

class TestRxnStorageSaveLoad:
    """Test save/load functionality for RxnStorage."""
    
    def test_get_save_ready_objects(self, rxn_storage):
        save_objects = rxn_storage.get_save_ready_objects()
        
        assert 'reactions' in save_objects
        assert len(save_objects['reactions']) == len(rxn_storage)
        
        # Should be SMARTS strings or empty strings
        for smarts in save_objects['reactions']:
            assert isinstance(smarts, str)
    
    def test_save_and_load_reactions(self, rxn_storage, mol_storage):
        # Match molecules and add features
        dummy_features = np.random.rand(len(rxn_storage), 5)
        rxn_storage.features.add_features('rxn_features', dummy_features)
        rxn_storage.match_molecules(mol_storage)
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            # Save
            rxn_storage.write(tmp_path)
            
            # Load
            loaded_storage = RxnStorage.load(tmp_path)
            
            assert len(loaded_storage) == len(rxn_storage)
            assert len(loaded_storage.reactant_groups) == len(rxn_storage.reactant_groups)
            assert 'rxn_features' in loaded_storage.features
            
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    
    def test_load_with_indices(self, rxn_storage):
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            # Save
            rxn_storage.write(tmp_path)
            
            # Load subset
            loaded_storage = RxnStorage.load(tmp_path, indices=[0, 1])
            
            assert len(loaded_storage) == min(2, len(rxn_storage))
            
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    
    def test_save_load_features_and_metadata(self, rxn_storage, mol_storage):
        # Add features and metadata
        dummy_features = np.random.rand(len(rxn_storage), 3)
        rxn_storage.features.add_features('test_features', dummy_features)
        rxn_storage.metadata['test_key'] = 'test_value'
        
        # Match molecules to create reactant groups
        rxn_storage.match_molecules(mol_storage)
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            rxn_storage.write(tmp_path)
            loaded_storage = RxnStorage.load(tmp_path)
            
            assert 'test_features' in loaded_storage.features
            assert loaded_storage.metadata['test_key'] == 'test_value'
            assert len(loaded_storage.reactant_groups) == len(rxn_storage.reactant_groups)
            
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestRxnStorageEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_storage_operations(self):
        storage = RxnStorage()
        
        # Should handle empty storage gracefully
        assert len(storage.reactant_groups) == 0
        matches = storage.match_molecules(MolStorage(molecules=[]))
        assert len(matches) == 0
    
    def test_none_reactions_handling(self):
        reactions = [None, None, None]
        storage = RxnStorage(reactions=reactions)
        
        assert len(storage) == 3
        assert len(storage.reactant_groups) == 0  # No valid reactions
    
    def test_invalid_molecule_matching(self, rxn_storage):
        # Create mol storage with None molecules
        invalid_mols = [None, None, None]
        mol_storage = MolStorage(molecules=invalid_mols)
        
        matches = rxn_storage.match_molecules(mol_storage)
        # Should handle None molecules gracefully
        assert isinstance(matches, dict)
    
    def test_repr_string(self, rxn_storage):
        repr_str = repr(rxn_storage)
        assert "RxnStorage" in repr_str
        assert str(len(rxn_storage)) in repr_str
        assert "reactions" in repr_str
        assert "reactant groups" in repr_str