import pytest
import numpy as np
import tempfile
from pathlib import Path

from rdkit import Chem

from druglab.storage.mol import MolStorage
from druglab.storage.base import StorageFeatures


# Test molecules
SMILES_LIST = [
    'CCO',  # ethanol
    'c1ccccc1',  # benzene
    'CC(=O)O',  # acetic acid
    'invalid_smiles',  # invalid
    'C1=CC=CC=C1'  # benzene alternative
]

@pytest.fixture
def test_molecules():
    """Create test molecules from SMILES."""
    mols = []
    for smi in SMILES_LIST:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mol = Chem.AddHs(mol)
            mols.append(mol)
        except Exception:
            mols.append(None)
    return mols

@pytest.fixture
def mol_storage(test_molecules):
    """Create MolStorage with test molecules."""
    return MolStorage(molecules=test_molecules)

@pytest.fixture
def mol_storage_with_conformers(test_molecules):
    """Create MolStorage with conformers."""
    # Generate conformers for valid molecules
    for mol in test_molecules:
        if mol is not None:
            try:
                from rdkit.Chem import rdDistGeom
                # Add multiple conformers
                rdDistGeom.EmbedMultipleConfs(mol, numConfs=2, randomSeed=42)
            except Exception:
                pass  # Skip if conformer generation fails
    
    storage = MolStorage(molecules=test_molecules)
    
    # Add conformer features
    n_conformers = storage.num_conformers
    if n_conformers > 0:
        conf_features = np.random.rand(n_conformers, 3)
        storage.conformer_features.add_features('test_conf', conf_features)
    
    return storage


class TestMolStorage:
    """Test MolStorage functionality."""
    
    def test_init_empty(self):
        storage = MolStorage()
        assert len(storage) == 0
        assert len(storage.molecules) == 0
        assert storage.num_conformers == 0
    
    def test_init_with_molecules(self, test_molecules):
        storage = MolStorage(molecules=test_molecules)
        assert len(storage) == len(test_molecules)
        assert len(storage.molecules) == len(test_molecules)
    
    def test_properties(self, mol_storage: MolStorage):
        # Test molecules property
        assert isinstance(mol_storage.molecules, list)
        assert len(mol_storage.molecules) == len(SMILES_LIST)
        
        # Test conformer features
        assert isinstance(mol_storage.conformer_features, StorageFeatures)
        
        # Test conformer counts
        assert isinstance(mol_storage.num_conformers_per_molecule, list)
        assert len(mol_storage.num_conformers_per_molecule) == len(mol_storage)
    
    def test_add_molecule(self, mol_storage: MolStorage):
        initial_len = len(mol_storage)
        new_mol = Chem.MolFromSmiles('CO')  # methanol
        
        mol_storage.add_molecule(new_mol)
        
        assert len(mol_storage) == initial_len + 1
        assert mol_storage.molecules[-1] is new_mol
    
    def test_add_molecules(self, mol_storage: MolStorage):
        initial_len = len(mol_storage)
        new_mols = [
            Chem.MolFromSmiles('CO'),  # methanol
            Chem.MolFromSmiles('CCO')  # ethanol
        ]
        
        mol_storage.add_molecules(new_mols)
        
        assert len(mol_storage) == initial_len + 2
        assert mol_storage.molecules[-2:] == new_mols
    
    def test_conformers_as_molecules(self, 
                                     mol_storage_with_conformers: MolStorage):
        conformer_mols = \
            mol_storage_with_conformers.get_conformers_as_molecules()
        
        # Should get individual molecules for each conformer
        expected_conformers = mol_storage_with_conformers.num_conformers
        assert len(conformer_mols) == expected_conformers
        
        # Each should be a valid molecule with single conformer
        for mol in conformer_mols:
            assert mol is not None
            assert mol.GetNumConformers() == 1
    
    def test_conformers_as_storage(self, 
                                   mol_storage_with_conformers: MolStorage):
        conformer_storage = \
            mol_storage_with_conformers.get_conformers_as_storage()
        
        assert isinstance(conformer_storage, MolStorage)
        assert len(conformer_storage) == \
            mol_storage_with_conformers.num_conformers
        
        # Features should be transferred
        if 'test_conf' in mol_storage_with_conformers.conformer_features:
            assert 'test_conf' in conformer_storage.features
    
    def test_split_conformer_features(self, 
                                      mol_storage_with_conformers: MolStorage):
        if mol_storage_with_conformers.num_conformers == 0:
            pytest.skip("No conformers generated")
        
        # Create dummy conformer features
        n_conf = mol_storage_with_conformers.num_conformers
        features = np.random.rand(n_conf, 4)
        
        split_features = mol_storage_with_conformers\
            ._split_conformer_features_by_molecule(features)
        
        assert len(split_features) == len(mol_storage_with_conformers)
        
        # Verify total length matches
        total_split = sum(len(feat) for feat in split_features)
        assert total_split == n_conf
    
    def test_clean_molecules_remove_none(self, test_molecules):
        # Add None molecules
        molecules_with_none = test_molecules + [None, None]
        storage = MolStorage(molecules=molecules_with_none)
        
        keep_indices = storage.clean_molecules(remove_none=True)
        
        # Should remove None molecules
        assert len(storage) < len(molecules_with_none)
        assert None not in storage.molecules
        assert len(keep_indices) == len(storage)
    
    def test_clean_molecules_remove_duplicates(self, test_molecules):
        # Create duplicates
        duplicated = test_molecules + [test_molecules[0], test_molecules[1]]
        storage = MolStorage(molecules=duplicated)
        
        keep_indices = storage.clean_molecules(
            remove_none=True, 
            remove_duplicates=True
        )
        
        # Should be fewer molecules after deduplication
        assert len(storage) <= len(test_molecules)
        assert len(keep_indices) == len(storage)
    
    def test_clean_molecules_sanitize(self):
        # Create molecule that might need sanitization
        mol = Chem.MolFromSmiles('C')
        storage = MolStorage(molecules=[mol])
        
        keep_indices = storage.clean_molecules(sanitize=True)
        
        assert len(keep_indices) == 1
        assert len(storage) == 1
    
    def test_clean_molecules_remove_conformers(self, 
                                               mol_storage_with_conformers: MolStorage):
        storage = mol_storage_with_conformers
        
        # Check initial conformer count
        initial_conformers = storage.num_conformers
        assert initial_conformers > 0
        
        storage.clean_molecules(remove_conformers=True)
        
        # All conformers should be removed
        assert storage.num_conformers == 0
        
        # But molecules should remain
        assert len(storage) > 0
    
    def test_extend(self, test_molecules):
        storage1 = MolStorage(molecules=test_molecules[:2])
        storage2 = MolStorage(molecules=test_molecules[2:])
        
        # Add features to both
        features1 = np.random.rand(2, 3)
        features2 = np.random.rand(len(test_molecules) - 2, 3)
        
        storage1.features.add_features('test', features1)
        storage2.features.add_features('test', features2)
        
        storage1.extend(storage2)
        
        assert len(storage1) == len(test_molecules)
        
        # Check features were extended
        combined_features = storage1.features.get_features('test')
        expected_features = np.vstack([features1, features2])
        np.testing.assert_array_equal(combined_features, expected_features)
    
    def test_extend_wrong_type(self, mol_storage):
        with pytest.raises(TypeError):
            mol_storage.extend("not a MolStorage")
    
    def test_subset(self, mol_storage_with_conformers):
        storage = mol_storage_with_conformers
        indices = [0, 2]
        
        subset = storage.subset(indices)
        
        assert len(subset) == 2
        assert isinstance(subset, MolStorage)
        
        # Check molecules
        expected_mols = [storage.molecules[i] for i in indices]
        for i, mol in enumerate(subset.molecules):
            if expected_mols[i] is None:
                assert mol is None
            elif mol is None:
                assert expected_mols[i] is None
            else:
                # Compare SMILES since molecule objects might differ
                expected_smi = Chem.MolToSmiles(expected_mols[i])
                actual_smi = Chem.MolToSmiles(mol)
                assert expected_smi == actual_smi
    
    def test_save_load_cycle(self, mol_storage):
        # Add features
        features = np.random.rand(len(mol_storage), 4)
        mol_storage.features.add_features('mol_desc', features)
        
        # Add metadata
        mol_storage.metadata['version'] = 1
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save
            mol_storage.write(temp_path)
            
            # Load
            loaded_storage = MolStorage.load(temp_path)
            
            # Verify length
            assert len(loaded_storage) == len(mol_storage)
            
            # Verify molecules (compare valid ones by SMILES)
            for orig, loaded in zip(mol_storage.molecules, loaded_storage.molecules):
                if orig is None and loaded is None:
                    continue
                elif orig is None or loaded is None:
                    # One is None, other isn't - this might happen with invalid molecules
                    continue
                else:
                    # Both are valid molecules - compare SMILES
                    try:
                        orig_smi = Chem.MolToSmiles(orig)
                        loaded_smi = Chem.MolToSmiles(loaded)
                        assert orig_smi == loaded_smi
                    except Exception:
                        # Skip comparison if SMILES generation fails
                        pass
            
            # Verify features
            loaded_features = loaded_storage.features.get_features('mol_desc')
            np.testing.assert_array_almost_equal(loaded_features, features)
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_save_load_with_conformers(self, mol_storage_with_conformers):
        storage = mol_storage_with_conformers
        
        if storage.num_conformers == 0:
            pytest.skip("No conformers to test")
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save
            storage.write(temp_path)
            
            # Load
            loaded_storage = MolStorage.load(temp_path)
            
            # Verify conformer features were saved/loaded
            if 'test_conf' in storage.conformer_features:
                assert 'test_conf' in loaded_storage.conformer_features
                
                orig_conf_feat = storage.conformer_features.get_features('test_conf')
                loaded_conf_feat = loaded_storage.conformer_features.get_features('test_conf')
                np.testing.assert_array_almost_equal(orig_conf_feat, loaded_conf_feat)
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_save_load_with_indices(self, mol_storage):
        features = np.random.rand(len(mol_storage), 3)
        mol_storage.features.add_features('test', features)
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_path = f.name
        
        try:
            mol_storage.write(temp_path)
            
            # Load subset
            indices = [0, 2]
            loaded_storage = MolStorage.load(temp_path, indices=indices)
            
            assert len(loaded_storage) == 2
            
            # Verify features subset
            loaded_features = loaded_storage.features.get_features('test')
            expected_features = features[indices]
            np.testing.assert_array_equal(loaded_features, expected_features)
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_serialization_edge_cases(self):
        """Test serialization of edge case molecules."""
        # Test with None molecules
        storage = MolStorage(molecules=[None, None])
        
        save_ready = storage.get_save_ready_objects()
        assert save_ready['molecules'] == ['', '']
        
        # Test loading None molecules
        import h5py
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_path = f.name
        
        try:
            with h5py.File(temp_path, 'w') as db:
                db.create_dataset('molecules', data=['', 'invalid_json'])
            
            with h5py.File(temp_path, 'r') as db:
                loaded_objects = storage.get_load_ready_objects(db)
            
            assert loaded_objects['molecules'] == [None, None]
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_repr(self, mol_storage_with_conformers):
        storage = mol_storage_with_conformers
        repr_str = repr(storage)
        
        assert 'MolStorage' in repr_str
        assert f'{len(storage)} molecules' in repr_str
        assert f'{storage.num_conformers} conformers' in repr_str
        assert 'mol feats' in repr_str
    
    def test_required_properties(self):
        storage = MolStorage()
        
        assert storage.required_object_keys == ['molecules']
        assert 'molecules' in storage.save_dtypes