import pytest
import numpy as np
from unittest.mock import patch
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from druglab.storage.mol import MolStorage
from druglab.storage import GenericMoleculePrepper
from druglab.storage.base import StorageFeatures, StorageMetadata


class TestGenericMoleculePrepper:
    """Test suite for GenericMoleculePrepper class."""
    
    @pytest.fixture
    def valid_molecules(self):
        """Create a list of valid RDKit molecules for testing."""
        smiles_list = [
            'CCCO',  # Ethanol
            'CCCC(=O)O',  # Acetic acid
            'c1ccccc1',  # Benzene
            'CCN(CC)CC',  # Triethylamine
            'CC(C)C',  # Isobutane
        ]
        return [Chem.MolFromSmiles(smi) for smi in smiles_list]
    
    @pytest.fixture
    def invalid_molecules(self):
        """Create a list of invalid/problematic molecules for testing."""
        return [
            None,  # None molecule
            Chem.MolFromSmiles(''),  # Empty molecule
            Chem.MolFromSmiles('C[C'),  # Invalid SMILES
            Chem.Mol(),  # Empty mol object
        ]
    
    @pytest.fixture
    def mixed_molecules(self, valid_molecules, invalid_molecules):
        """Create a mixed list of valid and invalid molecules."""
        # Filter out None values from invalid molecules
        invalid_filtered = [mol for mol in invalid_molecules if mol is not None]
        return valid_molecules[:3] + invalid_filtered + valid_molecules[3:]
    
    @pytest.fixture
    def salt_molecules(self):
        """Create molecules with salts for testing salt removal."""
        salt_smiles = [
            'CCO.[Na+].[Cl-]',  # Ethanol with NaCl
            'CC(=O)[O-].[Na+]',  # Sodium acetate
            'CC(C)(C)[NH3+].[Cl-]',  # tert-Butylamine HCl
        ]
        return [Chem.MolFromSmiles(smi) for smi in salt_smiles]
    
    @pytest.fixture
    def mol_storage(self, valid_molecules):
        """Create a MolStorage instance with valid molecules."""
        features = StorageFeatures()
        features.add_features(
            'test_feature',
            np.random.rand(len(valid_molecules), 5),
            featurizer=None
        )
        return MolStorage(
            molecules=valid_molecules,
            features=features,
            metadata=StorageMetadata(**{'test': 'data'})
        )
    
    @pytest.fixture
    def mixed_mol_storage(self, mixed_molecules):
        """Create a MolStorage instance with mixed valid/invalid molecules."""
        return MolStorage(molecules=mixed_molecules)
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        prepper = GenericMoleculePrepper()
        
        assert prepper.remove_salts is True
        assert prepper.keep_largest_frag is True
        assert prepper.neutralize is True
        assert prepper.standardize_tautomers is True
        assert prepper.addhs is False
        assert prepper.removehs is False
        assert prepper.cgen is False
        assert prepper.cgen_n == 1
        assert prepper.copt is False
        assert prepper.cclust is False
        assert prepper.calign is False
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        prepper = GenericMoleculePrepper(
            remove_salts=False,
            addhs=True,
            cgen=True,
            cgen_n=10,
            copt=True,
            cclust=True,
            calign=True,
            n_processes=4
        )
        
        assert prepper.remove_salts is False
        assert prepper.addhs is True
        assert prepper.cgen is True
        assert prepper.cgen_n == 10
        assert prepper.copt is True
        assert prepper.cclust is True
        assert prepper.calign is True
    
    def test_standardize_molecule_valid(self, valid_molecules):
        """Test molecule standardization with valid molecules."""
        prepper = GenericMoleculePrepper()
        
        for mol in valid_molecules:
            result = prepper._standardize_molecule(mol)
            assert result is not None
            assert isinstance(result, Chem.Mol)
    
    def test_standardize_molecule_none(self):
        """Test molecule standardization with None input."""
        prepper = GenericMoleculePrepper()
        result = prepper._standardize_molecule(None)
        assert result is None
    
    def test_standardize_molecule_salt_removal(self, salt_molecules):
        """Test salt removal functionality."""
        prepper = GenericMoleculePrepper(remove_salts=True)
        
        for mol in salt_molecules:
            if mol is not None:
                original_atoms = mol.GetNumAtoms()
                result = prepper._standardize_molecule(mol)
                
                if result is not None:
                    # After salt removal, should have fewer atoms
                    assert result.GetNumAtoms() <= original_atoms
    
    def test_prepare_molecule_basic(self, valid_molecules):
        """Test basic molecule preparation."""
        prepper = GenericMoleculePrepper()
        
        for mol in valid_molecules:
            result = prepper.prepare(mol)
            assert result is not None
            assert isinstance(result, Chem.Mol)
    
    def test_prepare_molecule_with_conformers(self, valid_molecules):
        """Test molecule preparation with conformer generation."""
        prepper = GenericMoleculePrepper(cgen=True, cgen_n=5)
        
        mol = valid_molecules[0]  # Use first molecule
        result = prepper.prepare(mol)
        
        if result is not None:
            # Should have generated conformers
            assert result.GetNumConformers() > 0
            assert result.GetNumConformers() <= 5
    
    def test_prepare_molecule_with_hydrogens(self, valid_molecules):
        """Test molecule preparation with hydrogen addition/removal."""
        mol = valid_molecules[0]
        original_atoms = mol.GetNumAtoms()
        
        # Test adding hydrogens
        prepper_add = GenericMoleculePrepper(addhs=True)
        result_add = prepper_add.prepare(mol)
        
        if result_add is not None:
            assert result_add.GetNumAtoms() >= original_atoms
        
        # Test removing hydrogens
        prepper_remove = GenericMoleculePrepper(addhs=True, removehs=True)
        result_remove = prepper_remove.prepare(mol)
        
        if result_remove is not None:
            # Should be back to original or fewer atoms
            assert result_remove.GetNumAtoms() <= result_add.GetNumAtoms()
    
    def test_prepare_molecule_invalid_input(self):
        """Test molecule preparation with invalid input."""
        prepper = GenericMoleculePrepper()
        
        # Test with None
        result = prepper.prepare(None)
        assert result is None
        
        # Test with empty molecule
        empty_mol = Chem.Mol()
        result = prepper.prepare(empty_mol)
        # May return None or a valid molecule depending on standardization
        # Just ensure it doesn't crash
    
    def test_modify_objects_valid(self, valid_molecules):
        """Test modify_objects method with valid molecules."""
        prepper = GenericMoleculePrepper()
        
        for mol in valid_molecules:
            obj_dict = {'molecules': mol}
            result = prepper.modify_objects(obj_dict, None)
            
            assert result is not None
            assert 'molecules' in result
            assert isinstance(result['molecules'], Chem.Mol)
    
    def test_modify_objects_invalid(self):
        """Test modify_objects method with invalid molecules."""
        prepper = GenericMoleculePrepper()
        
        # Test with None molecule
        obj_dict = {'molecules': None}
        result = prepper.modify_objects(obj_dict, None)
        assert result is None
    
    def test_modify_objects_exception_handling(self):
        """Test modify_objects method exception handling."""
        prepper = GenericMoleculePrepper()
        
        # Test with missing key
        obj_dict = {'wrong_key': Chem.MolFromSmiles('CCO')}
        result = prepper.modify_objects(obj_dict, None)
        assert result is None
    
    def test_cluster_conformers_single_conformer(self, valid_molecules):
        """Test conformer clustering with single conformer."""
        prepper = GenericMoleculePrepper()
        mol = valid_molecules[0]
        
        # Add single conformer
        rdDistGeom.EmbedMolecule(mol)
        result = prepper._cluster_conformers(mol)
        
        assert result.GetNumConformers() == 1
    
    def test_cluster_conformers_multiple_conformers(self, valid_molecules):
        """Test conformer clustering with multiple conformers."""
        prepper = GenericMoleculePrepper()
        mol = valid_molecules[1]
        
        # Add multiple conformers
        rdDistGeom.EmbedMultipleConfs(mol, 10)
        original_conformers = mol.GetNumConformers()
        
        if original_conformers > 1:
            result = prepper._cluster_conformers(mol)
            # Should have same or fewer conformers after clustering
            assert result.GetNumConformers() <= original_conformers
    
    @patch('druglab.storage.mol.preps.logger')
    def test_apply_modifications_remove_fails_true(self, 
                                                   mock_logger, 
                                                   mol_storage):
        """Test apply_modifications with remove_fails=True."""
        prepper = GenericMoleculePrepper()
        
        # Create some successful and failed modifications
        modified_objects = [
            {'molecules': Chem.MolFromSmiles('CCO')},
            None,  # Failed modification
            {'molecules': Chem.MolFromSmiles('CC(=O)O')},
            None,  # Failed modification
        ]
        
        # Mock the storage update to avoid importing utils
        with patch('druglab.storage.utils._list_to_dict') as mock_list_to_dict:
            mock_list_to_dict.return_value = {
                'molecules': [
                    Chem.MolFromSmiles('CCO'),
                    Chem.MolFromSmiles('CC(=O)O')
                ]
            }
            
            prepper.apply_modifications(
                mol_storage, 
                modified_objects, 
                None, 
                remove_fails=True
            )
            
            # Should log success/failure statistics
            mock_logger.info.assert_called()
    
    @patch('druglab.storage.mol.preps.logger')
    def test_apply_modifications_remove_fails_false(self, 
                                                    mock_logger, 
                                                    mol_storage):
        """Test apply_modifications with remove_fails=False."""
        prepper = GenericMoleculePrepper()
        
        # Create some successful and failed modifications
        modified_objects = [
            {'molecules': Chem.MolFromSmiles('CCO')},
            None,  # Failed modification
            {'molecules': Chem.MolFromSmiles('CC(=O)O')},
        ]
        
        # Mock the storage update functions
        with patch('druglab.storage.utils._dict_to_list') as mock_dict_to_list, \
             patch('druglab.storage.utils._list_to_dict') as mock_list_to_dict:
            
            # Mock original objects
            mock_dict_to_list.return_value = [
                Chem.MolFromSmiles('CCO'),
                Chem.MolFromSmiles('CCC'),  # Original for failed modification
                Chem.MolFromSmiles('CC(=O)O'),
            ]
            
            prepper.apply_modifications(
                mol_storage, 
                modified_objects, 
                None, 
                remove_fails=False
            )
            
            # Should call the mocked functions
            mock_dict_to_list.assert_called_once()
            mock_list_to_dict.assert_called_once()
            mock_logger.info.assert_called()
    
    def test_comprehensive_preparation_workflow(self, valid_molecules):
        """Test a comprehensive preparation workflow with multiple options."""
        prepper = GenericMoleculePrepper(
            remove_salts=True,
            neutralize=True,
            addhs=True,
            cgen=True,
            cgen_n=3,
            copt=True,
            cclust=True,
            calign=True,
            removehs=True
        )
        
        mol = valid_molecules[2]  # Use benzene
        result = prepper.prepare(mol)
        
        # Should successfully prepare the molecule
        assert result is not None
        assert isinstance(result, Chem.Mol)
        
        # Should have gone through all preparation steps
        # Exact conformer count may vary due to clustering
        
    def test_preparation_with_conformer_generation_failure(self):
        """Test handling of conformer generation failures."""
        prepper = GenericMoleculePrepper(cgen=True, cgen_n=10)
        
        # Create a molecule that might fail conformer generation
        # Using a very constrained molecule
        mol = Chem.MolFromSmiles('C1C45C=C3CC1C2CC5=CCC2OC4C=C3')
        
        # This should handle the case where conformer generation fails
        result = prepper.prepare(mol)
        assert result is None
    
    def test_error_handling_in_modify_objects(self):
        """Test error handling in modify_objects method."""
        prepper = GenericMoleculePrepper()
        
        # Test with various problematic inputs
        test_cases = [
            {},  # Empty dict
            {'molecules': 'not_a_molecule'},  # Wrong type
            None,  # None input
        ]
        
        for test_case in test_cases:
            result = prepper.modify_objects(test_case, None)
            assert result is None
    
    @pytest.mark.parametrize("remove_salts,neutralize,addhs", [
        (True, True, False),
        (False, True, True),
        (True, False, True),
        (False, False, False),
    ])
    def test_parameter_combinations(self, valid_molecules, remove_salts, neutralize, addhs):
        """Test various parameter combinations."""
        prepper = GenericMoleculePrepper(
            remove_salts=remove_salts,
            neutralize=neutralize,
            addhs=addhs
        )
        
        mol = valid_molecules[0]
        result = prepper.prepare(mol)
        
        # Should not crash with any combination
        assert result is None or isinstance(result, Chem.Mol)
    
    def test_conformer_id_sequential_assignment(self, valid_molecules):
        """Test that conformer IDs are assigned sequentially."""
        prepper = GenericMoleculePrepper(cgen=True, cgen_n=5)
        
        mol = valid_molecules[1]
        result = prepper.prepare(mol)
        
        if result is not None and result.GetNumConformers() > 1:
            conformers = result.GetConformers()
            for i, conf in enumerate(conformers):
                assert conf.GetId() == i