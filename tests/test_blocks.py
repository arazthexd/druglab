"""
tests/test_blocks.py
~~~~~~~~~~~~~~~~~~~~
Tests for specific out-of-the-box cheminformatics pipeline blocks.
Requires RDKit to be installed.
"""

import pytest
import numpy as np

pytest.importorskip("rdkit")

from druglab.db.molecule import MoleculeTable
from druglab.pipe.blocks import *

@pytest.fixture
def sample_table():
    """Provides a basic table of varied molecules."""
    smiles_list = [
        "CC(=O)Oc1ccccc1C(=O)O",         # Aspirin (Normal)
        "c1ccccc1.Cl",                   # Benzene + HCl (Needs desalting)
        "OC1=CCCC1",                    # Enol form (Needs tautomerization)
        "C",                             # Methane
        "[Na+].[Cl-]",                   # Pure inorganic
        "",                              # Invalid
    ]
    # from_smiles handles the "" string by making it None
    return MoleculeTable.from_smiles(smiles_list)

@pytest.fixture()
def clean_table() -> MoleculeTable:
    """A table with only valid, non-duplicate molecules – good for 3D tests."""
    smiles = [
        "CCO",
        "c1ccccc1",
        "CC(=O)O",
        "CCC",
        "c1ccc(cc1)C",
    ]
    return MoleculeTable.from_smiles(smiles)

class TestPreparations:
    def test_molecule_desalter(self, sample_table):
        block = MoleculeDesalter()
        out = block.run(sample_table)
        
        # Benzene + HCl -> Benzene
        assert out.smiles[1] == "c1ccccc1"
        # Pure inorganic remains as is or returns largest part
        assert out.smiles[4] in ("[Cl-]", "[Na+]")

    def test_tautomer_canonicalizer(self, sample_table):
        block = TautomerCanonicalizer()
        out = block.run(sample_table)
        
        # Keto should be canonicalized (phenol form is preferred for this specific ring)
        assert out.smiles[2] == "O=C1CCCC1"

    def test_hydrogen_modifier(self, sample_table):
        # Test Adding
        block_add = HydrogenModifier(add_hs=True)
        out_add = block_add.run(sample_table)
        
        assert out_add.objects[3].GetNumAtoms() == 5 # Methane (1C + 4H)
        
        # Test Removing
        block_rem = HydrogenModifier(add_hs=False)
        out_rem = block_rem.run(out_add)
        assert out_rem.objects[3].GetNumAtoms() == 1 # Back to 1C

    def test_molecule_sanitizer(self, sample_table):
        from rdkit import Chem
        
        # Create an explicitly invalid un-sanitized molecule
        mol = Chem.MolFromSmiles("c1ccccc1", sanitize=False)
        mol.GetAtomWithIdx(0).SetAtomicNum(1) # Break the valence
        
        table = MoleculeTable.from_mols([mol])
        block = MoleculeSanitizer()
        out = block.run(table)
        
        # Should catch the error and return None
        assert out.objects[0] is None

    def test_conformer_generator(self, clean_table):
        block = ConformerGenerator(n_confs=3, ff="MMFF94s")
        out = block.run(clean_table)
        assert isinstance(out, MoleculeTable)
        for mol in out.objects:
            assert mol is not None
            assert mol.GetNumConformers() == 1

        block = ConformerGenerator(n_confs=3, ff="MMFF94s", keep_all=True)
        out = block.run(clean_table)
        assert isinstance(out, MoleculeTable)
        for mol in out.objects:
            assert mol is not None
            assert mol.GetNumConformers() == 3

class TestFilters:
    def test_property_filter(self, sample_table):
        block = PropertyFilter(max_mw=150.0)
        out = block.run(sample_table)
        
        # Aspirin (180) is dropped.
        assert out.n == 5
        assert "CC(=O)Oc1ccccc1C(=O)O" not in [mol for mol in out.smiles if mol]

    def test_smarts_filter(self, sample_table):
        # Exclude Carboxylic acids
        block = SMARTSFilter(smarts="C(=O)[OH]", exclude=True)
        out = block.run(sample_table)
        
        # Aspirin should be dropped
        assert "CC(=O)Oc1ccccc1C(=O)O" not in out.smiles
        # Benzene should remain
        from rdkit import Chem
        assert Chem.MolToSmiles(Chem.MolFromSmiles("c1ccccc1.Cl")) in out.smiles
        
        # Test include mode (only acids)
        block_include = SMARTSFilter(smarts="C(=O)[OH]", exclude=False)
        out_inc = block_include.run(sample_table)
        assert out_inc.n == 1

    def test_element_filter(self, sample_table):
        # Only allow Carbon, Hydrogen, Oxygen (6, 1, 8)
        block = ElementFilter(allowed_elements=(6, 1, 8))
        out = block.run(sample_table)
        
        assert "CC(=O)Oc1ccccc1C(=O)O" in out.smiles # Aspirin kept
        assert "c1ccccc1.Cl" not in out.smiles       # Has Chlorine
        assert "[Na+].[Cl-]" not in out.smiles       # Na and Cl

    def test_validity_filter(self, sample_table):
        block = ValidityFilter()
        out = block.run(sample_table)
        
        # The invalid empty smiles ("") should be dropped
        assert out.n == 5
        assert all(mol is not None for mol in out.objects)

    def test_rule_of_five_filter(self):
        """Palmitic acid (high LogP) should be dropped under strict Ro5."""
        mols = MoleculeTable.from_smiles([
            "CC(=O)Oc1ccccc1C(=O)O",   # Aspirin – passes
            "CCCCCCCCCCCCCCCC(=O)O",    # Palmitic acid – LogP violation
        ])
        block = RuleOfFiveFilter(max_violations=0)
        out = block.run(mols)
        assert out.n == 1

    def test_catalog_filter(self):
        """
        Rhodanine scaffold is a well-known PAINS alerting structure.
        SMILES: C1CSC(=S)N1  (rhodanine)
        """
        from rdkit import Chem
        smiles = [
            "CCO",                           # clean
            "CC(=O)Oc1ccccc1C(=O)O",         # Aspirin – clean
            "O=C1CSC(=S)N1",                 # rhodanine – PAINS hit
        ]
        mols = MoleculeTable.from_smiles(smiles)
        block = CatalogFilter(catalogs=["PAINS"], exclude=True)
        out = block.run(mols)
        survivors = [s for s in out.smiles if s]
        rhodanine_canonical = Chem.MolToSmiles(Chem.MolFromSmiles("O=C1CSC(=S)N1"))
        assert rhodanine_canonical not in survivors

    def test_catalog_filter_multiprocess(self):
        smiles = ["CCO", "CCC", "CCCC", "O=C1CSC(=S)N1"]
        mols = MoleculeTable.from_smiles(smiles)
        out_sp = CatalogFilter(catalogs=["PAINS"], n_workers=1).run(mols)
        out_mp = CatalogFilter(catalogs=["PAINS"], n_workers=2).run(mols)
        assert out_sp.n == out_mp.n

    def test_unique_filter(self):
        mols = MoleculeTable.from_smiles(["c1ccccc1", "CCO", "c1ccccc1"])
        block = UniqueFilter(key="smiles")
        out = block.run(mols)
        assert out.n == 2

        mols = MoleculeTable.from_smiles(["c1ccccc1", "CCO", "c1ccccc1"])
        block = UniqueFilter(key="inchikey")
        out = block.run(mols)
        assert out.n == 2

    def test_unique_filter_eq_smiles(self):
        """Two different SMILES strings that represent the same molecule."""
        mols = MoleculeTable.from_smiles([
            "c1ccccc1",    # benzene
            "C1=CC=CC=C1", # same molecule, different notation
        ])
        block = UniqueFilter(key="smiles")
        out = block.run(mols)
        assert out.n == 1

class TestFeaturizers:
    def test_maccs_featurizer(self, sample_table):
        block = MACCSFeaturizer()
        out = block.run(sample_table)
        
        feat_key = block.name
        assert feat_key in out.features
        assert out.features[feat_key].shape == (6, 167)
        # Check invalid molecule handled gracefully (zeros)
        assert np.all(out.features[feat_key][5] == 0)

    def test_morgan_featurizer(self, sample_table):
        block = MorganFeaturizer(radius=2, n_bits=2048)
        out = block.run(sample_table)
        
        feat_key = block.get_feature_name()
        assert feat_key in out.features
        assert out.features[feat_key].shape == (6, 2048)
        # Check invalid molecule handled gracefully (zeros)
        assert np.all(out.features[feat_key][5] == 0)

    def test_gobbi_featurizer(self, sample_table):
        block = GobbiFeaturizer(n_bits=2048)
        out = block.run(sample_table)
        
        feat_key = block.get_feature_name()
        assert feat_key in out.features
        assert out.features[feat_key].shape == (6, 2048)
        # Check invalid molecule handled gracefully (zeros)
        assert np.all(out.features[feat_key][5] == 0)

    def test_rdkit2d_featurizer(self, sample_table):
        block = RDKit2DFeaturizer()
        out = block.run(sample_table)
        
        feat_key = block.get_feature_name()
        assert feat_key in out.features
        # Check invalid molecule handled gracefully (zeros)
        assert np.all(out.features[feat_key][5] == 0)

        arr = out.features[feat_key]
        assert arr.ndim == 2
        assert arr.shape[0] == sample_table.n
        assert arr.shape[1] > 150  # conservative lower bound

class TestUtilities:
    def test_sampler_block(self, sample_table):
        block = SamplerBlock(max_size=2, random_seed=0)
        out = block.run(sample_table)
        assert out.n == 2

        block = SamplerBlock(max_size=1000, random_seed=0)
        out = block.run(sample_table)
        assert out.n == sample_table.n

        b1 = SamplerBlock(max_size=3, random_seed=7)
        b2 = SamplerBlock(max_size=3, random_seed=7)
        out1 = b1.run(sample_table)
        out2 = b2.run(sample_table)
        assert out1.smiles == out2.smiles
    
    def test_molecule_file_reader_block(self):
        block = MoleculeFileReaderBlock(["tests/data/simple_4_mols.smi"])
        out = block.run()
        assert out.n == 4
        # TODO: Add tests for batch mode

if __name__ == "__main__":
    pytest.main([__file__, "-v"])