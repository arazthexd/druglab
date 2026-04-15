import pytest
import pandas as pd
import numpy as np

from druglab.db.table.molecule import MoleculeTable

def test_convert_numeric():
    """Verify that BaseTable converts string numerals to proper numerics."""
    # Data often formatted as strings from CSV reading
    smiles = ["CCO", "c1ccccc1"]
    meta = pd.DataFrame({
        "ID": ["mol1", "mol2"],
        "MolWt": ["46.07", "78.11"],  # String floats
        "NumRings": ["0", "1"]       # String ints
    })
    
    table = MoleculeTable.from_smiles(smiles, metadata=meta)
    table.backend.try_numerize_metadata()
    
    # ID should remain object/string
    assert pd.api.types.is_object_dtype(table.metadata["ID"]) or pd.api.types.is_string_dtype(table.metadata["ID"])
    
    # MolWt and NumRings should be auto-converted to numeric
    assert pd.api.types.is_numeric_dtype(table.metadata["MolWt"])
    assert pd.api.types.is_numeric_dtype(table.metadata["NumRings"])
    assert table.metadata["MolWt"].iloc[0] == 46.07

def test_pythonic_mask_filtering():
    """Verify boolean mask filtering works smoothly after removing filter_by_metadata."""
    table = MoleculeTable.from_smiles(["C", "CC", "CCC", "CCCC"])
    table.add_rdkit_descriptors(["MolWt"])
    
    # Using pythonic boolean mask instead of old string query
    mask = table.metadata["MolWt"] > 35
    heavy_table = table[mask]
    
    assert len(heavy_table) == 2
    assert heavy_table.smiles == ["CCC", "CCCC"]

def test_drop_metadata():
    """Verify column dropping functionalities."""
    table = MoleculeTable.from_smiles(["CCO"])
    table.add_metadata_column("Prop1", [10])
    table.add_metadata_column("Prop2", [20])
    table.add_metadata_column("Prop3", [30])
    
    assert set(table.metadata_columns) == {"smiles", "Prop1", "Prop2", "Prop3"}
    
    # Test plural dropping
    table.drop_metadata_columns(["Prop1", "Prop3"])
    assert set(table.metadata_columns) == {"smiles", "Prop2"}

def test_add_metadata_columns_batch():
    """Verify adding multiple completely new metadata columns."""
    table = MoleculeTable.from_smiles(["C", "CC"])
    
    # Test batch adding a dataframe
    external_df = pd.DataFrame({"NewProp": [100, 200]})
    table.add_metadata_columns(external_df)
    
    assert "NewProp" in table.metadata.columns
    assert list(table.metadata["NewProp"]) == [100, 200]
    
    # Test length mismatch protection
    bad_df = pd.DataFrame({"BadProp": [100]})
    with pytest.raises(ValueError):
        table.add_metadata_columns(bad_df)

def test_update_metadata_with_join_key():
    """Verify merging external metadata using a join key safely preserves table invariants."""
    meta = pd.DataFrame({"ID": ["A", "B", "C"]})
    table = MoleculeTable.from_smiles(["C", "CC", "CCC"], metadata=meta)
    
    external_df = pd.DataFrame({
        "ID": ["C", "A", "B"],  # Out of order
        "Score": [30, 10, 20]
    })
    
    table.merge_metadata(external_df, on="ID")
    
    # Verify order is strictly maintained
    assert list(table.metadata["ID"]) == ["A", "B", "C"]
    assert list(table.metadata["Score"]) == [10, 20, 30]
    
    # Ensure invalid merge key throws error
    with pytest.raises(ValueError, match="not found in table"):
        table.merge_metadata(external_df, on="MissingKey")