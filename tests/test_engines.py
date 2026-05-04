import pytest
import os
import duckdb
import pandas as pd
import numpy as np

from druglab.db import DuckDBEngine, PandasEngine

# ------------------------------------------------------------------
# 1. Parameterized Fixtures
# ------------------------------------------------------------------

@pytest.fixture(params=["duckdb", "pandas"])
def engine(request, tmp_path):
    """
    Yields identical data states across different engine implementations.
    Every test using this fixture runs multiple times, once for each param.
    """
    if request.param == "duckdb":
        db_file = tmp_path / "test.db"
        conn = duckdb.connect(str(db_file))
        
        # Setup DuckDB State
        conn.execute("CREATE TABLE molecules_metadata (_row_id INTEGER PRIMARY KEY, smiles VARCHAR, molwt DOUBLE)")
        conn.execute("INSERT INTO molecules_metadata VALUES (0, 'C', 16.0), (1, 'CC', 30.0), (2, 'CCC', 44.0), (3, 'CCCC', 58.0)")
        
        conn.execute("CREATE TABLE assays_metadata (_row_id INTEGER PRIMARY KEY, ic50 DOUBLE)")
        conn.execute("INSERT INTO assays_metadata VALUES (0, 1.5), (1, 8.2)")
        conn.close()
        
        eng = DuckDBEngine(str(db_file))
        yield eng
        eng.close()
        
    elif request.param == "pandas":
        # Setup Identical Pandas State
        df_mol = pd.DataFrame({
            "_row_id": [0, 1, 2, 3],
            "smiles": ["C", "CC", "CCC", "CCCC"],
            "molwt": [16.0, 30.0, 44.0, 58.0]
        })
        df_assay = pd.DataFrame({
            "_row_id": [0, 1],
            "ic50": [1.5, 8.2]
        })
        
        store = {
            "molecules_metadata": df_mol,
            "assays_metadata": df_assay
        }
        eng = PandasEngine(_store=store)
        yield eng


# ------------------------------------------------------------------
# 2. Common Engine Tests (Runs for ALL engines)
# ------------------------------------------------------------------

class TestCommonEngines:
    """Tests that enforce the BaseEngine contract across all implementations."""

    def test_materialize_all(self, engine):
        df = engine.materialize("molecules", "metadata")
        assert len(df) == 4
        assert df["smiles"].tolist() == ["C", "CC", "CCC", "CCCC"]

    def test_materialize_namespace_isolation(self, engine):
        df = engine.materialize("assays", "metadata")
        assert len(df) == 2
        assert "ic50" in df.columns
        assert "smiles" not in df.columns

    def test_pushdown_slice(self, engine):
        df = engine.materialize("molecules", "metadata", rows=slice(1, 3))
        assert len(df) == 2
        assert df["_row_id"].tolist() == [1, 2]

    def test_pushdown_list(self, engine):
        df = engine.materialize("molecules", "metadata", rows=[0, 3])
        assert len(df) == 2
        assert df["_row_id"].tolist() == [0, 3]

    def test_pushdown_empty_list(self, engine):
        df = engine.materialize("molecules", "metadata", rows=[])
        assert len(df) == 0
        assert "_row_id" in df.columns

    def test_spawn_view_chaining(self, engine):
        view1 = engine.spawn_view("molecules", rows=[1, 2, 3])
        df1 = view1.materialize("molecules", "metadata")
        assert df1["_row_id"].tolist() == [1, 2, 3]
        
        df2 = view1.materialize("molecules", "metadata", rows=slice(0, 2))
        assert df2["_row_id"].tolist() == [1, 2]
        
        view2 = view1.spawn_view("molecules", rows=[1])
        df3 = view2.materialize("molecules", "metadata")
        assert df3["_row_id"].tolist() == [2]

    def test_write_creates_new_table(self, engine):
        new_data = pd.DataFrame({"_row_id": [0], "name": ["Aspirin"]})
        engine.write("drugs", "properties", new_data)
        df_read = engine.materialize("drugs", "properties")
        assert len(df_read) == 1
        assert df_read["name"].iloc[0] == "Aspirin"

    def test_write_appends_to_existing_table(self, engine):
        append_data = pd.DataFrame({"_row_id": [4, 5], "smiles": ["C5", "C6"], "molwt": [1.0, 2.0]})
        engine.write("molecules", "metadata", append_data)
        df_all = engine.materialize("molecules", "metadata")
        assert len(df_all) == 6

    def test_write_raises_error_on_view(self, engine):
        view = engine.spawn_view("molecules", rows=[0, 1])
        dummy_data = pd.DataFrame({"_row_id": [99], "smiles": ["C"]})
        with pytest.raises(PermissionError):
            view.write("molecules", "metadata", dummy_data)

    def test_export_resets_indices(self, engine):
        """Ensures exporting a view resets the `_row_id` for tensor alignment."""
        view = engine.spawn_view("molecules", rows=[2, 3])
        new_engine = view.export()
        
        df = new_engine.materialize("molecules", "metadata")
        assert len(df) == 2
        # The crucial check: _row_ids must be 0-indexed now, not [2, 3]
        assert df["_row_id"].tolist() == [0, 1]
        assert df["smiles"].tolist() == ["CCC", "CCCC"]

    def test_export_resets_indices(self, engine):
        """Ensures exporting a view resets the `_row_id` for tensor alignment."""
        # The fixture has molecules: [0, 1, 2, 3]. We view indices 2 and 3.
        view = engine.spawn_view("molecules", rows=[2, 3])
        new_engine = view.export()
        
        df = new_engine.materialize("molecules", "metadata")
        assert len(df) == 2
        # The crucial check: _row_ids must be 0-indexed now, not [2, 3]
        assert df["_row_id"].tolist() == [0, 1]
        assert df["smiles"].tolist() == ["CCC", "CCCC"]
        
        # Ensure the other namespace (assays) was also copied over safely!
        df_assays = new_engine.materialize("assays", "metadata")
        assert len(df_assays) == 2

    def test_export_with_namespaces(self, engine):
        """Ensures that exporting a specific table ignores all other tables."""
        # Export ONLY the molecules table
        new_engine = engine.export(namespaces=["molecules"])
        
        df_mol = new_engine.materialize("molecules", "metadata")
        assert len(df_mol) == 4
        
        # Assays should NOT exist in the new engine
        with pytest.raises(KeyError, match="does not exist"):
            new_engine.materialize("assays", "metadata")

# ------------------------------------------------------------------
# 3. Engine-Specific Tests
# ------------------------------------------------------------------

class TestDuckDBSpecific:
    """Tests only applicable to DuckDB."""
    
    def test_export_creates_physical_file(self, tmp_path):
        """DuckDB export must create an actual file on disk."""
        db_file = tmp_path / "root.db"
        target_file = tmp_path / "exported.db"
        
        eng = DuckDBEngine(str(db_file))
        eng.write("test", "data", pd.DataFrame({"_row_id": [0], "val": [1]}))
        
        eng.export(target=str(target_file))
        assert os.path.exists(target_file)
        eng.close()

class TestPandasSpecific:
    """Tests only applicable to the Pandas Engine."""
    
    def test_export_creates_detached_memory(self):
        """Exporting a pandas engine must deep-copy data, not share references."""
        df = pd.DataFrame({"_row_id": [0], "val": [100]})
        eng = PandasEngine(_store={"test_data": df})
        
        new_eng = eng.export()
        new_df = new_eng.materialize("test", "data")
        
        # Modify the exported dataframe
        new_eng.write("test", "data", pd.DataFrame({"_row_id": [1], "val": [200]}))
        
        # Ensure original engine is untouched
        original_df = eng.materialize("test", "data")
        assert len(original_df) == 1
        assert len(new_eng.materialize("test", "data")) == 2

    def test_duckdb_zero_copy_export(self, tmp_path):
        """Proves DuckDB can use ATTACH to export to a brand new file safely."""
        source_file = tmp_path / "source.db"
        target_file = tmp_path / "target.db"
        
        # 1. Setup Source
        eng = DuckDBEngine(str(source_file))
        eng.write("drugs", "data", pd.DataFrame({"_row_id": [0, 1, 2], "val": [10, 20, 30]}))
        
        # 2. Restrict it
        view = eng.spawn_view("drugs", rows=[1, 2])
        
        # 3. Export to target
        new_eng = view.export(target=str(target_file))
        
        # 4. Verify Target File Exists and Works
        assert os.path.exists(target_file)
        
        df = new_eng.materialize("drugs", "data")
        assert len(df) == 2
        assert df["_row_id"].tolist() == [0, 1]
        assert df["val"].tolist() == [20, 30]
        
        # Cleanup
        eng.close()
        new_eng.close()