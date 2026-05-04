import os
from typing import ClassVar, Dict, Iterable

import duckdb
import numpy as np
import pandas as pd
import pytest

from druglab.db import DuckDBEngine, PandasEngine
from druglab.db.db.local import LocalDB
from druglab.db.table.base import BaseTable
from druglab.db.table.history import History, HistoryEntry
from druglab.db.types import IdxLike


# ===========================================================================
# 1. Parameterised engine fixture
# ===========================================================================

@pytest.fixture(params=["duckdb", "pandas"])
def engine(request, tmp_path):
    """
    Yields identical data states across different engine implementations.
    Every test using this fixture runs twice — once per backend.
    """
    if request.param == "duckdb":
        db_file = tmp_path / "test.db"
        conn = duckdb.connect(str(db_file))

        conn.execute(
            "CREATE TABLE molecules_metadata "
            "(_row_id INTEGER PRIMARY KEY, smiles VARCHAR, molwt DOUBLE)"
        )
        conn.execute(
            "INSERT INTO molecules_metadata VALUES "
            "(0, 'C', 16.0), (1, 'CC', 30.0), (2, 'CCC', 44.0), (3, 'CCCC', 58.0)"
        )
        conn.execute(
            "CREATE TABLE assays_metadata (_row_id INTEGER PRIMARY KEY, ic50 DOUBLE)"
        )
        conn.execute(
            "INSERT INTO assays_metadata VALUES (0, 1.5), (1, 8.2)"
        )
        conn.close()

        eng = DuckDBEngine(str(db_file))
        yield eng
        eng.close()

    elif request.param == "pandas":
        df_mol = pd.DataFrame({
            "_row_id": [0, 1, 2, 3],
            "smiles": ["C", "CC", "CCC", "CCCC"],
            "molwt": [16.0, 30.0, 44.0, 58.0],
        })
        df_assay = pd.DataFrame({
            "_row_id": [0, 1],
            "ic50": [1.5, 8.2],
        })
        store = {
            "molecules_metadata": df_mol,
            "assays_metadata": df_assay,
        }
        eng = PandasEngine(_store=store)
        yield eng


# ===========================================================================
# 2. Common engine tests  (run for ALL backends)
# ===========================================================================

class TestCommonEngines:
    """Enforces the BaseEngine contract across all implementations."""

    # --- basic materialise ---

    def test_materialize_all(self, engine):
        df = engine.materialize("molecules", "metadata")
        assert len(df) == 4
        assert df["smiles"].tolist() == ["C", "CC", "CCC", "CCCC"]

    def test_materialize_namespace_isolation(self, engine):
        df = engine.materialize("assays", "metadata")
        assert len(df) == 2
        assert "ic50" in df.columns
        assert "smiles" not in df.columns

    # --- row pushdown ---

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

    def test_pushdown_open_ended_slice(self, engine):
        """slice(2, None) must not crash and must return all rows from index 2 on."""
        df = engine.materialize("molecules", "metadata", rows=slice(2, None))
        assert len(df) == 2
        assert df["_row_id"].tolist() == [2, 3]

    # --- column pushdown ---

    def test_column_selection_single(self, engine):
        df = engine.materialize("molecules", "metadata", cols=["smiles"])
        assert list(df.columns) == ["smiles"]

    def test_column_selection_multiple(self, engine):
        df = engine.materialize("molecules", "metadata", cols=["_row_id", "molwt"])
        assert set(df.columns) == {"_row_id", "molwt"}
        assert "smiles" not in df.columns

    def test_column_selection_with_row_filter(self, engine):
        df = engine.materialize("molecules", "metadata", rows=[0, 2], cols=["smiles"])
        assert list(df.columns) == ["smiles"]
        assert df["smiles"].tolist() == ["C", "CCC"]

    def test_column_selection_none_returns_all(self, engine):
        df = engine.materialize("molecules", "metadata", cols=None)
        assert "smiles" in df.columns
        assert "molwt" in df.columns
        assert "_row_id" in df.columns

    # --- n_rows ---

    def test_n_rows_full(self, engine):
        assert engine.n_rows("molecules", "metadata") == 4

    def test_n_rows_after_write(self, engine):
        engine.write(
            "molecules",
            "metadata",
            pd.DataFrame({"_row_id": [4], "smiles": ["CCCCC"], "molwt": [72.0]}),
        )
        assert engine.n_rows("molecules", "metadata") == 5

    def test_n_rows_missing_table_raises(self, engine):
        with pytest.raises(KeyError):
            engine.n_rows("nonexistent", "table")

    # --- view chaining ---

    def test_spawn_view_basic(self, engine):
        view = engine.spawn_view("molecules", rows=[1, 2, 3])
        df = view.materialize("molecules", "metadata")
        assert df["_row_id"].tolist() == [1, 2, 3]

    def test_spawn_view_with_additional_row_filter(self, engine):
        view = engine.spawn_view("molecules", rows=[1, 2, 3])
        df = view.materialize("molecules", "metadata", rows=slice(0, 2))
        assert df["_row_id"].tolist() == [1, 2]

    def test_spawn_view_chaining(self, engine):
        view1 = engine.spawn_view("molecules", rows=[1, 2, 3])
        view2 = view1.spawn_view("molecules", rows=[1])
        df = view2.materialize("molecules", "metadata")
        assert df["_row_id"].tolist() == [2]

    def test_spawn_view_open_ended_slice(self, engine):
        """Views created with an open-ended slice must not crash."""
        view = engine.spawn_view("molecules", rows=slice(1, None))
        df = view.materialize("molecules", "metadata")
        assert df["_row_id"].tolist() == [1, 2, 3]

    # --- write ---

    def test_write_creates_new_table(self, engine):
        new_data = pd.DataFrame({"_row_id": [0], "name": ["Aspirin"]})
        engine.write("drugs", "properties", new_data)
        df_read = engine.materialize("drugs", "properties")
        assert len(df_read) == 1
        assert df_read["name"].iloc[0] == "Aspirin"

    def test_write_appends_to_existing_table(self, engine):
        append_data = pd.DataFrame({
            "_row_id": [4, 5],
            "smiles": ["C5", "C6"],
            "molwt": [1.0, 2.0],
        })
        engine.write("molecules", "metadata", append_data)
        df_all = engine.materialize("molecules", "metadata")
        assert len(df_all) == 6

    def test_write_raises_on_view(self, engine):
        view = engine.spawn_view("molecules", rows=[0, 1])
        dummy = pd.DataFrame({"_row_id": [99], "smiles": ["C"], "molwt": [16.0]})
        with pytest.raises(PermissionError):
            view.write("molecules", "metadata", dummy)

    # --- schema validation ---

    def test_write_rejects_extra_columns(self, engine):
        bad_data = pd.DataFrame({
            "_row_id": [4],
            "smiles": ["C"],
            "molwt": [16.0],
            "extra_col": ["oops"],
        })
        with pytest.raises(ValueError, match="extra columns"):
            engine.write("molecules", "metadata", bad_data)

    def test_write_rejects_missing_columns(self, engine):
        bad_data = pd.DataFrame({
            "_row_id": [4],
            "smiles": ["C"],
            # molwt is missing
        })
        with pytest.raises(ValueError, match="columns missing"):
            engine.write("molecules", "metadata", bad_data)

    def test_write_accepts_correct_schema(self, engine):
        good_data = pd.DataFrame({
            "_row_id": [4],
            "smiles": ["C"],
            "molwt": [16.0],
        })
        engine.write("molecules", "metadata", good_data)  # should not raise

    # --- export ---

    def test_export_resets_indices(self, engine):
        view = engine.spawn_view("molecules", rows=[2, 3])
        new_engine = view.export()

        df = new_engine.materialize("molecules", "metadata")
        assert len(df) == 2
        assert df["_row_id"].tolist() == [0, 1]
        assert df["smiles"].tolist() == ["CCC", "CCCC"]

    def test_export_copies_all_namespaces(self, engine):
        view = engine.spawn_view("molecules", rows=[2, 3])
        new_engine = view.export()

        # assays namespace should still be present (unmasked)
        df_assays = new_engine.materialize("assays", "metadata")
        assert len(df_assays) == 2

    def test_export_with_namespaces_filter(self, engine):
        new_engine = engine.export(namespaces=["molecules"])

        df_mol = new_engine.materialize("molecules", "metadata")
        assert len(df_mol) == 4

        with pytest.raises(KeyError, match="does not exist"):
            new_engine.materialize("assays", "metadata")


# ===========================================================================
# 3. Engine-specific tests
# ===========================================================================

class TestDuckDBSpecific:

    def test_export_creates_physical_file(self, tmp_path):
        db_file = tmp_path / "root.db"
        target_file = tmp_path / "exported.db"

        eng = DuckDBEngine(str(db_file))
        eng.write("test", "data", pd.DataFrame({"_row_id": [0], "val": [1]}))
        eng.export(target=str(target_file))

        assert os.path.exists(target_file)
        eng.close()

    def test_export_detaches_on_error(self, tmp_path):
        """DETACH must run even if an export step fails (finally block)."""
        db_file = tmp_path / "root.db"
        eng = DuckDBEngine(str(db_file))
        eng.write("test", "data", pd.DataFrame({"_row_id": [0], "val": [1]}))

        target = str(tmp_path / "already_exists.db")
        # Pre-create the target so export raises FileExistsError
        open(target, "w").close()

        with pytest.raises(FileExistsError):
            eng.export(target=target)

        # Connection must still be usable — proves DETACH ran
        result = eng.materialize("test", "data")
        assert len(result) == 1
        eng.close()

    def test_zero_copy_export_from_view(self, tmp_path):
        source_file = tmp_path / "source.db"
        target_file = tmp_path / "target.db"

        eng = DuckDBEngine(str(source_file))
        eng.write(
            "drugs",
            "data",
            pd.DataFrame({"_row_id": [0, 1, 2], "val": [10, 20, 30]}),
        )

        view = eng.spawn_view("drugs", rows=[1, 2])
        new_eng = view.export(target=str(target_file))

        assert os.path.exists(target_file)
        df = new_eng.materialize("drugs", "data")
        assert df["_row_id"].tolist() == [0, 1]
        assert df["val"].tolist() == [20, 30]

        eng.close()
        new_eng.close()


class TestPandasSpecific:

    def test_export_creates_detached_memory(self):
        df = pd.DataFrame({"_row_id": [0], "val": [100]})
        eng = PandasEngine(_store={"test_data": df})

        new_eng = eng.export()
        new_eng.write(
            "test", "data", pd.DataFrame({"_row_id": [1], "val": [200]})
        )

        assert len(eng.materialize("test", "data")) == 1
        assert len(new_eng.materialize("test", "data")) == 2

    def test_missing_column_raises_keyerror(self):
        eng = PandasEngine(
            _store={"mol_meta": pd.DataFrame({"_row_id": [0], "smiles": ["C"]})}
        )
        with pytest.raises(KeyError, match="not found"):
            eng.materialize("mol", "meta", cols=["nonexistent"])


# ===========================================================================
# 4. LocalDB context manager
# ===========================================================================

class TestLocalDBContextManager:

    def test_context_manager_closes_engines(self, tmp_path):
        db_path = str(tmp_path / "test.dldb")

        with LocalDB(db_path) as db:
            eng = db.request_engine("duckdb")
            assert eng is not None

        # After __exit__, _engines should be cleared
        assert len(db._engines) == 0

    def test_context_manager_suppresses_no_exceptions(self, tmp_path):
        db_path = str(tmp_path / "test.dldb")

        with pytest.raises(RuntimeError):
            with LocalDB(db_path) as db:
                raise RuntimeError("boom")

        # Engines still cleared despite the exception
        assert len(db._engines) == 0

    def test_context_manager_creates_directory(self, tmp_path):
        db_path = str(tmp_path / "nested" / "test.dldb")
        assert not os.path.exists(db_path)

        with LocalDB(db_path):
            assert os.path.exists(db_path)


# ===========================================================================
# 5. RestrictedDBProxy — engine caching
# ===========================================================================

class TestRestrictedDBProxyCache:

    def test_engine_cached_on_proxy(self, tmp_path):
        """Two calls to request_engine on the same proxy return the same object."""
        db_path = str(tmp_path / "test.dldb")

        with LocalDB(db_path) as db:
            eng = db.request_engine("duckdb")
            eng.write(
                "mol",
                "meta",
                pd.DataFrame({"_row_id": [0, 1, 2], "smiles": ["C", "CC", "CCC"]}),
            )

            proxy = db.spawn_restricted_view("mol", rows=[0, 1])
            e1 = proxy.request_engine("duckdb")
            e2 = proxy.request_engine("duckdb")
            assert e1 is e2


# ===========================================================================
# 6. BaseTable — history template method
# ===========================================================================

class _SimpleTable(BaseTable[str]):
    """Minimal concrete table for testing BaseTable behaviour."""

    DEFAULT_W2E: ClassVar[Dict[str, str]] = {"data": "pandas"}

    def _extend_impl(self, items: Iterable[str], **kwargs) -> None:
        rows = [{"_row_id": i, "value": v} for i, v in enumerate(items)]
        if rows:
            self._write_to_engine("data", pd.DataFrame(rows))

    def __len__(self) -> int:
        try:
            return len(self.materialize("data"))
        except KeyError:
            return 0


class TestBaseTableHistory:

    def test_extend_records_history(self, tmp_path):
        db_path = str(tmp_path / "test.dldb")
        with LocalDB(db_path) as db:
            table = db.create_table("words", _SimpleTable)
            assert len(table._history) == 0

            table.extend(["alpha", "beta", "gamma"])
            assert len(table._history) == 1

            entry = table._history[0]
            assert isinstance(entry, HistoryEntry)
            assert entry.rows_in == 0
            assert entry.rows_out == 3
            assert "_SimpleTable" in entry.operation

    def test_extend_accumulates_history(self, tmp_path):
        db_path = str(tmp_path / "test.dldb")
        with LocalDB(db_path) as db:
            table = db.create_table("words", _SimpleTable)
            table.extend(["a", "b"])
            table.extend(["c"])
            assert len(table._history) == 2
            assert table._history[1].rows_in == 2
            assert table._history[1].rows_out == 3

    def test_history_shared_across_views(self, tmp_path):
        """A view must share the same History object as its parent."""
        db_path = str(tmp_path / "test.dldb")
        with LocalDB(db_path) as db:
            table = db.create_table("words", _SimpleTable)
            table.extend(["x", "y", "z"])

            view = table.view(rows=[0, 1])
            # The view sees the same history list
            assert view._history is table._history
            assert len(view._history) == 1

    def test_history_entry_has_timestamp(self, tmp_path):
        db_path = str(tmp_path / "test.dldb")
        with LocalDB(db_path) as db:
            table = db.create_table("words", _SimpleTable)
            table.extend(["a"])
            assert table._history[0].timestamp != ""


# ===========================================================================
# 7. BaseTable — column view forwarding
# ===========================================================================

class TestBaseTableColsView:

    def test_view_with_cols_filters_columns(self, tmp_path):
        db_path = str(tmp_path / "test.dldb")
        with LocalDB(db_path) as db:
            eng = db.request_engine("pandas")
            eng.write(
                "mol",
                "data",
                pd.DataFrame({
                    "_row_id": [0, 1],
                    "smiles": ["C", "CC"],
                    "molwt": [16.0, 30.0],
                }),
            )

            table = db.create_table("mol", _SimpleTable)
            # Override _w2e so the table points to the pandas engine
            table._w2e = {"data": "pandas"}

            view = table.view(cols=["smiles"])
            df = view.materialize("data")
            assert list(df.columns) == ["smiles"]

    def test_explicit_cols_override_view_cols(self, tmp_path):
        db_path = str(tmp_path / "test.dldb")
        with LocalDB(db_path) as db:
            eng = db.request_engine("pandas")
            eng.write(
                "mol",
                "data",
                pd.DataFrame({
                    "_row_id": [0, 1],
                    "smiles": ["C", "CC"],
                    "molwt": [16.0, 30.0],
                }),
            )

            table = db.create_table("mol", _SimpleTable)
            table._w2e = {"data": "pandas"}

            view = table.view(cols=["smiles"])
            # Caller explicitly requests molwt — overrides the proxy's cols
            df = view.materialize("data", cols=["molwt"])
            assert list(df.columns) == ["molwt"]


# ===========================================================================
# 8. DEFAULT_W2E class-variable safety
# ===========================================================================

class TestDefaultW2ESafety:

    def test_instance_mutation_does_not_affect_class(self, tmp_path):
        db_path = str(tmp_path / "test.dldb")
        with LocalDB(db_path) as db:
            t1 = db.create_table("a", _SimpleTable)
            t1._w2e["new_key"] = "pandas"  # mutate the instance copy

            # A freshly created table should NOT see "new_key"
            db._tables.pop("a")
            t2 = db.create_table("a", _SimpleTable)
            assert "new_key" not in t2._w2e
            assert "new_key" not in _SimpleTable.DEFAULT_W2E