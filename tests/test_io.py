"""
tests/test_io.py
~~~~~~~~~~~~~~~~
Tests for druglab.io.  Written to run without RDKit by using a
lightweight stub injected into sys.modules before any druglab import.
"""

from __future__ import annotations

import csv
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# ---------------------------------------------------------------------------
# Minimal RDKit stub (no real RDKit required)
# ---------------------------------------------------------------------------

def _install_rdkit_stub() -> None:
    """Inject a thin RDKit shim so druglab.io imports succeed in CI."""

    class _Mol:
        def __init__(self, smiles: str = "C") -> None:
            self._smiles = smiles
            self._props: dict = {"_Name": "mol"}

        def HasProp(self, k: str) -> bool:
            return k in self._props

        def GetProp(self, k: str) -> str:
            return self._props[k]

        def GetPropsAsDict(self) -> dict:
            return dict(self._props)

        def SetProp(self, k: str, v) -> None:
            self._props[k] = v

        def SetDoubleProp(self, k: str, v: float) -> None:
            self._props[k] = v

        def SetIntProp(self, k: str, v: int) -> None:
            self._props[k] = v

    class _Supplier:
        def __init__(self, fh, sanitize=True, removeHs=True) -> None:
            self._mols = [_Mol("C"), _Mol("CC"), None]

        def __iter__(self):
            return iter(self._mols)

    rdkit   = types.ModuleType("rdkit")
    chem    = types.ModuleType("rdkit.Chem")
    allchem = types.ModuleType("rdkit.Chem.AllChem")

    chem.MolFromSmiles        = lambda smi, sanitize=True: _Mol(smi) if smi else None
    chem.MolFromMolFile       = lambda path, sanitize=True, removeHs=True: _Mol()
    chem.MolToSmiles          = lambda mol: mol._smiles if mol else ""
    chem.MolToMolBlock        = lambda mol: (
        "\n\n\n  0  0  0  0  0  0            999 V2000\nM  END\n"
    )
    chem.SDWriter             = MagicMock()
    chem.ForwardSDMolSupplier = _Supplier
    rdkit.Chem                = chem
    rdkit.Chem.AllChem        = allchem

    for name, mod in [("rdkit", rdkit), ("rdkit.Chem", chem), ("rdkit.Chem.AllChem", allchem)]:
        sys.modules.setdefault(name, mod)


_install_rdkit_stub()

# ---------------------------------------------------------------------------
# druglab.io imports (after stub is in place)
# ---------------------------------------------------------------------------

from druglab.io._record import MoleculeRecord, ReactionRecord
from druglab.io.exceptions import DrugLabIOError, RecordParseError, UnsupportedFormatError
from druglab.io.formats import SUPPORTED_FORMATS, detect_format
from druglab.io.readers import (
    BatchReader,
    CSVFormatReader,
    EagerReader,
    SMILESFormatReader,
    read_file,
)
from druglab.io.writers import CSVWriter, SMILESWriter, write_file

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

def _mol(smiles: str = "C") -> object:
    """Return a stub Mol via the shim."""
    from rdkit.Chem import MolFromSmiles
    return MolFromSmiles(smiles)


def _mol_record(smiles: str = "C", name: str = "mol") -> MoleculeRecord:
    return MoleculeRecord(mol=_mol(smiles), name=name, properties={"MW": 16.0})


@pytest.fixture
def smiles_file(tmp_path: Path):
    """Factory fixture: returns a callable that writes a .smi file."""
    def _make(content: str) -> Path:
        p = tmp_path / "molecules.smi"
        p.write_text(content, encoding="utf-8")
        return p
    return _make


@pytest.fixture
def csv_file(tmp_path: Path):
    """Factory fixture: returns a callable that writes a .csv or .tsv file."""
    def _make(content: str, suffix: str = ".csv") -> Path:
        p = tmp_path / f"data{suffix}"
        p.write_text(content, encoding="utf-8")
        return p
    return _make


@pytest.fixture
def n_smiles_file(tmp_path: Path):
    """Factory fixture: returns a callable that writes n stub SMILES lines."""
    def _make(n: int, stem: str = "mols") -> Path:
        p = tmp_path / f"{stem}.smi"
        p.write_text("".join(f"C mol{i}\n" for i in range(n)), encoding="utf-8")
        return p
    return _make


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TestExceptions:
    def test_unsupported_format_error_message(self):
        err = UnsupportedFormatError("xyz")
        assert "xyz" in str(err)
        assert err.fmt == "xyz"

    def test_record_parse_error_attributes(self):
        err = RecordParseError("file.sdf", 3, "bad mol")
        assert "file.sdf" in str(err)
        assert "3" in str(err)
        assert err.source == "file.sdf"
        assert err.record_index == 3
        assert err.reason == "bad mol"

    def test_druglab_io_error_is_ioerror(self):
        assert issubclass(DrugLabIOError, IOError)

    def test_unsupported_format_error_is_druglab_io_error(self):
        assert issubclass(UnsupportedFormatError, DrugLabIOError)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

class TestFormatDetection:
    @pytest.mark.parametrize("path,expected", [
        ("compounds.sdf", "sdf"),
        ("data.CSV",      "csv"),
        ("rxns.RXN",      "rxn"),
        ("mol.smi",       "smi"),
        ("batch.smiles",  "smiles"),
        ("sheet.tsv",     "tsv"),
        ("struct.mol",    "mol"),
    ])
    def test_known_formats(self, path: str, expected: str):
        assert detect_format(path) == expected

    def test_unknown_extension_raises(self):
        with pytest.raises(UnsupportedFormatError):
            detect_format("molecule.xyz")

    def test_no_extension_raises(self):
        with pytest.raises(UnsupportedFormatError):
            detect_format("noextfile")

    def test_supported_formats_contains_core_entries(self):
        for fmt in ("sdf", "csv", "smi", "rxn", "mol"):
            assert fmt in SUPPORTED_FORMATS


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

class TestMoleculeRecord:
    def test_valid_record(self):
        rec = MoleculeRecord(mol=_mol(), name="methane", source="test.smi", index=0)
        assert rec.is_valid()
        assert "methane" in repr(rec)
        assert rec.source == "test.smi"
        assert rec.index == 0

    def test_invalid_record_is_not_valid(self):
        rec = MoleculeRecord(mol=None)
        assert not rec.is_valid()

    def test_to_smiles_returns_none_for_invalid(self):
        rec = MoleculeRecord(mol=None)
        assert rec.to_smiles() is None

    def test_to_smiles_returns_string_for_valid(self):
        rec = MoleculeRecord(mol=_mol("CC"), name="ethane")
        result = rec.to_smiles()
        assert isinstance(result, str)

    def test_properties_stored(self):
        rec = MoleculeRecord(mol=_mol(), properties={"MW": 16.0, "logP": 0.5})
        assert rec.properties["MW"] == 16.0
        assert rec.properties["logP"] == 0.5


class TestReactionRecord:
    def test_invalid_record(self):
        rec = ReactionRecord(rxn=None, name="r1")
        assert not rec.is_valid()
        assert rec.n_reactants() == 0
        assert rec.n_products() == 0

    def test_name_stored(self):
        rec = ReactionRecord(rxn=None, name="retro_step_1")
        assert rec.name == "retro_step_1"


# ---------------------------------------------------------------------------
# SMILES format reader
# ---------------------------------------------------------------------------

class TestSMILESFormatReader:
    def test_basic_reading(self, smiles_file):
        path = smiles_file("C methane\nCC ethane\n")
        records = list(SMILESFormatReader().iter_records(str(path)))
        assert len(records) == 2
        assert records[0].name == "methane"
        assert records[1].name == "ethane"
        assert records[0].source == str(path)
        assert records[0].index == 0
        assert records[1].index == 1

    def test_smiles_stored_in_properties(self, smiles_file):
        path = smiles_file("C methane\n")
        records = list(SMILESFormatReader().iter_records(str(path)))
        assert "SMILES" in records[0].properties

    def test_comments_are_skipped(self, smiles_file):
        path = smiles_file("# this is a comment\nC methane\n")
        records = list(SMILESFormatReader().iter_records(str(path)))
        assert len(records) == 1
        assert records[0].name == "methane"

    def test_blank_lines_are_skipped(self, smiles_file):
        path = smiles_file("\nC methane\n\nCC ethane\n")
        records = list(SMILESFormatReader().iter_records(str(path)))
        assert len(records) == 2

    def test_nameless_smiles(self, smiles_file):
        path = smiles_file("C\nCC\n")
        records = list(SMILESFormatReader(name_col=None).iter_records(str(path)))
        assert all(r.name == "" for r in records)

    def test_on_error_skip_does_not_raise(self, smiles_file):
        path = smiles_file("C mol0\nCC mol1\n")
        records = list(SMILESFormatReader(on_error="skip").iter_records(str(path)))
        assert len(records) >= 0

    def test_read_all_convenience(self, smiles_file):
        path = smiles_file("C a\nCC b\nCCC c\n")
        records = SMILESFormatReader().read_all(str(path))
        assert isinstance(records, list)
        assert len(records) == 3


# ---------------------------------------------------------------------------
# CSV format reader
# ---------------------------------------------------------------------------

class TestCSVFormatReader:
    def test_basic_reading(self, csv_file):
        path = csv_file("SMILES,Name,MW\nC,methane,16\nCC,ethane,30\n")
        records = list(CSVFormatReader(smiles_col="SMILES", name_col="Name").iter_records(str(path)))
        assert len(records) == 2
        assert records[0].name == "methane"
        assert records[1].name == "ethane"

    def test_properties_populated(self, csv_file):
        path = csv_file("SMILES,Name,MW,logP\nC,methane,16,0.1\n")
        records = list(CSVFormatReader(smiles_col="SMILES", name_col="Name").iter_records(str(path)))
        assert "MW" in records[0].properties
        assert "logP" in records[0].properties

    def test_missing_smiles_column_raises(self, csv_file):
        path = csv_file("canonical,Name\nC,methane\n")
        with pytest.raises(DrugLabIOError, match="SMILES"):
            list(CSVFormatReader(smiles_col="SMILES").iter_records(str(path)))

    def test_tsv_auto_delimiter(self, csv_file):
        path = csv_file("SMILES\tName\nC\tmethane\n", suffix=".tsv")
        records = list(CSVFormatReader(smiles_col="SMILES", name_col="Name").iter_records(str(path)))
        assert len(records) == 1
        assert records[0].name == "methane"

    def test_source_and_index_set(self, csv_file):
        path = csv_file("SMILES,Name\nC,a\nCC,b\n")
        records = list(CSVFormatReader(smiles_col="SMILES").iter_records(str(path)))
        assert records[0].source == str(path)
        assert records[0].index == 0
        assert records[1].index == 1

    def test_no_name_col(self, csv_file):
        path = csv_file("SMILES,MW\nC,16\n")
        records = list(CSVFormatReader(smiles_col="SMILES", name_col=None).iter_records(str(path)))
        assert records[0].name == ""


# ---------------------------------------------------------------------------
# BatchReader
# ---------------------------------------------------------------------------

class TestBatchReader:
    def test_batches_cover_all_records(self, n_smiles_file):
        path = n_smiles_file(10)
        batches = list(BatchReader(str(path), batch_size=3))
        assert sum(len(b) for b in batches) == 10

    def test_all_full_batches_have_correct_size(self, n_smiles_file):
        path = n_smiles_file(10)
        batches = list(BatchReader(str(path), batch_size=3))
        for batch in batches[:-1]:
            assert len(batch) == 3

    def test_last_batch_is_remainder(self, n_smiles_file):
        path = n_smiles_file(10)
        batches = list(BatchReader(str(path), batch_size=3))
        assert len(batches[-1]) == 1  # 10 % 3 == 1

    def test_single_batch_when_size_exceeds_records(self, n_smiles_file):
        path = n_smiles_file(5)
        batches = list(BatchReader(str(path), batch_size=100))
        assert len(batches) == 1
        assert len(batches[0]) == 5

    def test_iter_records_yields_individual_records(self, n_smiles_file):
        path = n_smiles_file(5)
        records = list(BatchReader(str(path), batch_size=2).iter_records())
        assert len(records) == 5
        assert all(isinstance(r, MoleculeRecord) for r in records)

    def test_collect_returns_flat_list(self, n_smiles_file):
        path = n_smiles_file(7)
        records = BatchReader(str(path), batch_size=3).collect()
        assert isinstance(records, list)
        assert len(records) == 7

    def test_multi_file_concatenation(self, n_smiles_file):
        p1 = n_smiles_file(3, stem="a")
        p2 = n_smiles_file(4, stem="b")
        records = BatchReader([str(p1), str(p2)], batch_size=100).collect()
        assert len(records) == 7

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError):
            BatchReader("x.smi", batch_size=0)

    def test_repr_contains_class_name_and_batch_size(self):
        r = BatchReader("x.smi", batch_size=50)
        assert "BatchReader" in repr(r)
        assert "50" in repr(r)

    def test_n_batches_estimate(self, n_smiles_file):
        path = n_smiles_file(10)
        r = BatchReader(str(path), batch_size=3)
        assert r.n_batches_estimate(10) == 4  # ceil(10/3)


# ---------------------------------------------------------------------------
# EagerReader
# ---------------------------------------------------------------------------

class TestEagerReader:
    def test_read_returns_all_records(self, n_smiles_file):
        path = n_smiles_file(7)
        assert len(EagerReader(str(path)).read()) == 7

    def test_repeated_read_returns_same_object(self, n_smiles_file):
        path = n_smiles_file(3)
        reader = EagerReader(str(path))
        first = reader.read()
        assert reader.read() is first  # cached

    def test_len(self, n_smiles_file):
        path = n_smiles_file(4)
        assert len(EagerReader(str(path))) == 4

    def test_getitem(self, n_smiles_file):
        path = n_smiles_file(4)
        reader = EagerReader(str(path))
        assert isinstance(reader[0], MoleculeRecord)
        assert isinstance(reader[-1], MoleculeRecord)

    def test_iterable(self, n_smiles_file):
        path = n_smiles_file(3)
        assert len(list(EagerReader(str(path)))) == 3

    def test_invalidate_cache_clears_records(self, n_smiles_file):
        path = n_smiles_file(2)
        reader = EagerReader(str(path))
        reader.read()
        reader.invalidate_cache()
        assert reader._records is None

    def test_repr_before_load_says_not_loaded(self):
        assert "not loaded" in repr(EagerReader("x.smi"))

    def test_repr_after_load_shows_count(self, n_smiles_file):
        path = n_smiles_file(3)
        reader = EagerReader(str(path))
        reader.read()
        assert "3" in repr(reader)


# ---------------------------------------------------------------------------
# read_file convenience
# ---------------------------------------------------------------------------

class TestReadFile:
    def test_smiles_file(self, smiles_file):
        path = smiles_file("C methane\nCC ethane\n")
        assert len(read_file(str(path))) == 2

    def test_csv_file(self, csv_file):
        path = csv_file("SMILES,Name\nC,methane\nCC,ethane\n")
        assert len(read_file(str(path), smiles_col="SMILES")) == 2

    def test_unsupported_format_raises(self):
        with pytest.raises(UnsupportedFormatError):
            read_file("molecule.xyz")

    def test_kwargs_forwarded_to_reader(self, csv_file):
        path = csv_file("canonical_smi,Name\nC,methane\n")
        assert len(read_file(str(path), smiles_col="canonical_smi")) == 1


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

class TestSMILESWriter:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "out.smi"
        with SMILESWriter(str(path), overwrite=True) as w:
            w.write(_mol_record())
        assert path.exists()

    def test_content_contains_name(self, tmp_path):
        path = tmp_path / "out.smi"
        with SMILESWriter(str(path), overwrite=True) as w:
            w.write(_mol_record(name="caffeine"))
        assert "caffeine" in path.read_text()

    def test_write_many_returns_count(self, tmp_path):
        path = tmp_path / "out.smi"
        with SMILESWriter(str(path), overwrite=True) as w:
            n = w.write_many([_mol_record(), _mol_record(), _mol_record()])
        assert n == 3

    def test_overwrite_false_raises_if_exists(self, tmp_path):
        path = tmp_path / "out.smi"
        path.write_text("existing")
        with pytest.raises(DrugLabIOError):
            with SMILESWriter(str(path), overwrite=False) as w:
                w.write(_mol_record())

    def test_reaction_record_raises_type_error(self, tmp_path):
        path = tmp_path / "out.smi"
        with pytest.raises(TypeError):
            with SMILESWriter(str(path), overwrite=True) as w:
                w.write(ReactionRecord(rxn=None))

    def test_invalid_mol_is_skipped(self, tmp_path):
        path = tmp_path / "out.smi"
        with SMILESWriter(str(path), overwrite=True) as w:
            w.write(MoleculeRecord(mol=None, name="bad"))
        assert path.read_text() == ""


class TestCSVWriter:
    def test_creates_file_with_header(self, tmp_path):
        path = tmp_path / "out.csv"
        with CSVWriter(str(path), overwrite=True) as w:
            w.write(_mol_record())
        rows = list(csv.DictReader(path.open()))
        assert "SMILES" in rows[0]

    def test_name_column_written(self, tmp_path):
        path = tmp_path / "out.csv"
        with CSVWriter(str(path), overwrite=True) as w:
            w.write(_mol_record(name="aspirin"))
        rows = list(csv.DictReader(path.open()))
        assert rows[0]["Name"] == "aspirin"

    def test_properties_written_as_columns(self, tmp_path):
        path = tmp_path / "out.csv"
        rec = MoleculeRecord(mol=_mol(), name="x", properties={"MW": 16.0, "logP": 0.5})
        with CSVWriter(str(path), overwrite=True) as w:
            w.write(rec)
        rows = list(csv.DictReader(path.open()))
        assert "MW" in rows[0]

    def test_multiple_records(self, tmp_path):
        path = tmp_path / "out.csv"
        with CSVWriter(str(path), overwrite=True) as w:
            w.write_many([_mol_record("C", "a"), _mol_record("CC", "b")])
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 2

    def test_overwrite_false_raises_if_exists(self, tmp_path):
        path = tmp_path / "out.csv"
        path.write_text("existing")
        with pytest.raises(DrugLabIOError):
            with CSVWriter(str(path), overwrite=False) as w:
                w.write(_mol_record())

    def test_reaction_record_raises_type_error(self, tmp_path):
        path = tmp_path / "out.csv"
        with pytest.raises(TypeError):
            with CSVWriter(str(path), overwrite=True) as w:
                w.write(ReactionRecord(rxn=None))


# ---------------------------------------------------------------------------
# write_file convenience
# ---------------------------------------------------------------------------

class TestWriteFile:
    def test_smiles_roundtrip_count(self, tmp_path):
        path = tmp_path / "out.smi"
        assert write_file([_mol_record(), _mol_record()], str(path), overwrite=True) == 2

    def test_csv_roundtrip_count(self, tmp_path):
        path = tmp_path / "out.csv"
        assert write_file([_mol_record()] * 3, str(path), overwrite=True) == 3

    def test_single_record_accepted(self, tmp_path):
        path = tmp_path / "out.smi"
        assert write_file(_mol_record(), str(path), overwrite=True) == 1

    def test_unsupported_format_raises(self, tmp_path):
        path = tmp_path / "out.xyz"
        with pytest.raises(UnsupportedFormatError):
            write_file([_mol_record()], str(path))