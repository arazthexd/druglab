"""
tests/test_io_db_bridge.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the bridging functionality between druglab.io and druglab.db.
Verifies that IO records can be ingested into DB tables and vice-versa,
and tests the utility properties/functions for extracting raw RDKit objects.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Minimal RDKit stub (no real RDKit required)
# ---------------------------------------------------------------------------

def _install_rdkit_stub() -> None:
    """Inject a thin RDKit shim so druglab imports succeed in CI."""

    class _Mol:
        def __init__(self, raw=None, smiles: str = "C") -> None:
            self._smiles = smiles
        
        def ToBinary(self) -> bytes:
            return b"mol_binary"
            
        def GetNumAtoms(self) -> int:
            return 1

    class _Rxn:
        def __init__(self, raw=None, smarts: str = "[C:1]>>[C:1]O") -> None:
            self._smarts = smarts
            
        def ToBinary(self) -> bytes:
            return b"rxn_binary"
            
        def GetNumReactantTemplates(self) -> int:
            return 1
            
        def GetNumProductTemplates(self) -> int:
            return 1

    rdkit = types.ModuleType("rdkit")
    chem = types.ModuleType("rdkit.Chem")
    allchem = types.ModuleType("rdkit.Chem.AllChem")
    rdchemreactions = types.ModuleType("rdkit.Chem.rdChemReactions")
    descriptors = types.ModuleType("rdkit.Chem.Descriptors")

    chem.Mol = _Mol
    chem.MolFromSmiles = lambda smi, sanitize=True: _Mol(smiles=smi) if smi else None
    chem.MolToSmiles = lambda mol: getattr(mol, "_smiles", "")
    
    allchem.ReactionFromSmarts = lambda sma: _Rxn(smarts=sma) if sma else None
    allchem.ReactionFromRxnFile = lambda p: _Rxn()
    allchem.ReactionToSmarts = lambda rxn: getattr(rxn, "_smarts", "")
    
    rdchemreactions.ChemicalReaction = _Rxn
    rdchemreactions.PreprocessReaction = lambda rxn: []
    
    chem.AllChem = allchem
    chem.rdChemReactions = rdchemreactions
    chem.Descriptors = descriptors
    rdkit.Chem = chem

    for name, mod in [
        ("rdkit", rdkit), 
        ("rdkit.Chem", chem), 
        ("rdkit.Chem.AllChem", allchem),
        ("rdkit.Chem.rdChemReactions", rdchemreactions),
        ("rdkit.Chem.Descriptors", descriptors),
    ]:
        sys.modules.setdefault(name, mod)


_install_rdkit_stub()

# ---------------------------------------------------------------------------
# druglab imports (after stub is in place)
# ---------------------------------------------------------------------------

from druglab.db.table.molecule import MoleculeTable
from druglab.db.table.reaction import ReactionTable
from druglab.io._record import MoleculeRecord, ReactionRecord
from druglab.io.utils import get_mols, get_rxns

# ---------------------------------------------------------------------------
# IO Utilities Tests
# ---------------------------------------------------------------------------

class TestIOUtils:
    def test_get_mols_utility(self):
        from rdkit import Chem
        m1 = Chem.Mol()
        m2 = Chem.Mol()
        
        records = [
            MoleculeRecord(mol=m1),
            MoleculeRecord(mol=None),
            MoleculeRecord(mol=m2)
        ]
        
        # Default behavior: drop invalid
        valid_mols = get_mols(records)
        assert len(valid_mols) == 2
        assert valid_mols == [m1, m2]
        
        # Keep invalid
        all_mols = get_mols(records, drop_invalid=False)
        assert len(all_mols) == 3
        assert all_mols == [m1, None, m2]

    def test_get_rxns_utility(self):
        from rdkit.Chem.rdChemReactions import ChemicalReaction
        r1 = ChemicalReaction()
        
        records = [
            ReactionRecord(rxn=r1),
            ReactionRecord(rxn=None)
        ]
        
        # Default behavior: drop invalid
        valid_rxns = get_rxns(records)
        assert len(valid_rxns) == 1
        assert valid_rxns == [r1]
        
        # Keep invalid
        all_rxns = get_rxns(records, drop_invalid=False)
        assert len(all_rxns) == 2
        assert all_rxns == [r1, None]


# ---------------------------------------------------------------------------
# MoleculeTable Bridges Tests
# ---------------------------------------------------------------------------

class TestMoleculeTableBridge:
    def test_from_records(self):
        from rdkit import Chem
        mol = Chem.Mol()
        
        records = [
            MoleculeRecord(mol=mol, name="CompoundA", index=0, source="file.sdf", properties={"MW": 200.0, "logP": 1.5}),
            MoleculeRecord(mol=None, name="CompoundB", index=1, source="file.sdf", properties={"MW": 250.0}),
        ]
        
        table = MoleculeTable.from_records(records)
        
        assert table.n == 2
        assert len(table.objects) == 2
        assert table.objects[0] is mol
        assert table.objects[1] is None
        
        # Metadata check (ensure everything expanded properly into the dataframe)
        assert "name" in table.metadata.columns
        assert table.metadata["name"].tolist() == ["CompoundA", "CompoundB"]
        assert table.metadata["source"].tolist() == ["file.sdf", "file.sdf"]
        assert table.metadata["index"].tolist() == [0, 1]
        assert table.metadata["MW"].tolist() == [200.0, 250.0]
        
        # Missing property logP for the second element should be NaN
        assert table.metadata["logP"].iloc[0] == 1.5
        assert pd.isna(table.metadata["logP"].iloc[1])

    def test_to_records(self):
        from rdkit import Chem
        mols = [Chem.Mol(), None]
        table = MoleculeTable.from_mols(mols)
        
        # Build metadata similar to what would be found natively
        table.add_metadata_column("name", ["Mol1", "Mol2"])
        table.add_metadata_column("source", ["src1", "src2"])
        table.add_metadata_column("index", [10, 20])
        table.add_metadata_column("score", [0.95, np.nan])
        
        records = table.to_records()
        
        assert len(records) == 2
        assert isinstance(records[0], MoleculeRecord)
        
        # Record 1
        r1 = records[0]
        assert r1.mol is mols[0]
        assert r1.name == "Mol1"
        assert r1.source == "src1"
        assert r1.index == 10
        assert r1.properties["score"] == 0.95
        
        # Record 2
        r2 = records[1]
        assert r2.mol is None
        assert r2.name == "Mol2"
        assert r2.source == "src2"
        assert r2.index == 20
        # The NaN value in score should have been dropped
        assert "score" not in r2.properties

    # IMPORTANT: We patch the target function where it lives globally, because the
    # class methods import it locally inside the method body itself.
    @patch("druglab.io.read_file")
    def test_from_file_delegates_to_io(self, mock_read_file):
        from rdkit import Chem
        mock_read_file.return_value = [
            MoleculeRecord(mol=Chem.Mol(), name="MockMol")
        ]
        
        table = MoleculeTable.from_file("dummy.csv", smiles_col="CanonicalSMILES")
        
        # Check io routing
        mock_read_file.assert_called_once_with("dummy.csv", smiles_col="CanonicalSMILES")
        assert table.n == 1
        assert table.metadata["name"].iloc[0] == "MockMol"

    @patch("druglab.io.write_file")
    def test_to_file_delegates_to_io(self, mock_write_file):
        from rdkit import Chem
        table = MoleculeTable.from_mols([Chem.Mol()])
        table.add_metadata_column("name", ["MolToSave"])
        
        table.to_file("out.sdf", overwrite=True)
        
        # Check io routing
        mock_write_file.assert_called_once()
        args, kwargs = mock_write_file.call_args
        records = args[0]
        assert len(records) == 1
        assert records[0].name == "MolToSave"
        assert args[1] == "out.sdf"
        assert kwargs["overwrite"] is True

    def test_mols_property(self):
        from rdkit import Chem
        m1 = Chem.Mol()
        m2 = Chem.Mol()
        
        table = MoleculeTable.from_mols([m1, None, m2])
        mols = table.mols
        
        # The property should cleanly extract and filter Nones
        assert len(mols) == 2
        assert mols[0] is m1
        assert mols[1] is m2


# ---------------------------------------------------------------------------
# ReactionTable Bridges Tests
# ---------------------------------------------------------------------------

class TestReactionTableBridge:
    def test_from_records(self):
        from rdkit.Chem.rdChemReactions import ChemicalReaction
        rxn = ChemicalReaction()
        
        records = [
            ReactionRecord(rxn=rxn, name="RxnA", properties={"Yield": 95.5}),
            ReactionRecord(rxn=None, name="RxnB", properties={"Yield": 0.0}),
        ]
        
        table = ReactionTable.from_records(records)
        
        assert table.n == 2
        assert table.objects[0] is rxn
        assert table.objects[1] is None
        assert table.metadata["name"].tolist() == ["RxnA", "RxnB"]
        assert table.metadata["Yield"].tolist() == [95.5, 0.0]

    def test_to_records(self):
        from rdkit.Chem.rdChemReactions import ChemicalReaction
        rxns = [ChemicalReaction(), None]
        table = ReactionTable.from_reactions(rxns)
        
        table.add_metadata_column("name", ["R1", "R2"])
        table.add_metadata_column("success", [True, False])
        
        records = table.to_records()
        
        assert len(records) == 2
        assert isinstance(records[0], ReactionRecord)
        
        assert records[0].rxn is rxns[0]
        assert records[0].name == "R1"
        assert records[0].properties["success"] is True
        
        assert records[1].rxn is None
        assert records[1].name == "R2"

    @patch("druglab.io.read_file")
    def test_from_file_delegates_to_io(self, mock_read_file):
        from rdkit.Chem.rdChemReactions import ChemicalReaction
        mock_read_file.return_value = [
            ReactionRecord(rxn=ChemicalReaction(), name="MockRxn")
        ]
        
        table = ReactionTable.from_file("dummy.rxn", sanitize=False)
        
        mock_read_file.assert_called_once_with("dummy.rxn", sanitize=False)
        assert table.n == 1
        assert table.metadata["name"].iloc[0] == "MockRxn"

    @patch("druglab.io.write_file")
    def test_to_file_delegates_to_io(self, mock_write_file):
        from rdkit.Chem.rdChemReactions import ChemicalReaction
        table = ReactionTable.from_reactions([ChemicalReaction()])
        
        table.to_file("out.rxn", overwrite=True)
        
        mock_write_file.assert_called_once()
        args, kwargs = mock_write_file.call_args
        assert isinstance(args[0][0], ReactionRecord)
        assert args[1] == "out.rxn"
        assert kwargs["overwrite"] is True

    def test_rxns_property(self):
        from rdkit.Chem.rdChemReactions import ChemicalReaction
        r1 = ChemicalReaction()
        r2 = ChemicalReaction()
        
        table = ReactionTable.from_reactions([r1, None, r2])
        rxns = table.rxns
        
        assert len(rxns) == 2
        assert rxns == [r1, r2]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])