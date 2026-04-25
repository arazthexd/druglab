from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from druglab.db.table import ConformerTable, HistoryEntry, MoleculeTable, ReactionTable

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem
from rdkit.Chem import AllChem


def _assert_history_round_trip(original, loaded) -> None:
    assert len(loaded.history) == len(original.history)
    for left, right in zip(original.history, loaded.history):
        assert right.block_name == left.block_name
        assert right.rows_in == left.rows_in
        assert right.rows_out == left.rows_out
        assert right.config == left.config


def test_molecule_table_round_trip_persistence(tmp_path):
    table = MoleculeTable.from_smiles(["CCO", "c1ccccc1"])
    table.metadata = pd.DataFrame({"compound_id": ["m1", "m2"], "group": ["a", "b"]})

    fingerprint = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32)
    table.update_feature("fingerprints", fingerprint)
    table.append_history(
        HistoryEntry.now(
            block_name="test.molecule.round_trip",
            config={"source": "pytest"},
            rows_in=len(table),
            rows_out=len(table),
        )
    )

    save_path = tmp_path / "molecule_table"
    table.save(save_path)
    loaded = MoleculeTable.load(save_path)

    assert len(loaded) == len(table)
    pd.testing.assert_frame_equal(loaded.metadata, table.metadata)
    np.testing.assert_array_equal(loaded.get_feature("fingerprints"), table.get_feature("fingerprints"))
    _assert_history_round_trip(table, loaded)

    assert [Chem.MolToSmiles(m) for m in loaded.objects] == [Chem.MolToSmiles(m) for m in table.objects]


def test_reaction_table_round_trip_persistence(tmp_path):
    table = ReactionTable.from_smarts(["CCO>>CC=O", "[CH3:1][OH:2]>>[CH2:1]=[O:2]"])
    table.metadata = pd.DataFrame({"reaction_id": ["r1", "r2"], "category": ["ox", "ox"]})

    rxn_features = np.array([[1, 2], [3, 4]], dtype=np.int32)
    table.update_feature("fingerprints", rxn_features)
    table.append_history(
        HistoryEntry.now(
            block_name="test.reaction.round_trip",
            config={"source": "pytest"},
            rows_in=len(table),
            rows_out=len(table),
        )
    )

    save_path = tmp_path / "reaction_table"
    table.save(save_path)
    loaded = ReactionTable.load(save_path)

    assert len(loaded) == len(table)
    pd.testing.assert_frame_equal(loaded.metadata, table.metadata)
    np.testing.assert_array_equal(loaded.get_feature("fingerprints"), table.get_feature("fingerprints"))
    _assert_history_round_trip(table, loaded)

    assert [AllChem.ReactionToSmarts(r) for r in loaded.objects] == [AllChem.ReactionToSmarts(r) for r in table.objects]


def _make_single_conformer_mol(smiles: str, seed: int):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    status = AllChem.EmbedMolecule(mol, randomSeed=seed)
    assert status == 0
    return mol


def test_conformer_table_round_trip_persistence(tmp_path):
    conf_mols = [
        _make_single_conformer_mol("CCO", 7),
        _make_single_conformer_mol("CCN", 13),
    ]
    table = ConformerTable(
        objects=conf_mols,
        metadata=pd.DataFrame({"conformer_id": ["c1", "c2"], "parent_index": [0, 1]}),
    )

    conf_features = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)
    table.update_feature("fingerprints", conf_features)
    table.append_history(
        HistoryEntry.now(
            block_name="test.conformer.round_trip",
            config={"source": "pytest"},
            rows_in=len(table),
            rows_out=len(table),
        )
    )

    save_path = tmp_path / "conformer_table"
    table.save(save_path)
    loaded = ConformerTable.load(save_path)

    assert len(loaded) == len(table)
    pd.testing.assert_frame_equal(loaded.metadata, table.metadata)
    np.testing.assert_array_equal(loaded.get_feature("fingerprints"), table.get_feature("fingerprints"))
    _assert_history_round_trip(table, loaded)

    assert [Chem.MolToSmiles(m) for m in loaded.objects] == [Chem.MolToSmiles(m) for m in table.objects]
    assert [m.GetNumConformers() for m in loaded.objects] == [m.GetNumConformers() for m in table.objects]