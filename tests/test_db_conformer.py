"""
tests/test_db_conformer.py
~~~~~~~~~~~~~~~~~~~~~~~
Tests for ConformerTable and the conformer bridging methods on MoleculeTable.

These tests require a real RDKit installation because they exercise actual
3D coordinate manipulation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make sure the local src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# RDKit availability guard
# ---------------------------------------------------------------------------

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    _RDKIT = True
except ImportError:
    _RDKIT = False

pytestmark = pytest.mark.skipif(not _RDKIT, reason="RDKit not installed")

from druglab.db.molecule import MoleculeTable
from druglab.db.conformer import ConformerTable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mol_with_conformers(smiles: str, n_confs: int = 3, seed: int = 42) -> "Chem.Mol":
    """Return an RDKit Mol with ``n_confs`` embedded 3D conformers."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Bad SMILES: {smiles}"
    mol = Chem.AddHs(mol)
    params = AllChem.EmbedMultipleConfs(
        mol, numConfs=n_confs, randomSeed=seed, numThreads=1
    )
    assert mol.GetNumConformers() > 0, "Embedding failed"
    return mol


def _simple_3d_table(n_mols: int = 2, n_confs_each: int = 3) -> MoleculeTable:
    """Build a small MoleculeTable with 3D conformers."""
    smiles_list = ["CCO", "c1ccccc1", "CC(=O)O", "CCC"]
    mols = [
        _mol_with_conformers(smiles_list[i % len(smiles_list)], n_confs=n_confs_each)
        for i in range(n_mols)
    ]
    meta = pd.DataFrame({
        "name": [f"mol_{i}" for i in range(n_mols)],
        "activity": [float(i) for i in range(n_mols)],
    })
    return MoleculeTable(objects=mols, metadata=meta)


# ===========================================================================
# Section 1: unroll_conformers
# ===========================================================================

class TestUnrollConformers:

    def test_row_count(self):
        """Each conformer becomes exactly one row."""
        table = _simple_3d_table(n_mols=2, n_confs_each=3)
        conf_table = table.unroll_conformers()
        assert conf_table.n == 6  # 2 mols × 3 confs

    def test_returns_conformer_table_type(self):
        table = _simple_3d_table(n_mols=2, n_confs_each=2)
        conf_table = table.unroll_conformers()
        assert isinstance(conf_table, ConformerTable)

    def test_each_object_has_exactly_one_conformer(self):
        table = _simple_3d_table(n_mols=2, n_confs_each=3)
        conf_table = table.unroll_conformers()
        for mol in conf_table.objects:
            assert mol is not None
            assert mol.GetNumConformers() == 1

    def test_parent_index_column_created(self):
        table = _simple_3d_table(n_mols=2, n_confs_each=2)
        conf_table = table.unroll_conformers()
        assert "parent_index" in conf_table.metadata.columns

    def test_conf_id_column_created(self):
        table = _simple_3d_table(n_mols=2, n_confs_each=2)
        conf_table = table.unroll_conformers()
        assert "conf_id" in conf_table.metadata.columns

    def test_custom_id_col_name(self):
        table = _simple_3d_table(n_mols=1, n_confs_each=2)
        conf_table = table.unroll_conformers(id_col="mol_id")
        assert "mol_id" in conf_table.metadata.columns
        assert "parent_index" not in conf_table.metadata.columns

    def test_parent_metadata_duplicated(self):
        table = _simple_3d_table(n_mols=2, n_confs_each=3)
        conf_table = table.unroll_conformers()
        # mol_0 should appear 3 times (once per conformer)
        mol0_rows = conf_table.metadata[conf_table.metadata["parent_index"] == 0]
        assert len(mol0_rows) == 3
        assert all(mol0_rows["name"] == "mol_0")

    def test_features_empty(self):
        """Features must not be copied to prevent OOM on large datasets."""
        table = _simple_3d_table(n_mols=2, n_confs_each=2)
        # Add a heavy feature to the parent
        table.add_feature("fp", np.ones((2, 1024)))
        conf_table = table.unroll_conformers()
        assert conf_table.features == {}

    def test_history_entry_added(self):
        table = _simple_3d_table(n_mols=1, n_confs_each=2)
        conf_table = table.unroll_conformers()
        block_names = [e.block_name for e in conf_table.history]
        assert "MoleculeTable.unroll_conformers" in block_names

    def test_none_mol_skipped(self):
        """None entries in the parent table are silently skipped."""
        mol = _mol_with_conformers("CCO", n_confs=2)
        table = MoleculeTable(
            objects=[None, mol],
            metadata=pd.DataFrame({"name": ["bad", "good"]}),
        )
        conf_table = table.unroll_conformers()
        assert conf_table.n == 2  # only from the valid mol

    def test_mol_with_no_conformers_skipped(self):
        mol_2d = Chem.MolFromSmiles("CCO")  # 2D, no conformer
        mol_3d = _mol_with_conformers("CCO", n_confs=2)
        table = MoleculeTable(
            objects=[mol_2d, mol_3d],
            metadata=pd.DataFrame({"name": ["flat", "3d"]}),
        )
        conf_table = table.unroll_conformers()
        assert conf_table.n == 2  # only from mol_3d

    def test_original_table_unchanged(self):
        """unroll_conformers must not modify the parent table."""
        table = _simple_3d_table(n_mols=2, n_confs_each=3)
        n_before = table.n
        conf_before = table.objects[0].GetNumConformers()
        _ = table.unroll_conformers()
        assert table.n == n_before
        assert table.objects[0].GetNumConformers() == conf_before

    def test_coordinates_are_distinct(self):
        """Each single-conformer copy must carry the correct coordinates."""
        mol = _mol_with_conformers("c1ccccc1", n_confs=3)
        table = MoleculeTable(
            objects=[mol],
            metadata=pd.DataFrame({"name": ["benzene"]}),
        )
        conf_table = table.unroll_conformers()

        # Collect the first-atom positions from each row
        positions = []
        for m in conf_table.objects:
            pos = m.GetConformer(0).GetAtomPosition(0)
            positions.append((pos.x, pos.y, pos.z))

        # The three conformers should generally have different coordinates
        # (they're randomly seeded embeddings)
        assert len(set(positions)) > 1


# ===========================================================================
# Section 2: ConformerTable.collapse
# ===========================================================================

class TestCollapse:

    def test_roundtrip_row_count(self):
        """collapse() should restore one row per parent molecule."""
        table = _simple_3d_table(n_mols=3, n_confs_each=4)
        conf_table = table.unroll_conformers()
        collapsed = conf_table.collapse(groupby="parent_index")
        assert collapsed.n == 3

    def test_returns_molecule_table_type(self):
        table = _simple_3d_table(n_mols=2, n_confs_each=2)
        conf_table = table.unroll_conformers()
        collapsed = conf_table.collapse()
        assert type(collapsed) is MoleculeTable

    def test_conformer_count_restored(self):
        """Each molecule in the collapsed table should regain all conformers."""
        n_confs = 4
        table = _simple_3d_table(n_mols=2, n_confs_each=n_confs)
        conf_table = table.unroll_conformers()
        collapsed = conf_table.collapse()
        for mol in collapsed.objects:
            assert mol is not None
            assert mol.GetNumConformers() == n_confs

    def test_metadata_aggregation_min(self):
        """metadata_agg rules should be applied correctly."""
        mol = _mol_with_conformers("CCO", n_confs=3)
        table = MoleculeTable(
            objects=[mol],
            metadata=pd.DataFrame({"name": ["ethanol"]}),
        )
        conf_table = table.unroll_conformers()
        # Inject fake energies into the ConformerTable metadata
        conf_table.add_metadata_column("Energy", [10.0, 3.0, 7.0])
        collapsed = conf_table.collapse(metadata_agg={"Energy": "min"})
        assert collapsed.metadata["Energy"].iloc[0] == pytest.approx(3.0)

    def test_metadata_aggregation_mean(self):
        mol = _mol_with_conformers("CCO", n_confs=4)
        table = MoleculeTable(
            objects=[mol],
            metadata=pd.DataFrame({"name": ["ethanol"]}),
        )
        conf_table = table.unroll_conformers()
        conf_table.add_metadata_column("Score", [0.0, 2.0, 4.0, 6.0])
        collapsed = conf_table.collapse(metadata_agg={"Score": "mean"})
        assert collapsed.metadata["Score"].iloc[0] == pytest.approx(3.0)

    def test_features_aggregation(self):
        """features_agg rules should correctly stack and aggregate arrays."""
        mol = _mol_with_conformers("CCO", n_confs=3)
        table = MoleculeTable(
            objects=[mol],
            metadata=pd.DataFrame({"name": ["ethanol"]}),
        )
        conf_table = table.unroll_conformers()
        # Add fake 3D features manually
        fake_feats = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        conf_table.add_feature("desc3d", fake_feats)

        collapsed = conf_table.collapse(features_agg={"desc3d": "mean"})
        expected = fake_feats.mean(axis=0)
        assert np.allclose(collapsed.features["desc3d"][0], expected)

    def test_history_entry_added(self):
        table = _simple_3d_table(n_mols=1, n_confs_each=2)
        conf_table = table.unroll_conformers()
        collapsed = conf_table.collapse()
        block_names = [e.block_name for e in collapsed.history]
        assert "ConformerTable.collapse" in block_names

    def test_invalid_groupby_column_raises(self):
        table = _simple_3d_table(n_mols=1, n_confs_each=2)
        conf_table = table.unroll_conformers()
        with pytest.raises(ValueError, match="nonexistent"):
            conf_table.collapse(groupby="nonexistent")

    def test_partial_collapse_after_filter(self):
        """Collapse after filtering some conformers out."""
        table = _simple_3d_table(n_mols=2, n_confs_each=4)
        conf_table = table.unroll_conformers()

        # Keep only conformers whose conf_id < 2 (first two per molecule)
        mask = conf_table.metadata["conf_id"] < 2
        filtered = conf_table.subset(np.where(mask)[0])

        collapsed = filtered.collapse(groupby="parent_index")
        assert collapsed.n == 2
        for mol in collapsed.objects:
            assert mol.GetNumConformers() == 2


# ===========================================================================
# Section 3: MoleculeTable.update_from_conformers
# ===========================================================================

class TestUpdateFromConformers:

    def test_returns_new_molecule_table(self):
        table = _simple_3d_table(n_mols=2, n_confs_each=3)
        conf_table = table.unroll_conformers()
        result = table.update_from_conformers(conf_table)
        assert isinstance(result, MoleculeTable)
        assert result is not table

    def test_original_table_unchanged(self):
        table = _simple_3d_table(n_mols=2, n_confs_each=3)
        conf_table = table.unroll_conformers()
        n_before = table.n
        _ = table.update_from_conformers(conf_table)
        assert table.n == n_before

    def test_conformers_round_trip(self):
        """All conformers should be restored when no filtering occurs."""
        n_confs = 3
        table = _simple_3d_table(n_mols=2, n_confs_each=n_confs)
        conf_table = table.unroll_conformers()
        result = table.update_from_conformers(conf_table)
        for mol in result.objects:
            assert mol.GetNumConformers() == n_confs

    def test_drop_empty_true(self):
        """Molecules whose conformers were all filtered out are dropped."""
        table = _simple_3d_table(n_mols=3, n_confs_each=3)
        conf_table = table.unroll_conformers()

        # Keep only conformers from parent molecule 0 and 2
        mask = conf_table.metadata["parent_index"].isin([0, 2])
        filtered = conf_table.subset(np.where(mask)[0])

        result = table.update_from_conformers(filtered, drop_empty=True)
        assert result.n == 2

    def test_drop_empty_false(self):
        """Molecules whose conformers were filtered are kept with empty conformers."""
        table = _simple_3d_table(n_mols=3, n_confs_each=3)
        conf_table = table.unroll_conformers()

        # Keep only conformers from parent molecule 0
        mask = conf_table.metadata["parent_index"] == 0
        filtered = conf_table.subset(np.where(mask)[0])

        result = table.update_from_conformers(filtered, drop_empty=False)
        assert result.n == 3
        # Mol 1 and 2 should have 0 conformers
        assert result.objects[1].GetNumConformers() == 0
        assert result.objects[2].GetNumConformers() == 0

    def test_metadata_aggregation_written_back(self):
        """Aggregated metadata from conf_table should be merged into parent."""
        mol = _mol_with_conformers("CCO", n_confs=3)
        table = MoleculeTable(
            objects=[mol],
            metadata=pd.DataFrame({"name": ["ethanol"], "energy": [999.0]}),
        )
        conf_table = table.unroll_conformers()
        conf_table.add_metadata_column("energy", [5.0, 2.0, 8.0])

        result = table.update_from_conformers(
            conf_table, metadata_agg={"energy": "min"}
        )
        assert result.metadata["energy"].iloc[0] == pytest.approx(2.0)

    def test_feature_aggregation_written_back(self):
        """Aggregated features from conf_table should appear in result."""
        mol = _mol_with_conformers("CCO", n_confs=3)
        table = MoleculeTable(
            objects=[mol],
            metadata=pd.DataFrame({"name": ["ethanol"]}),
        )
        conf_table = table.unroll_conformers()
        fake_feats = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
        conf_table.add_feature("shape", fake_feats)

        result = table.update_from_conformers(
            conf_table, features_agg={"shape": "mean"}
        )
        assert "shape" in result.features
        expected = fake_feats.mean(axis=0)
        assert np.allclose(result.features["shape"][0], expected)

    def test_history_entry_added(self):
        table = _simple_3d_table(n_mols=1, n_confs_each=2)
        conf_table = table.unroll_conformers()
        result = table.update_from_conformers(conf_table)
        block_names = [e.block_name for e in result.history]
        assert "MoleculeTable.update_from_conformers" in block_names

    def test_invalid_id_col_raises(self):
        table = _simple_3d_table(n_mols=1, n_confs_each=2)
        conf_table = table.unroll_conformers()
        with pytest.raises(ValueError, match="bad_col"):
            table.update_from_conformers(conf_table, id_col="bad_col")

    def test_filtering_reduces_conformer_count(self):
        """Filtering down to 1 conformer per molecule updates counts correctly."""
        table = _simple_3d_table(n_mols=2, n_confs_each=4)
        conf_table = table.unroll_conformers()

        # Keep only the first conformer per molecule
        mask = conf_table.metadata["conf_id"] == conf_table.metadata["conf_id"].min()
        filtered = conf_table.subset(np.where(mask)[0])

        result = table.update_from_conformers(filtered)
        assert result.n == 2
        for mol in result.objects:
            assert mol.GetNumConformers() == 1


# ===========================================================================
# Section 4: ConformerTable as a MoleculeTable subclass
# ===========================================================================

class TestConformerTableInheritance:

    def test_is_molecule_table_subclass(self):
        assert issubclass(ConformerTable, MoleculeTable)

    def test_pipeline_works_on_conformer_table(self):
        """Standard druglab.pipe blocks should work on a ConformerTable."""
        from druglab.pipe import Pipeline, FunctionFilter

        table = _simple_3d_table(n_mols=3, n_confs_each=3)
        conf_table = table.unroll_conformers()

        # Keep only conformers with conf_id == 0
        pipe = Pipeline([
            FunctionFilter(func=lambda mol: mol is not None)
        ])
        result = pipe.run(conf_table)
        assert isinstance(result, ConformerTable)

    def test_subset_preserves_type(self):
        table = _simple_3d_table(n_mols=2, n_confs_each=3)
        conf_table = table.unroll_conformers()
        sub = conf_table.subset([0, 1, 2])
        assert isinstance(sub, ConformerTable)

    def test_concat_preserves_type(self):
        table = _simple_3d_table(n_mols=2, n_confs_each=2)
        ct1 = table.unroll_conformers()
        ct2 = table.unroll_conformers()
        combined = ConformerTable.concat([ct1, ct2])
        assert isinstance(combined, ConformerTable)

    def test_no_reference_to_parent(self):
        """ConformerTable must not store a reference to the parent table."""
        table = _simple_3d_table(n_mols=2, n_confs_each=2)
        conf_table = table.unroll_conformers()
        # Deleting the parent should not affect the ConformerTable
        del table
        assert conf_table.n == 4

    def test_decoupled_from_parent(self):
        """Mutating the parent table should not affect the ConformerTable."""
        table = _simple_3d_table(n_mols=2, n_confs_each=2)
        conf_table = table.unroll_conformers()
        # Mutate parent metadata
        metadata = table.metadata
        metadata.loc[0, "name"] = "MUTATED"
        table.metadata = metadata
        # ConformerTable should still have original values
        assert "MUTATED" not in conf_table.metadata["name"].values


# ===========================================================================
# Section 5: Full end-to-end workflow
# ===========================================================================

class TestEndToEnd:

    def test_full_explode_filter_collapse_workflow(self):
        """
        Realistic workflow:
        1. Build 3D table
        2. Unroll conformers
        3. Filter to keep only the first 2 conformers per molecule
        4. Collapse back
        """
        n_mols = 3
        n_confs = 5
        table = _simple_3d_table(n_mols=n_mols, n_confs_each=n_confs)
        conf_table = table.unroll_conformers()

        assert conf_table.n == n_mols * n_confs

        # Filter: keep only conf_ids 0 and 1
        mask = conf_table.metadata["conf_id"].isin([0, 1])
        filtered = conf_table.subset(np.where(mask)[0])

        assert filtered.n == n_mols * 2

        collapsed = filtered.collapse(groupby="parent_index")
        assert collapsed.n == n_mols
        for mol in collapsed.objects:
            assert mol.GetNumConformers() == 2

    def test_full_explode_filter_update_workflow(self):
        """
        Realistic workflow using update_from_conformers instead of collapse.
        """
        n_mols = 3
        n_confs = 4
        table = _simple_3d_table(n_mols=n_mols, n_confs_each=n_confs)

        # Add some fake energy data
        table.add_metadata_column("best_energy", [0.0] * n_mols)
        conf_table = table.unroll_conformers()

        # Inject fake energies per conformer
        rng = np.random.default_rng(seed=0)
        energies = rng.uniform(0, 10, size=conf_table.n).tolist()
        conf_table.add_metadata_column("energy", energies)

        # Filter: drop highest-energy conformers (keep energy < 8)
        mask = np.array(energies) < 8.0
        filtered = conf_table.subset(np.where(mask)[0])

        result = table.update_from_conformers(
            filtered, metadata_agg={"energy": "min"}, drop_empty=True
        )

        # All surviving molecules should have min energy < 8
        for val in result.metadata["energy"]:
            assert val < 8.0

    def test_history_audit_trail(self):
        """The full audit trail should be preserved across the pipeline."""
        table = _simple_3d_table(n_mols=2, n_confs_each=3)
        conf_table = table.unroll_conformers()
        collapsed = conf_table.collapse()

        block_names = [e.block_name for e in collapsed.history]
        assert "MoleculeTable.unroll_conformers" in block_names
        assert "ConformerTable.collapse" in block_names


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])