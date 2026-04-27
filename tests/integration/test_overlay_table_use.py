"""
tests/integration/test_overlay_table_use.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive tests Integration of OverlayBackend with BaseTable.

Covers:
1.  Zero-Copy Subsetting
2.  Flattening Nested Overlays
3.  CoW Feature Addition
4.  CoW Mutation (metadata)
5.  Commit
6.  Materialize & Save
7.  CoW Deletion (Tombstones)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from druglab.db.backend import EagerMemoryBackend
from druglab.db.backend.overlay import OverlayBackend
from druglab.db.table.base import BaseTable, HistoryEntry
from tests.shared.make_dummy_db import BackendContext, TableContext, _make_metadata, _make_dict_objects


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backend(n: int = 8) -> EagerMemoryBackend:
    return EagerMemoryBackend(
        objects=[{"id": i} for i in range(n)],
        metadata=pd.DataFrame({"val": list(range(n)), "label": [f"L{i}" for i in range(n)]}),
        features={
            "fp": np.arange(n * 4, dtype=np.float32).reshape(n, 4),
            "desc": np.ones((n, 2), dtype=np.float64),
        },
    )


class DummyTable(BaseTable[dict]):
    def _serialize_object(self, obj):
        return json.dumps(obj).encode()

    @classmethod
    def _deserialize_object(cls, raw):
        return json.loads(raw.decode())

    def _object_type_name(self):
        return "dict"


def _make_table(n: int = 8) -> DummyTable:
    return DummyTable(
        objects=[{"id": i} for i in range(n)],
        metadata=pd.DataFrame({"val": list(range(n)), "label": [f"L{i}" for i in range(n)]}),
        features={
            "fp": np.arange(n * 4, dtype=np.float32).reshape(n, 4),
            "desc": np.ones((n, 2), dtype=np.float64),
        },
    )


# ===========================================================================
# 1. Zero-Copy Subsetting
# ===========================================================================

class TestZeroCopySubsetting:

    def test_subset_returns_overlay_backend(self, tctx: TableContext):
        sub = tctx.table.subset([0, 2, 4])
        
        # Checking that the internal backend was wrapped
        from druglab.db.backend import OverlayBackend
        assert isinstance(sub._backend, OverlayBackend), (
            "subset() should wrap backend in OverlayBackend"
        )

    def test_subset_data_integrity(self, tctx: TableContext):
        indices = [1, 3, 5]
        sub = tctx.table.subset(indices)
        
        assert len(sub) == len(indices)
        
        # Verify Metadata (using first column name from context)
        first_col = tctx.meta_cols[0]
        generated_meta = _make_metadata(tctx.num_rows, *tctx.meta_cols).iloc[indices]
        assert sub.get_metadata()[first_col].tolist() == generated_meta[first_col].to_list()
        
        # Verify Objects (comparing against generated expectations)
        expected_objs = _make_dict_objects(tctx.num_rows)
        expected_objs = [expected_objs[i] for i in indices]
        assert sub.get_objects() == expected_objs

    def test_subset_feature_data_integrity(self, tctx: TableContext):
        table = tctx.table
        indices = [0, 2]
        sub = table.subset(indices)
        
        feat_name = tctx.feat_names[0]
        fp = sub.get_feature(feat_name)
        
        # Compare subset feature vs parent table feature (sliced)
        expected = table.get_feature(feat_name)[indices]
        np.testing.assert_array_equal(fp, expected)

    def test_no_base_data_duplication(self, tctx: TableContext):
        """The OverlayBackend must share the base backend, not copy it."""
        from druglab.db.backend import OverlayBackend
        
        base = tctx.table.backend
        indices = np.array([0, 1, 2], dtype=np.intp)
        overlay = OverlayBackend(base, indices)
        
        # Identity check: ensure they are the exact same object in memory
        assert overlay._base is base

    def test_base_feature_array_not_copied(self, tctx: TableContext):
        """Local features dict should be empty for a fresh overlay (no CoW yet)."""
        from druglab.db.backend import OverlayBackend
        
        overlay = OverlayBackend(tctx.table.backend, np.array([0, 1, 2], dtype=np.intp))
        
        # Verify the internal CoW dictionary is empty initially
        assert len(overlay._feature_store.delta.local) == 0

    def test_subset_with_bool_mask(self, tctx: TableContext):
        # Create a mask that matches the context's row count [True, False, True...]
        mask = np.array([i % 2 == 0 for i in range(tctx.num_rows)])
        sub = tctx.table.subset(mask)
        
        expected_indices = [i for i, val in enumerate(mask) if val]
        assert len(sub) == len(expected_indices)
        
        first_col = tctx.meta_cols[0]
        generated_meta = _make_metadata(tctx.num_rows, *tctx.meta_cols).iloc[expected_indices]
        assert sub.get_metadata()[first_col].tolist() == generated_meta[first_col].to_list()

    def test_subset_with_slice(self, tctx: TableContext):
        if tctx.num_rows < 6:
            pytest.skip("Test requires at least 6 rows")
            
        sub = tctx.table.subset(slice(2, 6))
        assert len(sub) == 4
        
        all_objs = _make_dict_objects(tctx.num_rows)
        expected_objs = all_objs[2:6]
        assert sub.get_objects() == expected_objs

# ===========================================================================
# 2. Flattening Nested Overlays
# ===========================================================================

class TestNestedOverlayFlattening:

    def test_table_subset_of_subset_is_flat(self, tctx: TableContext):
        # We start with the table from tctx
        sub1 = tctx.table.subset([0, 2, 4, 6, 8])
        sub2 = sub1.subset([1, 3])
        
        # Verify the backend of the second subset is already flattened
        assert isinstance(sub2._backend, OverlayBackend)
        assert isinstance(sub2._backend._base, EagerMemoryBackend)
        
        # Check identity against the original base backend via tctx.table
        assert sub2._backend._base is tctx.table._backend

# ===========================================================================
# 3. CoW Feature Addition
# ===========================================================================

class TestCoWFeatureAddition:

    def test_table_level_cow_feature(self, tctx: TableContext):
        # Using tctx.table directly
        sub = tctx.table.subset([0, 1, 2])
        
        new_name = "new_fp"
        sub.update_feature(new_name, np.ones((3, 2), dtype=np.float32))
        
        # Verification at the Table API level
        assert new_name in sub.get_feature_names()
        assert new_name not in tctx.table.get_feature_names()


# ===========================================================================
# 4. CoW Mutation
# ===========================================================================

class TestCoWMutation:

    def test_table_subset_metadata_mutation_isolated(self, tctx: TableContext):
        table = tctx.table
        first_col = tctx.meta_cols[0]
        
        # Selection within bounds
        indices = [2, 3, 4]
        sub = table.subset(indices)
        
        update_vals = [100, 200, 300]
        sub.update_metadata(pd.Series(update_vals, name=first_col))
        
        # Original table ground-truth check
        generated_meta = _make_metadata(tctx.num_rows, *tctx.meta_cols)
        assert table.get_metadata()[first_col].tolist() == generated_meta[first_col].tolist()
        
        # Subset reflects mutated state
        assert sub.get_metadata()[first_col].tolist() == update_vals


# ===========================================================================
# 5. Commit
# ===========================================================================

class TestCommit:

    def test_table_commit(self, tctx: TableContext):
        table = tctx.table
        first_col = tctx.meta_cols[0]
        
        # Create an identity subset
        indices = list(range(tctx.num_rows))
        sub = table.subset(indices)
        
        new_vals = [i + 500 for i in indices]
        sub.update_metadata(pd.Series(new_vals, name=first_col))
        
        # Commit through the Table API
        sub.commit()
        
        # The parent table (and its base backend) should now reflect the change
        assert table.get_metadata()[first_col].tolist() == new_vals


######################## BELOW IS NOT FINAL #################################

# ===========================================================================
# 6. Materialize & Save
# ===========================================================================

class TestMaterializeAndSave:

    def test_materialize_returns_eager_memory_backend(self):
        base = _make_backend(6)
        overlay = OverlayBackend(base, np.array([0, 2, 4], dtype=np.intp))
        result = overlay.materialize()
        assert isinstance(result, EagerMemoryBackend)

    def test_materialize_includes_local_feature(self):
        base = _make_backend(6)
        overlay = OverlayBackend(base, np.array([0, 1, 2], dtype=np.intp))
        overlay.update_feature("new_feat", np.array([[1.0], [2.0], [3.0]]))
        mat = overlay.materialize()
        assert "new_feat" in mat.get_feature_names()
        np.testing.assert_array_equal(mat.get_feature("new_feat"), [[1.0], [2.0], [3.0]])

    def test_materialize_includes_local_metadata(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.arange(4, dtype=np.intp))
        overlay.update_metadata({"val": [10, 20, 30, 40]})
        mat = overlay.materialize()
        assert mat.get_metadata()["val"].tolist() == [10, 20, 30, 40]

    def test_materialize_excludes_tombstoned_features(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.arange(4, dtype=np.intp))
        overlay.drop_feature("fp")
        mat = overlay.materialize()
        assert "fp" not in mat.get_feature_names()

    def test_materialize_excludes_tombstoned_metadata_cols(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.arange(4, dtype=np.intp))
        overlay.drop_metadata_columns("label")
        mat = overlay.materialize()
        assert "label" not in mat.get_metadata().columns

    def test_table_materialize_returns_table_with_eager_backend(self):
        table = _make_table(6)
        sub = table.subset([0, 2, 4])
        sub.update_feature("new_fp", np.ones((3, 2)))
        mat = sub.materialize()
        assert isinstance(mat._backend, EagerMemoryBackend)
        assert "new_fp" in mat.get_feature_names()

    def test_save_and_load_overlay(self):
        table = _make_table(6)
        sub = table.subset([1, 3, 5])
        sub.update_metadata({"val": [99, 88, 77]})
        sub.update_feature("fp", np.full((3, 4), 5.0, dtype=np.float32))

        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = Path(tmp) / "test_table.dlb"
            sub.save(bundle_path)

            loaded = DummyTable.load(bundle_path)

        assert len(loaded) == 3
        assert loaded.get_metadata()["val"].tolist() == [99, 88, 77]
        np.testing.assert_array_equal(
            loaded.get_feature("fp"),
            np.full((3, 4), 5.0, dtype=np.float32)
        )

    def test_save_and_load_preserves_objects(self):
        table = _make_table(4)
        sub = table.subset([0, 2])
        sub.update_objects({"id": 999}, idx=0)

        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = Path(tmp) / "test_obj.dlb"
            bundle_path = sub.save(bundle_path)
            loaded = DummyTable.load(bundle_path)

        assert loaded.get_objects(idx=0) == {"id": 999}
        assert loaded.get_objects(idx=1) == {"id": 2}

    def test_save_backend_class_is_eager(self):
        """Saved bundle always has backend_class=EagerMemoryBackend."""
        table = _make_table(4)
        sub = table.subset([0, 1])

        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = Path(tmp) / "test.dlb"
            sub.save(bundle_path)
            config = json.loads((bundle_path / "config.json").read_text())

        assert config["backend_class"] == "EagerMemoryBackend"


# ===========================================================================
# 7. CoW Deletion (Tombstones)
# ===========================================================================

class TestTombstones:

    def test_drop_feature_raises_on_overlay(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.arange(4, dtype=np.intp))
        overlay.drop_feature("fp")
        with pytest.raises(KeyError):
            overlay.get_feature("fp")

    def test_drop_feature_base_still_has_data(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.arange(4, dtype=np.intp))
        orig_fp = base.get_feature("fp").copy()
        overlay.drop_feature("fp")
        # Base backend still has the feature
        assert "fp" in base.get_feature_names()
        np.testing.assert_array_equal(base.get_feature("fp"), orig_fp)

    def test_drop_feature_not_in_overlay_names(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.arange(4, dtype=np.intp))
        overlay.drop_feature("fp")
        assert "fp" not in overlay.get_feature_names()

    def test_commit_propagates_feature_deletion_to_base(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.arange(4, dtype=np.intp))
        overlay.drop_feature("fp")
        overlay.commit()
        assert "fp" not in base.get_feature_names()

    def test_drop_metadata_col_raises_on_overlay(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.arange(4, dtype=np.intp))
        overlay.drop_metadata_columns("label")
        meta = overlay.get_metadata()
        assert "label" not in meta.columns

    def test_drop_metadata_col_base_still_has_col(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.arange(4, dtype=np.intp))
        overlay.drop_metadata_columns("label")
        assert "label" in base.get_metadata().columns

    def test_commit_propagates_metadata_col_deletion_to_base(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.arange(4, dtype=np.intp))
        overlay.drop_metadata_columns("label")
        overlay.commit()
        assert "label" not in base.get_metadata().columns

    def test_readd_dropped_feature(self):
        """Re-adding a tombstoned feature should work cleanly."""
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.arange(4, dtype=np.intp))
        overlay.drop_feature("fp")
        new_fp = np.full((4, 4), 99.0, dtype=np.float32)
        overlay.update_feature("fp", new_fp)
        np.testing.assert_array_equal(overlay.get_feature("fp"), new_fp)

    def test_table_level_drop_feature_tombstone(self):
        table = _make_table(4)
        sub = table.subset([0, 1, 2, 3])
        sub.drop_feature("fp")
        with pytest.raises(KeyError):
            sub.get_feature("fp")
        # Original table unaffected
        assert "fp" in table.get_feature_names()

    def test_materialize_after_drop_does_not_include_dropped(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.arange(4, dtype=np.intp))
        overlay.drop_feature("desc")
        mat = overlay.materialize()
        assert "desc" not in mat.get_feature_names()
        assert "fp" in mat.get_feature_names()

    def test_tombstone_cleared_after_commit(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.arange(4, dtype=np.intp))
        overlay.drop_feature("fp")
        overlay.commit()
        assert len(overlay._feature_store.delta.deleted) == 0


# ===========================================================================
# 8. Edge cases & additional coverage
# ===========================================================================

class TestEdgeCases:

    def test_overlay_full_view(self):
        """Default (no index_map) covers all rows."""
        base = _make_backend(4)
        overlay = OverlayBackend(base)
        assert len(overlay) == 4
        np.testing.assert_array_equal(overlay._index_map, np.arange(4))

    def test_overlay_length(self):
        base = _make_backend(8)
        overlay = OverlayBackend(base, np.array([0, 4, 7], dtype=np.intp))
        assert len(overlay) == 3

    def test_add_new_metadata_column_to_overlay(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.arange(4, dtype=np.intp))
        overlay.add_metadata_column("new_col", np.array([1.0, 2.0, 3.0, 4.0]))
        assert "new_col" in overlay.get_metadata().columns
        assert "new_col" not in base.get_metadata().columns

    def test_get_feature_names_reflects_local_and_base(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.arange(4, dtype=np.intp))
        overlay.update_feature("extra", np.ones(4))
        names = set(overlay.get_feature_names())
        assert "fp" in names
        assert "desc" in names
        assert "extra" in names

    def test_commit_raises_on_non_overlay_backend(self):
        table = _make_table(4)
        with pytest.raises(TypeError):
            table.commit()

    def test_materialize_on_concrete_table(self):
        table = _make_table(4)
        mat = table.materialize()
        assert len(mat) == 4

    def test_get_feature_shape_overlay(self):
        base = _make_backend(8)
        overlay = OverlayBackend(base, np.array([0, 1, 2], dtype=np.intp))
        shape = overlay.get_feature_shape("fp")
        assert shape == (3, 4)

    def test_get_feature_shape_local(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.arange(4, dtype=np.intp))
        overlay.update_feature("new", np.ones((4, 7)))
        assert overlay.get_feature_shape("new") == (4, 7)

    def test_drop_nonexistent_feature_raises(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.arange(4, dtype=np.intp))
        with pytest.raises(KeyError):
            overlay.drop_feature("does_not_exist")


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])