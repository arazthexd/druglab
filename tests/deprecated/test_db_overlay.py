"""
tests/test_db_overlay.py
~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive tests for OverlayBackend and its integration with BaseTable.

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

    def _deserialize_object(self, raw):
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

    def test_subset_returns_overlay_backend(self):
        table = _make_table()
        sub = table.subset([0, 2, 4])
        assert isinstance(sub._backend, OverlayBackend), (
            "subset() should wrap backend in OverlayBackend"
        )

    def test_subset_data_integrity(self):
        table = _make_table()
        sub = table.subset([1, 3, 5])
        assert len(sub) == 3
        assert sub.get_metadata()["val"].tolist() == [1, 3, 5]
        assert sub.get_objects() == [{"id": 1}, {"id": 3}, {"id": 5}]

    def test_subset_feature_data_integrity(self):
        table = _make_table()
        sub = table.subset([0, 2])
        fp = sub.get_feature("fp")
        expected = table.get_feature("fp")[[0, 2]]
        np.testing.assert_array_equal(fp, expected)

    def test_no_base_data_duplication(self):
        """The OverlayBackend must share the base backend, not copy it."""
        base = _make_backend()
        overlay = OverlayBackend(base, np.array([0, 1, 2], dtype=np.intp))
        # _base should be the same object (identity check)
        assert overlay._base is base

    def test_base_feature_array_not_copied(self):
        """Local features dict should be empty for a fresh overlay (no CoW yet)."""
        base = _make_backend()
        overlay = OverlayBackend(base, np.array([0, 1, 2], dtype=np.intp))
        assert len(overlay._local_features) == 0

    def test_subset_with_bool_mask(self):
        table = _make_table(6)
        mask = np.array([True, False, True, False, True, False])
        sub = table.subset(mask)
        assert len(sub) == 3
        assert sub.get_metadata()["val"].tolist() == [0, 2, 4]

    def test_subset_with_slice(self):
        table = _make_table(8)
        sub = table.subset(slice(2, 6))
        assert len(sub) == 4


# ===========================================================================
# 2. Flattening Nested Overlays
# ===========================================================================

class TestNestedOverlayFlattening:

    def test_nested_overlay_points_to_absolute_base(self):
        base = _make_backend(10)
        outer = OverlayBackend(base, np.array([0, 1, 2, 3, 4], dtype=np.intp))
        inner = OverlayBackend(outer, np.array([1, 3], dtype=np.intp))
        # Flattening: inner._base should be the original EagerMemoryBackend
        assert isinstance(inner._base, EagerMemoryBackend)
        assert inner._base is base

    def test_nested_overlay_not_an_overlay_of_overlay(self):
        base = _make_backend(10)
        outer = OverlayBackend(base, np.array([0, 2, 4, 6, 8], dtype=np.intp))
        inner = OverlayBackend(outer, np.array([0, 2], dtype=np.intp))
        assert not isinstance(inner._base, OverlayBackend)

    def test_nested_index_map_composition(self):
        """
        outer maps [0,2,4,6,8] from base (even rows).
        inner selects [1, 3] from outer → should map to [2, 6] in base.
        """
        base = _make_backend(10)
        outer = OverlayBackend(base, np.array([0, 2, 4, 6, 8], dtype=np.intp))
        inner = OverlayBackend(outer, np.array([1, 3], dtype=np.intp))
        np.testing.assert_array_equal(inner._index_map, np.array([2, 6]))

    def test_nested_overlay_data_correct(self):
        base = _make_backend(10)
        outer = OverlayBackend(base, np.array([0, 2, 4, 6, 8], dtype=np.intp))
        inner = OverlayBackend(outer, np.array([0, 2], dtype=np.intp))
        # inner sees rows 0 and 4 from base
        meta = inner.get_metadata()
        assert meta["val"].tolist() == [0, 4]

    def test_triple_nesting_flattens_correctly(self):
        base = _make_backend(20)
        o1 = OverlayBackend(base, np.arange(10, dtype=np.intp))        # rows 0-9
        o2 = OverlayBackend(o1, np.arange(5, dtype=np.intp))           # rows 0-4
        o3 = OverlayBackend(o2, np.array([1, 3], dtype=np.intp))       # rows 1, 3
        assert isinstance(o3._base, EagerMemoryBackend)
        assert o3._base is base
        np.testing.assert_array_equal(o3._index_map, np.array([1, 3]))

    def test_table_subset_of_subset_is_flat(self):
        table = _make_table(10)
        sub1 = table.subset([0, 2, 4, 6, 8])
        sub2 = sub1.subset([1, 3])
        assert isinstance(sub2._backend, OverlayBackend)
        assert isinstance(sub2._backend._base, EagerMemoryBackend)


# ===========================================================================
# 3. CoW Feature Addition
# ===========================================================================

class TestCoWFeatureAddition:

    def test_new_feature_readable_from_overlay(self):
        base = _make_backend(6)
        overlay = OverlayBackend(base, np.array([0, 1, 2], dtype=np.intp))
        new_feat = np.array([[10.0], [20.0], [30.0]])
        overlay.update_feature("new_feat", new_feat)
        result = overlay.get_feature("new_feat")
        np.testing.assert_array_equal(result, new_feat)

    def test_new_feature_not_in_base(self):
        base = _make_backend(6)
        overlay = OverlayBackend(base, np.array([0, 1, 2], dtype=np.intp))
        overlay.update_feature("new_feat", np.ones((3, 1)))
        assert "new_feat" not in base.get_feature_names()

    def test_new_feature_raises_on_base(self):
        base = _make_backend(6)
        overlay = OverlayBackend(base, np.array([0, 1, 2], dtype=np.intp))
        overlay.update_feature("new_feat", np.ones((3, 1)))
        with pytest.raises(KeyError):
            base.get_feature("new_feat")

    def test_new_feature_stored_in_local_delta(self):
        base = _make_backend(6)
        overlay = OverlayBackend(base, np.array([0, 1, 2], dtype=np.intp))
        overlay.update_feature("my_feat", np.zeros((3,)))
        assert "my_feat" in overlay._local_features

    def test_feature_visible_in_names(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.arange(4, dtype=np.intp))
        overlay.update_feature("extra", np.zeros(4))
        assert "extra" in overlay.get_feature_names()
        assert "extra" not in base.get_feature_names()

    def test_table_level_cow_feature(self):
        table = _make_table(6)
        sub = table.subset([0, 1, 2])
        sub.update_feature("new_fp", np.ones((3, 2)))
        assert "new_fp" in sub.get_feature_names()
        assert "new_fp" not in table.get_feature_names()


# ===========================================================================
# 4. CoW Mutation
# ===========================================================================

class TestCoWMutation:

    def test_metadata_update_visible_in_overlay(self):
        base = _make_backend(6)
        overlay = OverlayBackend(base, np.array([1, 2, 3], dtype=np.intp))
        overlay.update_metadata({"val": [99, 98, 97]})
        result = overlay.get_metadata()["val"].tolist()
        assert result == [99, 98, 97]

    def test_metadata_update_does_not_touch_base(self):
        base = _make_backend(6)
        original_vals = base.get_metadata()["val"].tolist()
        overlay = OverlayBackend(base, np.array([1, 2, 3], dtype=np.intp))
        overlay.update_metadata({"val": [99, 98, 97]})
        assert base.get_metadata()["val"].tolist() == original_vals

    def test_partial_row_update_in_overlay(self):
        base = _make_backend(6)
        overlay = OverlayBackend(base, np.arange(6, dtype=np.intp))
        overlay.update_metadata({"val": [42]}, idx=np.array([2]))
        assert overlay.get_metadata()["val"].iloc[2] == 42
        assert base.get_metadata()["val"].iloc[2] == 2  # base untouched

    def test_object_update_visible_in_overlay(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.array([0, 1], dtype=np.intp))
        overlay.update_objects({"id": 999}, idx=0)
        assert overlay.get_objects(idx=0) == {"id": 999}

    def test_object_update_does_not_touch_base(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.array([0, 1], dtype=np.intp))
        overlay.update_objects({"id": 999}, idx=0)
        assert base.get_objects(idx=0) == {"id": 0}

    def test_feature_update_isolated_to_overlay(self):
        base = _make_backend(6)
        orig_fp = base.get_feature("fp").copy()
        overlay = OverlayBackend(base, np.array([0, 1, 2], dtype=np.intp))
        new_vals = np.full((3, 4), -1.0, dtype=np.float32)
        overlay.update_feature("fp", new_vals)
        # Base unchanged
        np.testing.assert_array_equal(base.get_feature("fp"), orig_fp)
        # Overlay reflects change
        np.testing.assert_array_equal(overlay.get_feature("fp"), new_vals)

    def test_table_subset_metadata_mutation_isolated(self):
        table = _make_table(8)
        sub = table.subset([2, 3, 4])
        sub.update_metadata({"val": [100, 200, 300]})
        # Original table untouched
        assert table.get_metadata()["val"].tolist() == list(range(8))
        # Subset reflects change
        assert sub.get_metadata()["val"].tolist() == [100, 200, 300]


# ===========================================================================
# 5. Commit
# ===========================================================================

class TestCommit:

    def test_commit_flushes_feature_to_base(self):
        base = _make_backend(6)
        overlay = OverlayBackend(base, np.arange(6, dtype=np.intp))
        overlay.update_feature("fp", np.full((6, 4), 7.0, dtype=np.float32))
        overlay.commit()
        np.testing.assert_array_equal(
            base.get_feature("fp"),
            np.full((6, 4), 7.0, dtype=np.float32)
        )

    def test_commit_clears_local_features(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.arange(4, dtype=np.intp))
        overlay.update_feature("fp", np.zeros((4, 4), dtype=np.float32))
        overlay.commit()
        assert len(overlay._local_features) == 0

    def test_commit_flushes_metadata_to_base(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.arange(4, dtype=np.intp))
        overlay.update_metadata({"val": [10, 20, 30, 40]})
        overlay.commit()
        assert base.get_metadata()["val"].tolist() == [10, 20, 30, 40]

    def test_commit_clears_local_metadata(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.arange(4, dtype=np.intp))
        overlay.update_metadata({"val": [10, 20, 30, 40]})
        overlay.commit()
        assert overlay._local_metadata is None

    def test_commit_flushes_objects_to_base(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.arange(4, dtype=np.intp))
        overlay.update_objects({"id": 999}, idx=0)
        overlay.commit()
        assert base.get_objects(idx=0) == {"id": 999}

    def test_commit_clears_local_objects(self):
        base = _make_backend(4)
        overlay = OverlayBackend(base, np.arange(4, dtype=np.intp))
        overlay.update_objects({"id": 999}, idx=1)
        overlay.commit()
        assert len(overlay._local_objects) == 0

    def test_commit_partial_index_flushes_correctly(self):
        """Overlay covers rows [2, 3, 4] of a 6-row base."""
        base = _make_backend(6)
        overlay = OverlayBackend(base, np.array([2, 3, 4], dtype=np.intp))
        overlay.update_metadata({"val": [200, 300, 400]})
        overlay.commit()
        # Only base rows 2, 3, 4 should be modified
        vals = base.get_metadata()["val"].tolist()
        assert vals[:2] == [0, 1]
        assert vals[2:5] == [200, 300, 400]
        assert vals[5] == 5

    def test_table_commit(self):
        table = _make_table(4)
        sub = table.subset([0, 1, 2, 3])
        sub.update_metadata({"val": [9, 8, 7, 6]})
        sub.commit()
        # The base table's backend should now reflect the changes
        assert table.get_metadata()["val"].tolist() == [9, 8, 7, 6]


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
            sub.save(bundle_path)
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
        assert len(overlay._deleted_features) == 0


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