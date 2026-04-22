from typing import TYPE_CHECKING

import numpy as np

import pytest
from conftest import BackendContext, TableContext

from druglab.db.table import FEAT, META, OBJ
from druglab.db.indexing import RowSelection

# ===========================================================================
# Section 1: Integration — EagerMemoryBackend uses normalize_row_index
# ===========================================================================

class TestBackendIntegration:
    """
    Smoke tests confirming the backend delegates index resolution through
    the shared indexing module.
    """

    def test_get_feature_float_array_rejected(self, bctx: BackendContext):
        with pytest.raises(TypeError, match="float"):
            bctx.backend.get_feature(bctx.feat_names[0], idx=np.array([1.0, 2.0]))

    def test_get_metadata_float_array_rejected(self, bctx: BackendContext):
        with pytest.raises(TypeError, match="float"):
            bctx.backend.get_metadata(idx=np.array([0.0, 1.0]))

    def test_get_objects_float_array_rejected(self, bctx: BackendContext):
        with pytest.raises(TypeError):
            bctx.backend.get_objects(idx=np.array([0.0, 1.0]))

    def test_get_feature_object_array_rejected(self, bctx: BackendContext):
        with pytest.raises(TypeError, match="[Oo]bject"):
            bctx.backend.get_feature(bctx.feat_names[0], idx=np.array([0, 1], dtype=object))

    def test_get_feature_out_of_bounds_raises(self, bctx: BackendContext):
        # Dynamically create an index that is guaranteed to be out of bounds
        invalid_idx = np.array([bctx.num_rows + 1])
        with pytest.raises(IndexError):
            bctx.backend.get_feature(bctx.feat_names[0], idx=invalid_idx)

    def test_get_metadata_bool_mask_wrong_length_raises(self, bctx: BackendContext):
        # Intentionally creating a mask that doesn't match ctx.num_rows
        wrong_length_mask = np.array([True, False]) 
        with pytest.raises(IndexError, match="length"):
            bctx.backend.get_metadata(idx=wrong_length_mask)

# ===========================================================================
# Section 2: Integration — BaseTable uses normalize_row_index via backend
# ===========================================================================

class TestTableIntegration:
    """
    Confirm that table-level multi-axis indexing propagates strict errors
    from the new indexing module.
    """

    def test_feat_pushdown_float_array_rejected(self, tctx: TableContext):
        with pytest.raises(TypeError, match="float"):
            _ = tctx.table[FEAT, tctx.feat_names[0], np.array([0.0, 1.0])]

    def test_feat_pushdown_object_array_rejected(self, tctx: TableContext):
        with pytest.raises(TypeError, match="[Oo]bject"):
            _ = tctx.table[FEAT, tctx.feat_names[0], np.array([0, 1], dtype=object)]

    def test_meta_pushdown_out_of_bounds_raises(self, tctx: TableContext):
        # Always past the end of the current fixture size
        invalid_idx = tctx.num_rows + 99
        with pytest.raises(IndexError):
            _ = tctx.table[META, invalid_idx]

    def test_bool_mask_with_wrong_length_raises_at_table_level(self, tctx: TableContext):
        """Table.subset enforces boolean mask length independently."""
        # Create a mask that is intentionally not equal to ctx.num_rows
        wrong_length_mask = np.array([True, False])
        
        with pytest.raises(ValueError):
            tctx.table.subset(wrong_length_mask)