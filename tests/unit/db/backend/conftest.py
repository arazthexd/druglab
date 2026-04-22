from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from druglab.db.backend import (
    BaseStorageBackend,
    EagerMemoryBackend,
    MemoryMetadataMixin,
    MemoryObjectMixin,
    MemoryFeatureMixin,
)
from druglab.db.table import BaseTable, HistoryEntry, META, OBJ, FEAT, M, O, F
from tests.shared.make_dummy_db import (
    _make_dummy_dict_memory_backend_context, 
    BackendContext, TableContext,
    _make_metadata, _make_dict_objects, _make_features
)

# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(name="bctx")
def backend_context() -> BackendContext:
    return _make_dummy_dict_memory_backend_context(
        n_rows = 40,
        meta_cols = ("col1", "col2"),
        feat_sizes = {"feat1": 4, "feat2": 8}
    )

@pytest.fixture(name="tctx")
def table_context(bctx: BackendContext) -> TableContext:
    return bctx.get_table_context()
    