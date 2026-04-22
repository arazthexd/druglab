import numpy as np
import pandas as pd

import pytest

from druglab.db.backend import EagerMemoryBackend, BaseStorageBackend
from druglab.db.table import BaseTable
from tests.shared.make_dummy_db import (
    BackendContext, TableContext, 
    _make_dummy_dict_memory_backend_context
)

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

