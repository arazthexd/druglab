from __future__ import annotations
from typing import NamedTuple, TypeVar, Generic
import json

import numpy as np
import pandas as pd

from druglab.db.backend import BaseStorageBackend, EagerMemoryBackend
from druglab.db.table import BaseTable

# ===========================================================================
# Context Classes
# ===========================================================================

class TableContext(NamedTuple):
    table: BaseTable
    num_rows: int
    meta_cols: tuple[str]
    feat_sizes: tuple[int]
    feat_names: tuple[str]
    obj_type: type

BCT = TypeVar("BCT")

class BackendContext(NamedTuple, Generic[BCT]):
    backend: BaseStorageBackend | BCT
    num_rows: int
    meta_cols: tuple[str]
    feat_sizes: dict[str, int]
    obj_type: type

    @property
    def feat_names(self) -> tuple[str]:
        return tuple(self.feat_sizes.keys())

    def get_table(self) -> BaseTable:
        return DictTable(_backend=self.backend)
    
    def get_table_context(self) -> TableContext:
        return TableContext(
            table=self.get_table(),
            num_rows=self.num_rows,
            meta_cols=self.meta_cols,
            feat_sizes=self.feat_sizes,
            feat_names=self.feat_names,
            obj_type=self.obj_type,
        )
    
# ===========================================================================
# Dummy Table Classes
# ===========================================================================

class DictTable(BaseTable[dict]):
    def _serialize_object(self, obj):
        return json.dumps(obj).encode()

    def _deserialize_object(self, raw):
        return json.loads(raw.decode())

    def _object_type_name(self):
        return "dict"
    
# ===========================================================================
# Dummy Factory Methods
# ===========================================================================
    
def _make_metadata(n: int, *cols: str) -> pd.DataFrame:
    return pd.DataFrame({col: range(n*i, n*(i+1)) for i, col in enumerate(cols)})

def _make_features(n: int, **nfs: dict[str, int]) -> dict:
    return {
        name: i * np.arange(n * nf, dtype=np.float32).reshape(n, nf)
        for i, (name, nf) in enumerate(nfs.items())
    }

def _make_dict_objects(n: int) -> list:
    return [{"id": i} for i in range(n)]

# ===========================================================================
# Dummy Backend/Table Constructors
# ===========================================================================

def _make_dummy_dict_memory_backend(
    n_rows: int = 10, 
    meta_cols: tuple[str] = ("col1", "col2"),
    feat_sizes: dict[str, int] = {"feat1": 4, "feat2": 8}
) -> EagerMemoryBackend:

    return EagerMemoryBackend(
        objects=_make_dict_objects(n_rows),
        metadata=_make_metadata(n_rows, *meta_cols),
        features=_make_features(n_rows, **feat_sizes),
    )

def _make_dummy_dict_memory_backend_context(
    n_rows: int = 10, 
    meta_cols: tuple[str] = ("col1", "col2"),
    feat_sizes: dict[str, int] = {"feat1": 4, "feat2": 8}
) -> BackendContext[EagerMemoryBackend]:
    
    return BackendContext(
        backend=_make_dummy_dict_memory_backend(
            n_rows=n_rows,
            meta_cols=meta_cols,
            feat_sizes=feat_sizes
        ),
        num_rows=n_rows,
        meta_cols=meta_cols,
        feat_sizes=feat_sizes,
        obj_type=dict,
    )