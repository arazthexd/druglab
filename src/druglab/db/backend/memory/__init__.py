"""Composed in-memory backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import uuid as _uuid_mod

import numpy as np
import pandas as pd

from ..base import BaseStorageBackend
from .feature import MemoryFeatureStore, MemoryFeatureMixin
from .metadata import MemoryMetadataStore, MemoryMetadataMixin
from .objects import MemoryObjectStore, MemoryObjectMixin

__all__ = [
    "MemoryMetadataStore",
    "MemoryObjectStore",
    "MemoryFeatureStore",
    "MemoryMetadataMixin",
    "MemoryObjectMixin",
    "MemoryFeatureMixin",
    "EagerMemoryBackend",
]


class EagerMemoryBackend(BaseStorageBackend):
    BACKEND_NAME = "EagerMemoryBackend"

    def __init__(
        self,
        objects: Optional[List[Any]] = None,
        metadata: Optional[pd.DataFrame] = None,
        features: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self._object_store = MemoryObjectStore(objects)
        self._metadata_store = MemoryMetadataStore(metadata, n_rows_hint=self._object_store.n_rows())
        self._feature_store = MemoryFeatureStore(features, n_rows_hint=self._object_store.n_rows())
        self.validate()

    def __len__(self) -> int:
        """
        Return the global, official length of the dataset.

        Returns
        -------
        int
            The authoritative row count (delegated to object count).
        """
        return self._n_objects()

    def get_objects(self, idx=None):
        return self._object_store.get_objects(idx)

    def update_objects(self, objs, idx=None, **kwargs) -> None:
        self._object_store.update_objects(objs, idx)

    def _n_objects(self) -> int:
        return self._object_store.n_rows()

    def _validate_objects(self) -> None:
        return

    def get_metadata(self, idx=None, cols=None):
        return self._metadata_store.get_metadata(idx, cols)

    def add_metadata_column(self, name, value, idx=None, na=None, **kwargs) -> None:
        self._metadata_store.add_metadata_column(name, value, idx=idx, na=na)

    def update_metadata(self, values, idx=None, **kwargs) -> None:
        self._metadata_store.update_metadata(values, idx=idx)

    def drop_metadata_columns(self, cols=None) -> None:
        self._metadata_store.drop_metadata_columns(cols)

    def get_metadata_columns(self):
        return self._metadata_store.get_metadata_columns()

    def _n_metadata_rows(self) -> int:
        return self._metadata_store.n_rows()

    def _validate_metadata(self) -> None:
        return

    def get_feature(self, name: str, idx=None) -> np.ndarray:
        return self._feature_store.get_feature(name, idx)

    def update_feature(self, name: str, array: np.ndarray, idx=None, na=None, **kwargs) -> None:
        self._feature_store.update_feature(name, array, idx=idx, na=na)

    def drop_feature(self, name: str) -> None:
        self._feature_store.drop_feature(name)

    def get_feature_names(self):
        return self._feature_store.get_feature_names()

    def get_feature_shape(self, name: str) -> tuple:
        return self._feature_store.get_feature_shape(name)

    def _n_feature_rows(self) -> int:
        return self._feature_store.n_rows()

    def _validate_features(self) -> None:
        if not self.get_feature_names():
            return
        n = self._n_feature_rows()
        for name in self.get_feature_names():
            if n != self.get_feature_shape(name)[0]:
                raise ValueError(f"Feature '{name}' has {self.get_feature_shape(name)[0]} rows, expected {n}")
            
    def validate(self) -> None:
        expected_len = len(self)
        meta_len = self._n_metadata_rows()
        feat_len = self._n_feature_rows()
        obj_len = self._n_objects()
        if not (expected_len == meta_len == feat_len == obj_len):
            raise ValueError(
                f"Backend Dimension Mismatch!\n"
                f"Global Length: {expected_len}\n"
                f"Metadata Rows: {meta_len}\n"
                f"Feature Rows:  {feat_len}\n"
                f"Object Count:  {obj_len}"
            )

    def _gather_materialized_state(self, target_path: Optional[Path] = None, index_map: Optional[np.ndarray] = None):
        result = {}
        result.update(self._object_store.gather_materialized_state(index_map=index_map))
        result.update(self._metadata_store.gather_materialized_state(index_map=index_map))
        result.update(self._feature_store.gather_materialized_state(index_map=index_map))
        return result

    def save_storage_context(self, path: Path, object_writer: Optional[Callable[[List[Any], Path], None]] = None, **kwargs: Any) -> None:
        self._metadata_store.save(path)
        self._object_store.save(path, object_writer=object_writer)
        self._feature_store.save(path)

    @classmethod
    def load_storage_context(
        cls,
        path: Path,
        object_reader: Optional[Callable[[Path], List[Any]]] = None,
        mmap_features: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return {
            "metadata": MemoryMetadataStore.load(path),
            "objects": MemoryObjectStore.load(path, object_reader=object_reader),
            "features": MemoryFeatureStore.load(path, mmap_features=mmap_features),
        }