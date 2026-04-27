from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..composite import CompositeStorageBackend
from .feature import MemoryFeatureStore
from .metadata import MemoryMetadataStore
from .objects import MemoryObjectStore

__all__ = [
    "MemoryMetadataStore",
    "MemoryObjectStore",
    "MemoryFeatureStore",
    "EagerMemoryBackend",
]


class EagerMemoryBackend(CompositeStorageBackend):
    BACKEND_NAME = "EagerMemoryBackend"

    def __init__(
        self,
        objects: Optional[List[Any]] = None,
        metadata: Optional[pd.DataFrame] = None,
        features: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        object_store = MemoryObjectStore(objects)
        metadata_store = MemoryMetadataStore(metadata, n_rows_hint=object_store.n_rows())
        feature_store = MemoryFeatureStore(features, n_rows_hint=object_store.n_rows())
        super().__init__(
            object_store=object_store,
            metadata_store=metadata_store,
            feature_store=feature_store,
        )

    @classmethod
    def load_storage_context(
        cls,
        path: Path,
        object_reader: Optional[Callable[[Path], List[Any]]] = None,
        mmap_features: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        object_store = MemoryObjectStore.load(path, object_reader=object_reader)
        metadata_store = MemoryMetadataStore.load(path)
        feature_store = MemoryFeatureStore.load(path, mmap_features=mmap_features)
        return {
            "objects": object_store.get_objects(),
            "metadata": metadata_store.get_metadata(),
            "features": {
                name: feature_store.get_feature(name)
                for name in feature_store.get_feature_names()
            },
        }