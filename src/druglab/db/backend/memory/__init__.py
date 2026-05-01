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
        metadata_store = MemoryMetadataStore(
            metadata, n_rows_hint=object_store.n_rows()
        )
        feature_store = MemoryFeatureStore(
            features, n_rows_hint=object_store.n_rows()
        )
        super().__init__(
            object_store=object_store,
            metadata_store=metadata_store,
            feature_store=feature_store,
        )

    # ------------------------------------------------------------------
    # Override save_storage_context to accept metadata_format kwarg
    # ------------------------------------------------------------------

    def save_storage_context(
        self,
        path: Path,
        metadata_format: str = "parquet",
        **kwargs: Any,
    ) -> None:
        """Delegate to CompositeStorageBackend, forwarding *metadata_format*."""
        super().save_storage_context(path, metadata_format=metadata_format, **kwargs)

    # ------------------------------------------------------------------
    # load_storage_context returns raw constructor kwargs
    # ------------------------------------------------------------------

    @classmethod
    def load_storage_context(
        cls,
        path: Path,
        object_reader: Optional[Callable[[Path], List[Any]]] = None,
        mmap_features: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Reconstruct ``EagerMemoryBackend`` constructor kwargs from a bundle.

        Delegates to ``CompositeStorageBackend.load_storage_context`` (which
        reads the manifest and threads ``format`` to the metadata store), then
        extracts the raw arrays so ``EagerMemoryBackend.__init__`` can
        re-validate dimensions.
        """
        # Let the composite parent do the heavy lifting (reads manifest, format, etc.)
        stores = CompositeStorageBackend.load_storage_context(
            path, object_reader=object_reader, **kwargs
        )
        object_store: MemoryObjectStore = stores["object_store"]
        metadata_store: MemoryMetadataStore = stores["metadata_store"]
        feature_store: MemoryFeatureStore = stores["feature_store"]

        # Re-load features with optional memory mapping (EagerMemoryBackend-specific).
        if mmap_features:
            from druglab.db.backend.memory.feature import MemoryFeatureStore as _FS
            feature_store = _FS.load(path, mmap_features=True)

        return {
            "objects": object_store.get_objects(),
            "metadata": metadata_store.get_metadata(),
            "features": {
                name: feature_store.get_feature(name)
                for name in feature_store.get_feature_names()
            },
        }