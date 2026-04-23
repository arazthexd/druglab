"""
druglab.db.backend.memory
~~~~~~~~~~~~~~~~~~~~~~~~~~
In-memory storage mixins and the EagerMemoryBackend concrete class.

Each mixin handles exactly one data dimension (metadata, objects, features).
They compose via multiple inheritance into ``EagerMemoryBackend``, the
default backend for new tables.

Index normalisation is handled by ``druglab.db.indexing.normalize_row_index``
(via the ``_resolve_idx`` shim defined at the bottom of this module for
internal backward-compatibility).

All reads/writes via ``get_*`` and ``update_*`` methods support ``idx`` 
arguments which is handled by the ``druglab.db.indexing`` module. The same 
interface works identically whether data lives in RAM or on disk.
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ...indexing import normalize_row_index, RowSelection
from ..base import BaseStorageBackend
from .objects import MemoryObjectMixin
from .metadata import MemoryMetadataMixin
from .feature import MemoryFeatureMixin

__all__ = [
    "MemoryMetadataMixin",
    "MemoryObjectMixin",
    "MemoryFeatureMixin",
    "EagerMemoryBackend",
]

####### DEPRECATED #######
# TODO: Remove and replace with composable classes for various capabilities.

class EagerMemoryBackend(
    MemoryMetadataMixin,
    MemoryObjectMixin,
    MemoryFeatureMixin,
    BaseStorageBackend,
):
    """
    Fully eager, in-memory unified storage backend.

    Lifecycle
    ---------
    No custom ``__init__`` is needed.  The cooperative MRO chain propagates
    ``objects=``, ``metadata=``, and ``features=`` kwargs to the appropriate
    mixins.  ``BaseStorageBackend.__init__`` then fires the three lifecycle
    hooks in order:

    1. ``initialize_storage_context`` -- each mixin sets up its storage.
    2. ``bind_capabilities``          -- ``MemoryFeatureMixin`` builds the
                                         live ``FeatureRegistry``.
    3. ``post_initialize_validate``   -- cross-domain dimension check.
    """

    BACKEND_NAME = "EagerMemoryBackend"

    # No __init__ override needed -- cooperative chain handles everything.

    def __len__(self) -> int:
        """
        Return the global, official length of the dataset.

        Returns
        -------
        int
            The authoritative row count (delegated to object count).
        """
        return self._n_objects()

    def create_view(self, indices: Sequence[int]) -> "EagerMemoryBackend":
        """
        Return an independent, deep-copied view restricted to specific indices.
        """
        sel = RowSelection.from_raw(
            np.asarray(indices, dtype=np.intp) if indices else np.array([], dtype=np.intp),
            len(self),
        )

        if sel.is_empty:
            return EagerMemoryBackend()

        new_objects = [copy.deepcopy(self._objects[i]) for i in sel.positions]
        new_metadata = self._metadata.iloc[sel.positions].reset_index(drop=True).copy()
        new_features = {k: v[sel.positions].copy() for k, v in self._features.items()}

        return EagerMemoryBackend(
            objects=new_objects,
            metadata=new_metadata,
            features=new_features,
        )

    # def save(self, path: Path, serializer: Optional[Callable] = None) -> None:
    #     """
    #     Persist backend state into a '.dlb' bundle directory.

    #     Writes metadata as Parquet (or CSV), serializes the entire object 
    #     list into a single pickle file, and saves feature arrays natively 
    #     as `.npy` files.

    #     Parameters
    #     ----------
    #     path : Path
    #         The target `.dlb` directory path (pre-created by the Orchestrator).
    #     serializer : Optional[Callable], default None
    #         An optional function `(obj) -> bytes` to serialize generic objects.
    #     """
    #     path = Path(path)

    #     # --- metadata ---
    #     if not self._metadata.empty:
    #         try:
    #             self._metadata.to_parquet(path / "metadata.parquet", index=False)
    #         except Exception:
    #             self._metadata.to_csv(path / "metadata.csv", index=False)
    #     else:
    #         # Always write metadata so row count is preserved (even if no columns).
    #         pd.DataFrame(index=range(len(self._objects))).to_csv(
    #             path / "metadata.csv", index=True
    #         )

    #     # --- objects ---
    #     obj_dir = path / "objects"
    #     obj_dir.mkdir(exist_ok=True)

    #     # stream_v2: Stream all object payloads (serialized or raw) to prevent list-level pickle OOM spikes.
    #     with open(obj_dir / "objects.pkl", "wb") as f:
    #         pickle.dump(
    #             {
    #                 "format": "stream_v2",
    #                 "count": len(self._objects),
    #                 "serialized": serializer is not None,
    #             },
    #             f,
    #         )
    #         for obj in self._objects:
    #             payload = serializer(obj) if serializer is not None else obj
    #             pickle.dump(payload, f)

    #     # --- features ---
    #     feat_dir = path / "features"
    #     feat_dir.mkdir(exist_ok=True)
    #     for name, arr in self._features.items():
    #         safe_name = name.replace("/", "_").replace("\\", "_")
    #         np.save(str(feat_dir / f"{safe_name}.npy"), arr)

    # @classmethod
    # def load(
    #     cls,
    #     path: Path,
    #     deserializer: Optional[Callable] = None,
    #     mmap_features: bool = False,
    # ) -> "EagerMemoryBackend":
    #     """
    #     Reconstruct the backend from a '.dlb' bundle directory.

    #     Parameters
    #     ----------
    #     path : Path
    #         The location of the `.dlb` bundle.
    #     deserializer : Optional[Callable], default None
    #         An optional function `(bytes) -> obj` to reconstruct stored objects.
    #     mmap_features : bool, default False
    #         If True, loads `.npy` feature files as memory-mapped arrays rather 
    #         than fully pulling them into RAM.

    #     Returns
    #     -------
    #     EagerMemoryBackend
    #         A fully populated instance of the in-memory backend.
    #     """
    #     path = Path(path)

    #     # --- metadata ---
    #     parquet_path = path / "metadata.parquet"
    #     csv_path = path / "metadata.csv"
    #     if parquet_path.exists():
    #         metadata = pd.read_parquet(parquet_path)
    #     elif csv_path.exists():
    #         metadata = pd.read_csv(csv_path)
    #     else:
    #         metadata = pd.DataFrame()

    #     # --- objects ---
    #     obj_path = path / "objects" / "objects.pkl"
    #     if obj_path.exists():
    #         with open(obj_path, "rb") as f:
    #             raw_payload = pickle.load(f)

    #             if isinstance(raw_payload, dict) and raw_payload.get("format") in {
    #                 "stream_v1", "stream_v2"
    #             }:
    #                 count = int(raw_payload["count"])
    #                 raw_list = [pickle.load(f) for _ in range(count)]
    #                 payload_is_serialized = raw_payload.get(
    #                     "format"
    #                 ) == "stream_v1" or bool(raw_payload.get("serialized", False))
    #             else:
    #                 raw_list = raw_payload
    #                 payload_is_serialized = deserializer is not None
    #             if deserializer is not None and payload_is_serialized:
    #                 objects = [deserializer(r) for r in raw_list]
    #             else:
    #                 objects = raw_list
    #     else:
    #         objects = []

    #     # --- features ---
    #     feat_dir = path / "features"
    #     features: Dict[str, np.ndarray] = {}
    #     if feat_dir.exists():
    #         for npy_path in sorted(feat_dir.glob("*.npy")):
    #             name = npy_path.stem
    #             if mmap_features:
    #                 features[name] = np.load(str(npy_path), mmap_mode="r")
    #             else:
    #                 features[name] = np.load(str(npy_path), allow_pickle=False)

    #     return cls(objects=objects, metadata=metadata, features=features)