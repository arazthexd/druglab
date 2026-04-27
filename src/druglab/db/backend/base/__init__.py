"""
druglab.db.backend.base
~~~~~~~~~~~~~~~~~~~~~~~
Abstract base interface that all storage backends must implement.

The interface enforces strict Query Pushdown: index/slice arguments must be
passed directly to the backend so that out-of-core implementations (Zarr,
SQLite, HDF5) can read exactly the bytes they need without loading full
arrays into memory first.

Index normalisation is handled by ``druglab.db.indexing``, which is the
single source of truth for all row-addressing in DrugLab.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid as _uuid_mod

import numpy as np
import pandas as pd
from typing_extensions import Self

__all__ = ["BaseStorageBackend"]

class BaseStorageBackend(ABC):
    """
    Minimal unified interface for managing DrugLab table state.

    Thread-safety notice
    --------------------
    Concrete storage backends mutate internal arrays/lists in place and do
    not provide write locks. Concurrent writes against the same backend
    instance are not thread-safe and are not process-safe.

    Pipeline orchestration is responsible for synchronization. Multiprocessing
    code must use ``OverlayBackend`` scatter-gather (prefetch -> detach ->
    worker mutation -> attach -> commit) instead of sharing mutable base
    backends across workers.

    ``backend.schema_uuid`` is a per-instance random UUID used by
    ``OverlayBackend.attach()`` to verify that a re-attached backend is the
    same instance (or an intentional clone) as the one detached from.
    """

    def __init__(self) -> None:
        self.schema_uuid: str = str(_uuid_mod.uuid4())

    # ------------------------------------------------------------------
    # Required domain APIs
    # ------------------------------------------------------------------

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    # Objects
    @abstractmethod
    def get_objects(self, idx=None):
        raise NotImplementedError

    @abstractmethod
    def update_objects(self, objs, idx=None, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def _n_objects(self) -> int:
        raise NotImplementedError

    # Metadata
    @abstractmethod
    def get_metadata(self, idx=None, cols=None):
        raise NotImplementedError

    @abstractmethod
    def add_metadata_column(self, name, value, idx=None, na=None, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_metadata(self, values, idx=None, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def drop_metadata_columns(self, cols=None) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_metadata_columns(self):
        raise NotImplementedError

    @abstractmethod
    def _n_metadata_rows(self) -> int:
        raise NotImplementedError

    # Features
    @abstractmethod
    def get_feature(self, name: str, idx=None) -> np.ndarray:
        raise NotImplementedError

    def get_features(
        self,
        names: Optional[List[str]] = None,
        idx=None,
    ) -> Dict[str, np.ndarray]:
        if names is None:
            names = self.get_feature_names()
        return {name: self.get_feature(name, idx) for name in names}

    @abstractmethod
    def update_feature(self, name: str, array: np.ndarray, idx=None, na=None, **kwargs) -> None:
        raise NotImplementedError

    def update_features(self, arrays: Dict[str, np.ndarray], idx=None, na=None, **kwargs) -> None:
        for name, array in arrays.items():
            self.update_feature(name, array, idx, na, **kwargs)

    @abstractmethod
    def drop_feature(self, name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_feature_names(self):
        raise NotImplementedError

    @abstractmethod
    def get_feature_shape(self, name: str) -> tuple:
        raise NotImplementedError

    def _n_feature_rows(self) -> int:
        names = self.get_feature_names()
        if not names:
            return len(self)
        return self.get_feature_shape(names[0])[0]

    # ------------------------------------------------------------------
    # Lifecycle-free materialization and persistence hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _gather_materialized_state(
        self,
        target_path: Optional[Path] = None,
        index_map: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def save_storage_context(self, path: Path, **kwargs: Any) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_storage_context(cls, path: Path, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def clone(
        self,
        target_path: Optional[Path] = None,
        index_map: Optional[np.ndarray] = None,
    ) -> "BaseStorageBackend":
        """
        Build a new backend instance of *this* class with the same state.
 
        Parameters
        ----------
        target_path : Path, optional
            Reserved for future out-of-core backends.
        index_map : np.ndarray of dtype np.intp, optional
            Absolute row positions to include.  ``None`` → all rows.
 
        Returns
        -------
        BaseStorageBackend
            A new instance of ``type(self)`` with state matching *index_map*.
        """
        gathered = self._gather_materialized_state(target_path=target_path, index_map=index_map)
        new_instance = self.__class__(**gathered)
        # Clones intentionally get a NEW uuid so attach() distinguishes them.
        new_instance.schema_uuid = str(_uuid_mod.uuid4())
        return new_instance
    
    def materialize(
        self,
        target_path: Optional[Path] = None,
    ) -> "BaseStorageBackend":
        """
        Return a disconnected backend instance representing this backend's
        current logical state.

        Concrete backends are already materialized, so their safe behavior is
        to return a deep copy via ``clone()``. Proxy backends (for example
        ``OverlayBackend``) may override this method to collapse deferred
        deltas into a concrete backend.
        """
        return self.clone(target_path=target_path)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str, **kwargs: Any) -> None:
        path = Path(path)
        if path.exists():
            print("WARNING: A .dlb bundle already exists. Overwriting.")
        path.mkdir(parents=True, exist_ok=True)
        self.save_storage_context(path=path, **kwargs)

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> Self:
        path = Path(path)
        cls_kwargs = cls.load_storage_context(path=path, **kwargs)
        return cls(**cls_kwargs)

    def get_name(self) -> str:
        return self.__class__.__name__

    def get_module(self) -> str:
        return self.__class__.__module__

    @abstractmethod
    def validate(self) -> None:
        raise NotImplementedError