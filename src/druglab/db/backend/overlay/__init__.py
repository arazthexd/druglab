"""Overlay backend with mixin-based CoW delta handling."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

import numpy as np

from ...indexing import normalize_row_index
from ..base import BaseStorageBackend
from .mixins import OverlayFeatureMixin, OverlayMetadataMixin, OverlayObjectMixin

if TYPE_CHECKING:
    from druglab.db.indexing import INDEX_LIKE

__all__ = ["OverlayBackend"]


class OverlayBackend(
    OverlayObjectMixin,
    OverlayMetadataMixin,
    OverlayFeatureMixin,
    BaseStorageBackend,
):
    """A zero-copy proxy backend with Copy-on-Write semantics."""

    BACKEND_NAME = "OverlayBackend"

    def __init__(
        self,
        base_backend: BaseStorageBackend,
        index_map: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        if isinstance(base_backend, OverlayBackend):
            outer_map = base_backend._index_map
            composed = outer_map.copy() if index_map is None else outer_map[np.asarray(index_map, dtype=np.intp)]
            self._base = base_backend._base
            self._index_map = composed
        else:
            self._base = base_backend
            self._index_map = (
                np.arange(len(base_backend), dtype=np.intp)
                if index_map is None
                else np.asarray(index_map, dtype=np.intp)
            )

        self._initialize_overlay_context(**kwargs)

    def __len__(self) -> int:
        return len(self._index_map)

    def _translate(self, overlay_idx: np.ndarray) -> np.ndarray:
        return self._index_map[overlay_idx]

    def _n_rows(self) -> int:
        return len(self._index_map)

    def _resolve_overlay_idx(self, idx: Optional[INDEX_LIKE]) -> np.ndarray:
        positions = normalize_row_index(idx, self._n_rows())
        if positions is None:
            return np.arange(self._n_rows(), dtype=np.intp)
        return positions

    def initialize_storage_context(self, **kwargs) -> None:
        return

    def bind_capabilities(self) -> None:
        return

    def post_initialize_validate(self) -> None:
        return

    def validate(self) -> None:
        return

    def clone(
        self,
        target_path: Optional[Path] = None,
        index_map: Optional[np.ndarray] = None,
    ) -> BaseStorageBackend:
        clone_index_map = self._index_map.copy() if index_map is None else np.asarray(index_map, dtype=np.intp)
        cloned = self.__class__(self._base, clone_index_map)
        cloned._local_features = copy.deepcopy(self._local_features)
        cloned._local_metadata = None if self._local_metadata is None else self._local_metadata.copy(deep=True)
        cloned._local_objects = copy.deepcopy(self._local_objects)
        cloned._deleted_features = copy.deepcopy(self._deleted_features)
        cloned._deleted_metadata_cols = copy.deepcopy(self._deleted_metadata_cols)
        return cloned

    def materialize(self, target_path: Optional[Path] = None) -> BaseStorageBackend:
        concrete: BaseStorageBackend = self._base.clone(
            target_path=target_path,
            index_map=self._index_map,
        )
        self._apply_materialized_deltas(concrete)
        return concrete

    def commit(self) -> None:
        self._commit_deltas(self._base)

    def save(
        self,
        path: Path,
        object_writer: Optional[Callable[[list[Any], Path], None]] = None,
        **kwargs,
    ) -> None:
        self.materialize().save(path, object_writer=object_writer, **kwargs)

    def get_name(self) -> str:
        return self._base.__class__.__name__

    def get_module(self) -> str:
        return self._base.__class__.__module__