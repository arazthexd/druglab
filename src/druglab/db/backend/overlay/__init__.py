"""Composition-based OverlayBackend."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ...indexing import normalize_row_index
from ..base import BaseStorageBackend
from .deltas import ColumnSlice, ViewConfig
from .identity import DetachedStateError, SchemaIdentity
from .mixins import OverlayFeatureStore, OverlayMetadataStore, OverlayObjectStore

if TYPE_CHECKING:
    from druglab.db.indexing import INDEX_LIKE

__all__ = ["OverlayBackend"]


class OverlayBackend(BaseStorageBackend):
    BACKEND_NAME = "OverlayBackend"

    def __init__(self, base_backend: BaseStorageBackend, index_map: Optional[np.ndarray] = None, **kwargs) -> None:
        if isinstance(base_backend, OverlayBackend):
            outer_map = base_backend._index_map
            composed = outer_map.copy() if index_map is None else outer_map[np.asarray(index_map, dtype=np.intp)]
            self._base = base_backend._base
            self._index_map = composed
        else:
            self._base = base_backend
            self._index_map = np.arange(len(base_backend), dtype=np.intp) if index_map is None else np.asarray(index_map, dtype=np.intp)

        self._base_identity = SchemaIdentity.capture(self._base) if self._base is not None else None
        self._view_config = ViewConfig()

        self._object_store = OverlayObjectStore(self)
        self._metadata_store = OverlayMetadataStore(self)
        self._feature_store = OverlayFeatureStore(self)
        self._sync_legacy_refs()

    def _sync_legacy_refs(self) -> None:
        """Compatibility aliases for existing tests/callers."""
        self._obj_delta = self._object_store.delta
        self._meta_delta = self._metadata_store.delta
        self._feat_delta = self._feature_store.delta
        self._meta_cache = self._metadata_store.cache
        self._feat_cache = self._feature_store.cache

    def __len__(self) -> int:
        return len(self._index_map)

    def _translate(self, overlay_positions: np.ndarray) -> np.ndarray:
        return self._index_map[overlay_positions]

    def _n_rows(self) -> int:
        return len(self._index_map)

    def _resolve_overlay_idx(self, idx: Optional["INDEX_LIKE"]) -> np.ndarray:
        positions = normalize_row_index(idx, self._n_rows())
        if positions is None:
            return np.arange(self._n_rows(), dtype=np.intp)
        return positions

    # Delegated object API
    def get_objects(self, idx=None):
        return self._object_store.get_objects(idx)

    def update_objects(self, objs, idx=None, **kwargs) -> None:
        self._object_store.update_objects(objs, idx)

    def _n_objects(self) -> int:
        return self._n_rows()

    # Delegated metadata API
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
        return self._n_rows()

    # Delegated feature API
    def get_feature(self, name: str, idx=None):
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
        return self._n_rows()

    def validate(self) -> None:
        return

    def set_view(self, features: Optional[List[str]] = None, meta_cols: Optional[List[str]] = None, feature_col_slices: Optional[Dict[str, Tuple[int, int]]] = None) -> None:
        col_slices = {}
        if feature_col_slices:
            for name, (start, stop) in feature_col_slices.items():
                col_slices[name] = ColumnSlice(start=start, stop=stop, read_only=True)
        self._view_config = ViewConfig(
            allowed_features=set(features) if features is not None else None,
            allowed_meta_cols=set(meta_cols) if meta_cols is not None else None,
            feature_col_slices=col_slices,
        )

    def clear_view(self) -> None:
        """Remove all view restrictions (reset to unrestricted)."""
        self._view_config = ViewConfig()

    def prefetch(self, features: Optional[List[str]] = None, meta_cols: Optional[List[str]] = None, rows: Optional["INDEX_LIKE"] = None) -> None:
        if self._base is None:
            raise DetachedStateError("Cannot prefetch: overlay is already detached.")
        overlay_positions = self._resolve_overlay_idx(rows) if rows is not None else None
        feat_names = features if features is not None else self._base.get_feature_names()
        self._feature_store.prefetch(feat_names, overlay_positions)
        all_base_cols = self._base.get_metadata_columns()
        col_list = meta_cols if meta_cols is not None else all_base_cols
        if col_list:
            self._metadata_store.prefetch(col_list, overlay_positions)

    def detach(self) -> None:
        """
        Sever the reference to the base backend.

        After detaching the overlay is self-contained: any read that can be
        satisfied from the delta or the prefetch cache will succeed; any read
        that would fall through to the (now absent) base will raise
        ``DetachedStateError``.

        This is the standard preparation for sending an overlay across a
        process boundary (e.g. via pickle).
        """
        self._base = None

    def attach(self, base_backend: BaseStorageBackend) -> None:
        """
        Re-attach a base backend after detaching or cross-process transport.

        Parameters
        ----------
        base_backend : BaseStorageBackend
            The backend to attach.  Must be schema-compatible with the identity
            captured at creation time (same UUID, row count, feature schema,
            and metadata schema).

        Raises
        ------
        ValueError
            If the supplied backend is not schema-compatible.
        """
        if self._base_identity is None:
            # Overlay was created without a base (unusual); accept anything.
            self._base = base_backend
            self._base_identity = SchemaIdentity.capture(base_backend)
            return
        self._base_identity.validate_compatible(SchemaIdentity.capture(base_backend))
        self._base = base_backend

    @property
    def is_detached(self) -> bool:
        """``True`` when the overlay has no base backend reference."""
        return self._base is None

    def clone(self, target_path: Optional[Path] = None, index_map: Optional[np.ndarray] = None) -> "OverlayBackend":
        clone_index_map = self._index_map.copy() if index_map is None else np.asarray(index_map, dtype=np.intp)
        cloned = object.__new__(self.__class__)
        cloned._base = self._base
        cloned._base_identity = self._base_identity
        cloned._index_map = clone_index_map
        cloned._view_config = ViewConfig(
            allowed_features=set(self._view_config.allowed_features) if self._view_config.allowed_features is not None else None,
            allowed_meta_cols=set(self._view_config.allowed_meta_cols) if self._view_config.allowed_meta_cols is not None else None,
            feature_col_slices=dict(self._view_config.feature_col_slices),
        )
        cloned._object_store = OverlayObjectStore(cloned)
        cloned._metadata_store = OverlayMetadataStore(cloned)
        cloned._feature_store = OverlayFeatureStore(cloned)
        cloned._object_store.delta.local = dict(self._object_store.delta.local)
        cloned._metadata_store._delta = self._metadata_store.delta.deep_copy()
        cloned._feature_store._delta = self._feature_store.delta.deep_copy()
        cloned._sync_legacy_refs()
        return cloned

    def materialize(self, target_path: Optional[Path] = None) -> BaseStorageBackend:
        """
        Collapse the overlay into a standalone concrete backend.

        Applies the full delta stack on top of the (optionally row-sliced)
        base data.  The result is an ``EagerMemoryBackend`` (or whatever class
        ``self._base`` is) that is completely independent of this overlay.
        """
        if self._base is None:
            raise DetachedStateError("Cannot materialize a detached overlay.")
        concrete = self._base.clone(target_path=target_path, index_map=self._index_map)
        self._feature_store.apply_materialized_deltas(concrete)
        self._metadata_store.apply_materialized_deltas(concrete)
        self._object_store.apply_materialized_deltas(concrete)
        return concrete

    def commit(self) -> None:
        """
        Flush all local deltas to the base backend.

        Only **delta** data is written; cached data is ignored.  After commit
        all delta stores are cleared; subsequent reads fall through to the base
        (or cache) again.

        Raises
        ------
        DetachedStateError
            If the overlay is currently detached.
        """
        if self._base is None:
            raise DetachedStateError("Cannot commit: overlay is detached.")
        self._feature_store.commit(self._base, self._index_map)
        self._metadata_store.commit(self._base, self._index_map)
        self._object_store.commit(self._base, self._index_map)

    def save(self, path: Path, object_writer: Optional[Callable[[list, Path], None]] = None, **kwargs) -> None:
        self.materialize().save(path, object_writer=object_writer, **kwargs)

    def get_name(self) -> str:
        return self._base.__class__.__name__ if self._base is not None else "OverlayBackend"

    def get_module(self) -> str:
        return self._base.__class__.__module__ if self._base is not None else __name__