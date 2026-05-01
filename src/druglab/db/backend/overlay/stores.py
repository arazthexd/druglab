"""
druglab.db.backend.overlay.stores
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overlay domain stores — strict proxy wrappers over their matching base stores.

Each store:
* Inherits from the matching ``Base*Store`` ABC so it satisfies the same
  contract as any concrete store.
* Wraps *exactly one* domain-specific base store (no cross-domain access).
* Intercepts reads/writes using its own ``FeatureDelta`` / ``MetadataDelta`` /
  ``ObjectDelta`` and ``FeatureCache`` / ``MetadataCache``.
* Falls through to the wrapped base store only on a cache/delta miss.
* Supports ``detach()`` (sets ``_base_store = None``) and ``attach()``
  (restores it), raising ``DetachedStateError`` for any base-required reads
  while detached.

Index translation
-----------------
Each store holds its own copy of the overlay's ``_index_map`` (a 1-D
``np.intp`` array mapping overlay positions → base positions).  Row
resolution uses ``normalize_row_index`` from ``druglab.db.indexing``.

View configuration
------------------
Each store holds a *reference* to a shared ``ViewConfig`` dataclass.
``OverlayBackend.set_view()`` rebuilds the ``ViewConfig`` and updates the
reference in every proxy store, so the allowlists stay in sync.

Commit / materialize hooks
--------------------------
The ``commit(base_backend)`` and ``apply_materialized_deltas(concrete_backend)``
helpers accept a full ``BaseStorageBackend`` for API compatibility with
``OverlayBackend.commit()`` / ``OverlayBackend.materialize()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from ...indexing import normalize_row_index
from ..base.stores import BaseFeatureStore, BaseMetadataStore, BaseObjectStore
from .deltas import (
    FeatureCache,
    FeatureDelta,
    MetadataCache,
    MetadataDelta,
    ObjectDelta,
    ViewConfig,
)
from .identity import DetachedStateError

if TYPE_CHECKING:
    from ..base import BaseStorageBackend
    from druglab.db.indexing import INDEX_LIKE

__all__ = [
    "OverlayFeatureStore",
    "OverlayMetadataStore",
    "OverlayObjectStore",
]


# ---------------------------------------------------------------------------
# Shared index-resolution helper
# ---------------------------------------------------------------------------

def _resolve(idx: Optional["INDEX_LIKE"], index_map: np.ndarray) -> np.ndarray:
    """
    Resolve *idx* against the overlay dimension (``len(index_map)``) and
    return a 1-D absolute-overlay-position array.
    """
    n = len(index_map)
    pos = normalize_row_index(idx, n)
    return pos if pos is not None else np.arange(n, dtype=np.intp)

# ===========================================================================
# OverlayFeatureStore
# ===========================================================================

class OverlayFeatureStore(BaseFeatureStore):
    """
    Proxy feature store that layers a ``FeatureDelta`` and ``FeatureCache``
    on top of a ``BaseFeatureStore``.

    Parameters
    ----------
    base_store : BaseFeatureStore
        The concrete store to fall through to on a miss.
    index_map : np.ndarray
        1-D ``np.intp`` array: overlay row i → base row ``index_map[i]``.
    view_config : ViewConfig
        Shared allowlist / column-slice configuration.  Stored by reference
        so ``OverlayBackend.set_view()`` updates stay visible here.
    """

    def __init__(
        self,
        base_store: Optional[BaseFeatureStore],
        index_map: np.ndarray,
        view_config: ViewConfig,
    ) -> None:
        self._base_store: Optional[BaseFeatureStore] = base_store
        self._index_map = index_map
        self._view_config = view_config
        self._delta = FeatureDelta()
        self._cache = FeatureCache()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def delta(self) -> FeatureDelta:
        return self._delta
    
    @property
    def cache(self) -> FeatureCache:
        return self._cache

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_base(self, name: str) -> BaseFeatureStore:
        if self._base_store is None:
            raise DetachedStateError(
                f"Feature '{name}' is not in the delta or cache, "
                "and the overlay is detached."
            )
        return self._base_store

    def _apply_col_slice(self, name: str, arr: np.ndarray) -> np.ndarray:
        cs = self._view_config.get_col_slice(name)
        return cs.apply(arr) if cs is not None else arr

    # ------------------------------------------------------------------
    # BaseFeatureStore abstract interface
    # ------------------------------------------------------------------

    def get_feature(
        self,
        name: str,
        idx: Optional["INDEX_LIKE"] = None,
    ) -> np.ndarray:
        self._view_config.check_feature(name)
        if self._delta.is_deleted(name):
            raise KeyError(f"Feature '{name}' has been deleted from this overlay.")

        overlay_positions = _resolve(idx, self._index_map)

        # 1. Delta
        if self._delta.has(name):
            arr = self._delta.get(name)[overlay_positions]
            return self._apply_col_slice(name, arr)

        # 2. Cache
        if self._cache.has(name):
            arr = self._cache.get(name)[overlay_positions]
            return self._apply_col_slice(name, arr)

        # 3. Base store (strict domain proxy — no metadata/object cross-talk)
        base_positions = self._index_map[overlay_positions]
        arr = self._require_base(name).get_feature(name, idx=base_positions)
        return self._apply_col_slice(name, arr)

    def update_feature(
        self,
        name: str,
        array: np.ndarray,
        idx: Optional["INDEX_LIKE"] = None,
        na: Any = None,
    ) -> None:
        self._view_config.check_feature(name)
        if self._view_config.is_col_sliced(name):
            raise RuntimeError(
                f"Feature '{name}' is read-only due to column slicing."
            )
        self._delta.deleted.discard(name)

        n = len(self._index_map)
        array = np.asarray(array)
        overlay_positions = _resolve(idx, self._index_map)

        if not self._delta.has(name):
            if idx is None:
                if array.shape[0] != n:
                    raise ValueError(
                        f"Array length {array.shape[0]} does not match overlay length {n}."
                    )
                self._delta.set(name, array.copy())
            else:
                # Initialise from base (or scratch) and patch the target rows.
                base = self._base_store
                if (
                    base is not None
                    and name in base.get_feature_names()
                    and not self._delta.is_deleted(name)
                ):
                    full = base.get_feature(name, idx=self._index_map).copy()
                else:
                    shape = (n,) + array.shape[1:]
                    if na is None and np.issubdtype(array.dtype, np.floating):
                        na = np.nan
                    elif na is None:
                        na = 0
                    full = np.full(shape, na, dtype=array.dtype)
                full[overlay_positions] = array
                self._delta.set(name, full)
        else:
            existing = self._delta.get(name)
            if idx is None:
                if array.shape[0] != n:
                    raise ValueError(
                        f"Array length {array.shape[0]} does not match overlay length {n}."
                    )
                self._delta.set(name, array.copy())
            else:
                existing[overlay_positions] = array

        self._cache.evict(name)

    def drop_feature(self, name: str) -> None:
        self._view_config.check_feature(name)
        base_names = (
            self._base_store.get_feature_names()
            if self._base_store is not None
            else []
        )
        if name not in base_names and not self._delta.has(name):
            raise KeyError(f"Feature '{name}' does not exist.")
        self._delta.delete(name)
        self._cache.evict(name)

    def get_feature_names(self) -> List[str]:
        base_names: Set[str] = (
            set(self._base_store.get_feature_names())
            if self._base_store is not None
            else set()
        )
        visible = (base_names - self._delta.deleted) | set(self._delta.names())
        return self._view_config.apply_feature_filter(list(visible))

    def get_feature_shape(self, name: str) -> tuple:
        self._view_config.check_feature(name)
        if self._delta.is_deleted(name):
            raise KeyError(f"Feature '{name}' has been deleted.")
        raw_shape = (
            self._delta.get(name).shape
            if self._delta.has(name)
            else self._require_base(name).get_feature_shape(name)
        )
        cs = self._view_config.get_col_slice(name)
        n = len(self._index_map)
        if cs is not None:
            return (n, cs.stop - cs.start) + raw_shape[2:]
        return (n,) + raw_shape[1:]

    def n_rows(self) -> int:
        return len(self._index_map)

    def gather_materialized_state(
        self,
        index_map: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        # Overlay stores do not expose materialized state directly.
        # Use OverlayBackend.materialize() instead.
        raise NotImplementedError(
            "OverlayFeatureStore does not expose gathered state. "
            "Use OverlayBackend.materialize()."
        )

    def save(self, path: Path, **kwargs: Any) -> None:
        raise NotImplementedError(
            "OverlayFeatureStore persists via OverlayBackend.materialize().save()."
        )

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "OverlayFeatureStore":
        raise NotImplementedError(
            "OverlayFeatureStore cannot be loaded directly."
        )

    # ------------------------------------------------------------------
    # Prefetch / cache management
    # ------------------------------------------------------------------

    def prefetch(
        self,
        names: List[str],
        overlay_positions: Optional[np.ndarray] = None,
    ) -> None:
        """Bulk-read feature rows from the base store into the cache."""
        if self._base_store is None:
            raise DetachedStateError(
                "Cannot prefetch features: overlay is detached."
            )
        if overlay_positions is None:
            overlay_positions = np.arange(len(self._index_map), dtype=np.intp)
        base_positions = self._index_map[overlay_positions]
        base_positions = base_positions[base_positions >= 0]
        if base_positions.size == 0:
            return
        for name in names:
            if self._delta.has(name) or self._delta.is_deleted(name):
                continue
            self._cache.put(
                name,
                self._base_store.get_feature(name, idx=base_positions),
            )

    # ------------------------------------------------------------------
    # Commit / materialize helpers
    # ------------------------------------------------------------------

    def append(self, data: Any) -> None:
        n_new = next(iter(data.values())).shape[0] if data else 0
        if n_new == 0:
            return
        n_current = len(self._index_map)
        self._index_map = np.concatenate([self._index_map, np.full(n_new, -1, dtype=np.intp)])
        for name, values in data.items():
            if self._delta.has(name):
                existing = self._delta.get(name)
                self._delta.set(name, np.vstack([existing, values]))
            else:
                if self._base_store is not None and name in self._base_store.get_feature_names():
                    base_arr = self._base_store.get_feature(name, idx=self._index_map[:n_current])
                    self._delta.set(name, np.vstack([base_arr, values]))
                else:
                    self._delta.set(name, values.copy())
            self._cache.evict(name)

    def commit(self, base_backend: "BaseStorageBackend") -> None:
        """Flush all deltas to *base_backend* and clear the delta store."""
        virtual_mask = self._index_map == -1
        n_appended = int(np.count_nonzero(virtual_mask))

        mutate_idx = np.where(self._index_map >= 0)[0]
        if mutate_idx.size > 0:
            mapped_idx = self._index_map[mutate_idx]
            for name, arr in self._delta.local.items():
                base_backend.update_feature(name, arr[mutate_idx], idx=mapped_idx)

        if n_appended > 0:
            append_payload = {name: arr[virtual_mask] for name, arr in self._delta.local.items()}
            if append_payload:
                base_backend._feature_store.append(append_payload) # TODO: ALERT / Add ABC contract to base backend

        for name in self._delta.deleted:
            if name in base_backend.get_feature_names():
                base_backend.drop_feature(name)
    
        self._delta.clear()

    def apply_materialized_deltas(
        self, concrete_backend: "BaseStorageBackend"
    ) -> None:
        """Replay deltas onto an already-cloned concrete backend."""
        for name in self._delta.deleted:
            if name in concrete_backend.get_feature_names():
                concrete_backend.drop_feature(name)
        for name in self._delta.names():
            if name not in self._delta.deleted:
                concrete_backend.update_feature(name, self._delta.get(name))

    def deep_copy_delta(self) -> FeatureDelta:
        return self._delta.deep_copy()


# ===========================================================================
# OverlayMetadataStore
# ===========================================================================

class OverlayMetadataStore(BaseMetadataStore):
    """
    Proxy metadata store that layers a ``MetadataDelta`` and
    ``MetadataCache`` on top of a ``BaseMetadataStore``.
    """

    def __init__(
        self,
        base_store: Optional[BaseMetadataStore],
        index_map: np.ndarray,
        view_config: ViewConfig,
    ) -> None:
        self._base_store: Optional[BaseMetadataStore] = base_store
        self._index_map = index_map
        self._view_config = view_config
        self._delta = MetadataDelta()
        self._cache = MetadataCache()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def delta(self) -> MetadataDelta:
        return self._delta
    
    @property
    def cache(self) -> MetadataCache:
        return self._cache

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_base(self) -> BaseMetadataStore:
        if self._base_store is None:
            raise DetachedStateError(
                "Metadata read missed delta/cache while detached."
            )
        return self._base_store

    def _visible_columns(self) -> List[str]:
        base_cols: Set[str] = (
            set(self._base_store.get_metadata_columns())
            if self._base_store is not None
            else set()
        )
        local_cols: Set[str] = (
            set(self._delta.local.columns)
            if self._delta.local is not None
            else set()
        )
        return self._view_config.apply_meta_col_filter(
            list((base_cols - self._delta.deleted_cols) | local_cols)
        )

    # ------------------------------------------------------------------
    # BaseMetadataStore abstract interface
    # ------------------------------------------------------------------

    def get_metadata(
        self,
        idx: Optional["INDEX_LIKE"] = None,
        cols: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        overlay_positions = _resolve(idx, self._index_map)
        base_positions = self._index_map[overlay_positions]

        base_df = (
            self._require_base()
            .get_metadata(idx=base_positions)
            .reset_index(drop=True)
        )

        # Strip deleted columns from base result
        drop_from_base = [
            c for c in self._delta.deleted_cols if c in base_df.columns
        ]
        if drop_from_base:
            base_df = base_df.drop(columns=drop_from_base)

        # Overlay local delta columns
        if self._delta.local is not None:
            local_slice = (
                self._delta.local.iloc[overlay_positions].reset_index(drop=True)
            )
            for col in local_slice.columns:
                if col not in self._delta.deleted_cols:
                    base_df[col] = local_slice[col].values

        # Apply view allowlist
        if self._view_config.allowed_meta_cols is not None:
            visible = [
                c for c in base_df.columns
                if c in self._view_config.allowed_meta_cols
            ]
            base_df = base_df[visible]

        if cols is not None:
            if isinstance(cols, str):
                cols = [cols]
            for c in cols:
                self._view_config.check_meta_col(c)
            base_df = base_df[cols]

        return base_df.reset_index(drop=True)

    def add_metadata_column(
        self,
        name: str,
        value: Union[pd.Series, np.ndarray, List[Any]],
        idx: Optional["INDEX_LIKE"] = None,
        na: Any = None,
    ) -> None:
        self._view_config.check_meta_col(name)
        self._delta.deleted_cols.discard(name)
        value = np.asarray(value)
        n = len(self._index_map)
        self._delta.ensure_local(n)
        overlay_positions = _resolve(idx, self._index_map)

        if idx is None:
            self._delta.local[name] = value
        else:
            if name not in self._delta.local.columns:
                if na is None and np.issubdtype(value.dtype, np.floating):
                    na = np.nan
                elif na is None:
                    na = 0
                self._delta.local[name] = np.full(n, na, dtype=value.dtype)
            col_pos = self._delta.local.columns.get_loc(name)
            self._delta.local.iloc[overlay_positions, col_pos] = value
        self._cache.evict_col(name)

    def update_metadata(
        self,
        values: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
        idx: Optional["INDEX_LIKE"] = None,
    ) -> None:
        if isinstance(values, pd.DataFrame):
            val_dict = {col: values[col].values for col in values.columns}
        elif isinstance(values, pd.Series):
            if values.name is None:
                raise ValueError("Series must have a name.")
            val_dict = {values.name: values.values}
        else:
            val_dict = dict(values)

        all_visible = self._visible_columns()
        for col in val_dict:
            self._view_config.check_meta_col(col)
            if col not in all_visible:
                raise KeyError(
                    f"Column '{col}' does not exist. Use add_metadata_column."
                )

        n = len(self._index_map)
        overlay_positions = _resolve(idx, self._index_map)
        self._delta.ensure_local(n)

        for col, val in val_dict.items():
            if col not in self._delta.local.columns:
                # Pull the full column from base, aligned to this overlay's rows
                base_col = (
                    self._require_base()
                    .get_metadata(idx=self._index_map, cols=[col])
                    .reset_index(drop=True)[col]
                    .values
                )
                self._delta.local[col] = base_col
            if idx is None:
                self._delta.local[col] = val
            else:
                col_pos = self._delta.local.columns.get_loc(col)
                self._delta.local.iloc[overlay_positions, col_pos] = val
            self._cache.evict_col(col)

    def drop_metadata_columns(
        self,
        cols: Optional[Union[str, List[str]]] = None,
    ) -> None:
        if cols is None:
            self._delta.drop_all_visible(self._visible_columns())
            self._cache.clear()
            return
        if isinstance(cols, str):
            cols = [cols]
        for col in cols:
            self._delta.drop_col(col)
            self._cache.evict_col(col)

    def get_metadata_columns(self) -> List[str]:
        return self._visible_columns()

    def n_rows(self) -> int:
        return len(self._index_map)

    def gather_materialized_state(
        self,
        index_map: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "OverlayMetadataStore does not expose gathered state. "
            "Use OverlayBackend.materialize()."
        )

    def save(self, path: Path, **kwargs: Any) -> None:
        raise NotImplementedError(
            "OverlayMetadataStore persists via OverlayBackend.materialize().save()."
        )

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "OverlayMetadataStore":
        raise NotImplementedError(
            "OverlayMetadataStore cannot be loaded directly."
        )

    # ------------------------------------------------------------------
    # Prefetch / cache management
    # ------------------------------------------------------------------

    def prefetch(
        self,
        cols: List[str],
        overlay_positions: Optional[np.ndarray] = None,
    ) -> None:
        """Bulk-read metadata columns from the base store into the cache."""
        if self._base_store is None:
            raise DetachedStateError(
                "Cannot prefetch metadata: overlay is detached."
            )
        base_positions = (
            self._index_map
            if overlay_positions is None
            else self._index_map[overlay_positions]
        )
        self._cache.put(
            self._base_store.get_metadata(
                idx=base_positions, cols=cols
            ).reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Commit / materialize helpers
    # ------------------------------------------------------------------

    def append(self, data: Any) -> None:
        n_new = len(data)
        if n_new == 0:
            return
        n_current = len(self._index_map)
        self._index_map = np.concatenate([self._index_map, np.full(n_new, -1, dtype=np.intp)])
        self._delta.ensure_local(n_current + n_new)
        for col in data.columns:
            self._delta.local.loc[n_current:, col] = data[col].values
        self._cache.clear()

    def commit(self, base_backend: "BaseStorageBackend") -> None:
        """Flush all deltas to *base_backend* and clear the delta store."""
        virtual_mask = self._index_map == -1
        mutate_idx = np.where(self._index_map >= 0)[0]
        n_appended = int(np.count_nonzero(virtual_mask))
        base_len = len(base_backend)

        if self._delta.local is not None and not self._delta.local.empty:
            base_cols = set(base_backend.get_metadata_columns())
            if mutate_idx.size > 0:
                mapped_idx = self._index_map[mutate_idx]
                for col in self._delta.local.columns:
                    val = self._delta.local[col].values[mutate_idx]
                    if col in base_cols:
                        base_backend.update_metadata({col: val}, idx=mapped_idx)
                    else:
                        base_backend.add_metadata_column(col, val, idx=mapped_idx)
            if n_appended > 0:
                base_backend._metadata_store.append(self._delta.local.iloc[virtual_mask].reset_index(drop=True))

        if self._delta.deleted_cols:
            existing = set(base_backend.get_metadata_columns())
            cols_to_drop = list(self._delta.deleted_cols & existing)
            if cols_to_drop:
                base_backend.drop_metadata_columns(cols_to_drop)

        self._delta.clear()

    def apply_materialized_deltas(
        self, concrete_backend: "BaseStorageBackend"
    ) -> None:
        """Replay deltas onto an already-cloned concrete backend."""
        if self._delta.deleted_cols:
            existing_cols = set(concrete_backend.get_metadata_columns())
            to_drop = list(self._delta.deleted_cols & existing_cols)
            if to_drop:
                concrete_backend.drop_metadata_columns(to_drop)
        if self._delta.local is not None and not self._delta.local.empty:
            existing_cols = set(concrete_backend.get_metadata_columns())
            for col in self._delta.local.columns:
                if col in self._delta.deleted_cols:
                    continue
                val = self._delta.local[col].values
                if col in existing_cols:
                    concrete_backend.update_metadata({col: val})
                else:
                    concrete_backend.add_metadata_column(col, val)

    def deep_copy_delta(self) -> MetadataDelta:
        return self._delta.deep_copy()


# ===========================================================================
# OverlayObjectStore
# ===========================================================================

class OverlayObjectStore(BaseObjectStore):
    """
    Proxy object store that tracks individual mutations in a sparse
    ``ObjectDelta`` and falls through to a ``BaseObjectStore`` on a miss.
    """

    def __init__(
        self,
        base_store: Optional[BaseObjectStore],
        index_map: np.ndarray,
    ) -> None:
        self._base_store: Optional[BaseObjectStore] = base_store
        self._index_map = index_map
        self._delta = ObjectDelta()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def delta(self) -> ObjectDelta:
        return self._delta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_base(self, overlay_idx: int) -> BaseObjectStore:
        if self._base_store is None:
            raise DetachedStateError(
                f"Object at overlay index {overlay_idx} is not in the delta "
                "and the overlay is detached."
            )
        return self._base_store

    # ------------------------------------------------------------------
    # BaseObjectStore abstract interface
    # ------------------------------------------------------------------

    def get_objects(
        self,
        idx: Optional["INDEX_LIKE"] = None,
    ):
        scalar = isinstance(idx, (int, np.integer))
        overlay_positions = _resolve(idx, self._index_map)

        results = []
        for oi in overlay_positions:
            oi_int = int(oi)
            if self._delta.has(oi_int):
                results.append(self._delta.get(oi_int))
            else:
                base_pos = int(self._index_map[oi_int])
                results.append(
                    self._require_base(oi_int).get_objects(idx=base_pos)
                )
        return results[0] if scalar else results

    def update_objects(
        self,
        objs,
        idx: Optional["INDEX_LIKE"] = None,
    ) -> None:
        overlay_positions = _resolve(idx, self._index_map)

        if idx is None:
            if len(objs) != len(overlay_positions):
                raise ValueError(
                    "Length of objs must match overlay length."
                )
            for oi, obj in zip(overlay_positions, objs):
                self._delta.set(int(oi), obj)
            return

        if isinstance(idx, (int, np.integer)):
            self._delta.set(int(overlay_positions[0]), objs)
            return

        if len(objs) != len(overlay_positions):
            raise ValueError("Length of objs must match index length.")
        for oi, obj in zip(overlay_positions, objs):
            self._delta.set(int(oi), obj)

    def n_rows(self) -> int:
        return len(self._index_map)

    def gather_materialized_state(
        self,
        index_map: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "OverlayObjectStore does not expose gathered state. "
            "Use OverlayBackend.materialize()."
        )

    def save(self, path: Path, **kwargs: Any) -> None:
        raise NotImplementedError(
            "OverlayObjectStore persists via OverlayBackend.materialize().save()."
        )

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "OverlayObjectStore":
        raise NotImplementedError(
            "OverlayObjectStore cannot be loaded directly."
        )

    # ------------------------------------------------------------------
    # Commit / materialize helpers
    # ------------------------------------------------------------------

    def append(self, data: Any) -> None:
        n_new = len(data)
        if n_new == 0:
            return
        n_current = len(self._index_map)
        self._index_map = np.concatenate([self._index_map, np.full(n_new, -1, dtype=np.intp)])
        for offset, obj in enumerate(data):
            self._delta.set(n_current + offset, obj)

    def commit(self, base_backend: "BaseStorageBackend") -> None:
        """Flush all deltas to *base_backend* and clear the delta store."""
        appended_objs = []
        for overlay_idx, obj in sorted(self._delta.local.items()):
            base_idx = int(self._index_map[overlay_idx])
            if base_idx == -1:
                appended_objs.append(obj)
            else:
                base_backend.update_objects(obj, idx=base_idx)
        if appended_objs:
            base_backend._object_store.append(appended_objs)
        self._delta.clear()

    def apply_materialized_deltas(
        self, concrete_backend: "BaseStorageBackend"
    ) -> None:
        """Replay deltas onto an already-cloned concrete backend."""
        for oi, obj in self._delta.local.items():
            concrete_backend.update_objects(obj, idx=int(oi))

    def deep_copy_delta(self) -> ObjectDelta:
        return self._delta.deep_copy()