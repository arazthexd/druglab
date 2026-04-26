"""
druglab.db.backend.overlay.mixins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Mixin classes that implement the feature, metadata, and object domains for an
overlay backend.

An overlay backend sits on top of a base storage backend and provides copy-on-write
mutations, a read cache, and optional view constraints (allowlists and column slices).
The mixins in this module are designed to be used together in a cooperative multiple
inheritance hierarchy (e.g., along with `OverlayBackend`) and rely on a shared
context provided by the concrete class.

Key concepts
------------
- **Delta**: A mutable, COW (copy-on-write) container that stores local changes without
  affecting the base backend. Each mixin owns its own delta dataclass:
  `FeatureDelta`, `MetadataDelta`, or `ObjectDelta`.
- **Cache**: A prefetch cache that holds read-only copies of data from the base backend.
  Reads check the delta first, then the cache, then the base.
- **Three-tier read resolution**:
    1. Delta (local mutations)
    2. Cache (prefetched data)
    3. Base backend (fallback, raises `DetachedStateError` if detached)
- **View constraints**: A `ViewConfig` object that may limit which feature names or
  metadata columns are visible and apply column slices to feature arrays.
- **Detach awareness**: When the overlay is detached (`self._base is None`), reads that
  miss the delta and cache raise `DetachedStateError`.

Module contents
---------------
- `_OverlayLifecycleBase` : Cooperative base class defining terminal lifecycle hooks
  (`_initialize_overlay_context`, `_apply_materialized_deltas`, `_commit_deltas`).
- `OverlayFeatureMixin` : Implements `BaseFeatureMixin` using `FeatureDelta` and
  `FeatureCache`. Handles row indexing translation and column slicing.
- `OverlayMetadataMixin` : Implements `BaseMetadataMixin` using `MetadataDelta` and
  `MetadataCache`. Manages DataFrame-like columnar metadata with support for column
  deletion and partial updates.
- `OverlayObjectMixin` : Implements `BaseObjectMixin` using `ObjectDelta` (sparse map
  from overlay row index to object). Objects are not cached due to potential size or
  unpicklability.

All mixins expect the following attributes to be present on `self`:
- `_base : Optional[BaseStorageBackend]`
- `_index_map : np.ndarray` - mapping from overlay row index to base row index
- `_view_config : ViewConfig` (not used by `OverlayObjectMixin`)
- Methods `_n_rows()`, `_resolve_overlay_idx()`, `_translate()` provided by the
  concrete overlay class.

Lifecycle hooks are called by the overlay container during initialization,
materialisation (copying deltas to a concrete backend), and commit (applying deltas
to the base backend).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from ...indexing import normalize_row_index
from ..base import BaseFeatureMixin, BaseMetadataMixin, BaseObjectMixin, BaseStorageBackend
from .deltas import (
    ColumnSlice,
    FeatureCache,
    FeatureDelta,
    MetadataCache,
    MetadataDelta,
    ObjectDelta,
    ViewConfig,
)
from .identity import DetachedStateError
from .protocol import OverlayContextProtocol

if TYPE_CHECKING:
    from druglab.db.indexing import INDEX_LIKE


# ---------------------------------------------------------------------------
# Lifecycle base (cooperative terminal hooks for overlay chain)
# ---------------------------------------------------------------------------

class _OverlayLifecycleBase:
    """Terminal hooks for the cooperative overlay lifecycle chain."""

    def _initialize_overlay_context(self, **kwargs: Any) -> None:
        return

    def _apply_materialized_deltas(self, concrete_backend: BaseStorageBackend) -> None:
        return

    def _commit_deltas(self, base_backend: BaseStorageBackend) -> None:
        return


# ---------------------------------------------------------------------------
# OverlayFeatureMixin
# ---------------------------------------------------------------------------

class OverlayFeatureMixin(BaseFeatureMixin, _OverlayLifecycleBase, OverlayContextProtocol):
    """
    Feature domain mixin for ``OverlayBackend``.

    State is encapsulated in:
    * ``_feat_delta : FeatureDelta``   - CoW mutations
    * ``_feat_cache : FeatureCache``   - prefetch cache (read-only hits)

    View constraints are checked via ``_view_config.check_feature(name)``
    and column slices are applied before returning data.
    """

    # Declared to satisfy type checker; concrete class provides values.
    _base: Optional[BaseStorageBackend]
    _index_map: np.ndarray
    _view_config: ViewConfig

    # ------------------------------------------------------------------
    # Overlay lifecycle init
    # ------------------------------------------------------------------

    def _initialize_overlay_context(self, **kwargs: Any) -> None:
        self._feat_delta = FeatureDelta()
        self._feat_cache = FeatureCache()
        super()._initialize_overlay_context(**kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _n_feature_rows(self) -> int:
        return self._n_rows() # TODO: Bad dependency

    def _base_required_for_feature(self, name: str) -> BaseStorageBackend:
        """Return ``_base`` or raise ``DetachedStateError``."""
        if self._base is None:
            raise DetachedStateError(
                f"Feature '{name}' is not in the delta or cache, and the "
                "overlay is detached.  Prefetch or re-attach before reading."
            )
        return self._base

    # ------------------------------------------------------------------
    # BaseFeatureMixin API
    # ------------------------------------------------------------------

    def get_feature(self, name: str, idx: Optional["INDEX_LIKE"] = None) -> np.ndarray:
        self._view_config.check_feature(name)

        if self._feat_delta.is_deleted(name):
            raise KeyError(f"Feature '{name}' has been deleted from this overlay.")

        overlay_positions = self._resolve_overlay_idx(idx) # TODO: Bad dependency

        # Tier 1: Delta
        if self._feat_delta.has(name):
            arr = self._feat_delta.get(name)[overlay_positions]
            return self._apply_col_slice(name, arr)

        # Tier 2: Cache
        if self._feat_cache.has(name):
            arr = self._feat_cache.get(name)[overlay_positions]
            return self._apply_col_slice(name, arr)

        # Tier 3: Base
        base = self._base_required_for_feature(name)
        base_positions = self._translate(overlay_positions)
        arr = base.get_feature(name, idx=base_positions)
        return self._apply_col_slice(name, arr)

    def _apply_col_slice(self, name: str, arr: np.ndarray) -> np.ndarray:
        cs = self._view_config.get_col_slice(name)
        if cs is not None:
            return cs.apply(arr)
        return arr

    def update_feature(
        self,
        name: str,
        array: np.ndarray,
        idx: Optional["INDEX_LIKE"] = None,
        na: Any = None,
        **kwargs,
    ) -> None:
        self._view_config.check_feature(name)

        # Column-sliced features are read-only
        if self._view_config.is_col_sliced(name):
            raise RuntimeError(
                f"Feature '{name}' is exposed through a column slice and is "
                "read-only.  Partial-column mutations are not yet supported."
            )

        self._feat_delta.deleted.discard(name)

        n = self._n_rows()
        array = np.asarray(array)
        overlay_positions = self._resolve_overlay_idx(idx)

        if not self._feat_delta.has(name):
            if idx is None:
                if array.shape[0] != n:
                    raise ValueError(
                        f"Array length {array.shape[0]} does not match overlay length {n}."
                    )
                self._feat_delta.set(name, array.copy())
            else:
                # Build a full overlay-length array, filling from base or na
                if (
                    self._base is not None
                    and name in self._base.get_feature_names()
                    and not self._feat_delta.is_deleted(name)
                ):
                    full = self._base.get_feature(name, idx=self._index_map).copy()
                else:
                    shape = (n,) + array.shape[1:]
                    if na is None and np.issubdtype(array.dtype, np.floating):
                        na = np.nan
                    elif na is None:
                        na = 0
                    full = np.full(shape, na, dtype=array.dtype)
                full[overlay_positions] = array
                self._feat_delta.set(name, full)
        else:
            existing = self._feat_delta.get(name)
            if idx is None:
                if array.shape[0] != n:
                    raise ValueError(
                        f"Array length {array.shape[0]} does not match overlay length {n}."
                    )
                self._feat_delta.set(name, array.copy())
            else:
                existing[overlay_positions] = array

        # Invalidate cache for this name
        self._feat_cache.evict(name)

    def drop_feature(self, name: str) -> None:
        self._view_config.check_feature(name)
        base_names = self._base.get_feature_names() if self._base is not None else []
        if name not in base_names and not self._feat_delta.has(name):
            raise KeyError(f"Feature '{name}' does not exist.")
        self._feat_delta.delete(name)
        self._feat_cache.evict(name)

    def get_feature_names(self) -> List[str]:
        base_names: Set[str] = (
            set(self._base.get_feature_names()) if self._base is not None else set()
        )
        visible = (base_names - self._feat_delta.deleted) | set(self._feat_delta.names())
        return self._view_config.apply_feature_filter(list(visible))

    def get_feature_shape(self, name: str) -> tuple:
        self._view_config.check_feature(name)
        if self._feat_delta.is_deleted(name):
            raise KeyError(f"Feature '{name}' has been deleted.")
        if self._feat_delta.has(name):
            raw_shape = self._feat_delta.get(name).shape
        else:
            base = self._base_required_for_feature(name)
            raw_shape = base.get_feature_shape(name)

        cs = self._view_config.get_col_slice(name)
        if cs is not None:
            col_width = cs.stop - cs.start
            return (self._n_rows(),) + (col_width,) + raw_shape[2:]
        return (self._n_rows(),) + raw_shape[1:]

    def _validate_features(self) -> None:
        return  # Overlay skips cross-dimension validation

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def _apply_materialized_deltas(self, concrete_backend: BaseStorageBackend) -> None:
        for name in self._feat_delta.deleted:
            if name in concrete_backend.get_feature_names():
                concrete_backend.drop_feature(name)

        for name in self._feat_delta.names():
            if name not in self._feat_delta.deleted:
                concrete_backend.update_feature(name, self._feat_delta.get(name))

        super()._apply_materialized_deltas(concrete_backend)

    # ------------------------------------------------------------------
    # Commit  (delta only – cache is ignored)
    # ------------------------------------------------------------------

    def _commit_deltas(self, base_backend: BaseStorageBackend) -> None:
        for name, arr in self._feat_delta.local.items():
            base_backend.update_feature(name, arr, idx=self._index_map)

        for name in self._feat_delta.deleted:
            if name in base_backend.get_feature_names():
                base_backend.drop_feature(name)

        self._feat_delta.clear()
        # Cache is intentionally NOT cleared on commit – it remains valid.
        super()._commit_deltas(base_backend)

    # ------------------------------------------------------------------
    # Prefetch
    # ------------------------------------------------------------------

    def _prefetch_features(
        self,
        names: List[str],
        overlay_positions: Optional[np.ndarray] = None,
    ) -> None:
        """Load feature arrays from base into the cache for the given names."""
        if self._base is None:
            raise DetachedStateError(
                "Cannot prefetch features: overlay is detached."
            )
        if overlay_positions is None:
            overlay_positions = np.arange(self._n_rows(), dtype=np.intp)

        base_positions = self._translate(overlay_positions)

        for name in names:
            if self._feat_delta.has(name) or self._feat_delta.is_deleted(name):
                # Delta takes precedence; no need to cache.
                continue
            arr = self._base.get_feature(name, idx=base_positions)
            # Store a *full* overlay-length cache entry (row-ordered)
            if overlay_positions is not None and len(overlay_positions) < self._n_rows():
                # Partial prefetch: only cache what was requested
                full = self._feat_cache.get(name) if self._feat_cache.has(name) else np.empty(0)
                if full.shape[0] != self._n_rows():
                    # Allocate fresh
                    base_full = self._base.get_feature(name, idx=self._index_map)
                    self._feat_cache.put(name, base_full)
                else:
                    full[overlay_positions] = arr
            else:
                self._feat_cache.put(name, arr)


# ---------------------------------------------------------------------------
# OverlayMetadataMixin
# ---------------------------------------------------------------------------

class OverlayMetadataMixin(BaseMetadataMixin, _OverlayLifecycleBase, OverlayContextProtocol):
    """
    Metadata domain mixin for ``OverlayBackend``.

    State is encapsulated in:
    * ``_meta_delta : MetadataDelta``   - CoW mutations
    * ``_meta_cache : MetadataCache``   - prefetch cache
    """

    _base: Optional[BaseStorageBackend]
    _index_map: np.ndarray
    _view_config: ViewConfig

    # ------------------------------------------------------------------

    def _initialize_overlay_context(self, **kwargs: Any) -> None:
        self._meta_delta = MetadataDelta()
        self._meta_cache = MetadataCache()
        super()._initialize_overlay_context(**kwargs)

    # ------------------------------------------------------------------

    def _n_metadata_rows(self) -> int:
        return self._n_rows()

    def _base_required_for_meta(self) -> BaseStorageBackend:
        if self._base is None:
            raise DetachedStateError(
                "Metadata read missed the delta and cache, and the overlay is "
                "detached.  Prefetch or re-attach before reading."
            )
        return self._base

    # ------------------------------------------------------------------
    # BaseMetadataMixin API
    # ------------------------------------------------------------------

    def get_metadata(
        self,
        idx: Optional["INDEX_LIKE"] = None,
        cols: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        overlay_positions = self._resolve_overlay_idx(idx) # TODO: Bad dependency
        base_positions = self._translate(overlay_positions) # TODO: Bad dependency

        # Tier 3: Base (may raise DetachedStateError if detached)
        base = self._base_required_for_meta()

        base_df = base.get_metadata(idx=base_positions).reset_index(drop=True)

        # Strip deleted columns from base result
        drop_from_base = [
            c for c in self._meta_delta.deleted_cols if c in base_df.columns
        ]
        if drop_from_base:
            base_df = base_df.drop(columns=drop_from_base)

        # Overlay local delta columns
        if self._meta_delta.local is not None:
            local_slice = (
                self._meta_delta.local.iloc[overlay_positions].reset_index(drop=True)
            )
            for col in local_slice.columns:
                if col not in self._meta_delta.deleted_cols:
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
        **kwargs,
    ) -> None:
        self._view_config.check_meta_col(name)
        self._meta_delta.deleted_cols.discard(name)

        value = np.asarray(value)
        n = self._n_rows()
        self._meta_delta.ensure_local(n)

        overlay_positions = self._resolve_overlay_idx(idx)
        if idx is None:
            self._meta_delta.local[name] = value
        else:
            if name not in self._meta_delta.local.columns:
                if na is None and np.issubdtype(value.dtype, np.floating):
                    na = np.nan
                elif na is None:
                    na = 0
                self._meta_delta.local[name] = np.full(n, na, dtype=value.dtype)
            col_pos = self._meta_delta.local.columns.get_loc(name)
            self._meta_delta.local.iloc[overlay_positions, col_pos] = value

        # Invalidate cache
        self._meta_cache.evict_col(name)

    def update_metadata(
        self,
        values: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
        idx: Optional["INDEX_LIKE"] = None,
        **kwargs,
    ) -> None:
        n = self._n_rows()

        if isinstance(values, pd.DataFrame):
            val_dict = {col: values[col].values for col in values.columns}
        elif isinstance(values, pd.Series):
            if values.name is None:
                raise ValueError("Series must have a name.")
            val_dict = {values.name: values.values}
        else:
            val_dict = dict(values)

        # Validate all columns exist
        all_visible = self._visible_metadata_columns()
        for col in val_dict:
            self._view_config.check_meta_col(col)
            if col not in all_visible:
                raise KeyError(
                    f"Column '{col}' does not exist. Use add_metadata_column."
                )

        overlay_positions = self._resolve_overlay_idx(idx)
        self._meta_delta.ensure_local(n)

        for col, val in val_dict.items():
            if col not in self._meta_delta.local.columns:
                # Copy from base so we have a full local column
                base = self._base_required_for_meta()
                base_col = base.get_metadata(
                    idx=self._index_map, cols=[col]
                ).reset_index(drop=True)[col].values
                self._meta_delta.local[col] = base_col

            if idx is None:
                self._meta_delta.local[col] = val
            else:
                col_pos = self._meta_delta.local.columns.get_loc(col)
                self._meta_delta.local.iloc[overlay_positions, col_pos] = val

            self._meta_cache.evict_col(col)

    def drop_metadata_columns(
        self, cols: Optional[Union[str, List[str]]] = None
    ) -> None:
        if cols is None:
            visible = self._visible_metadata_columns()
            self._meta_delta.drop_all_visible(visible)
            self._meta_cache.clear()
            return

        if isinstance(cols, str):
            cols = [cols]
        for col in cols:
            self._meta_delta.drop_col(col)
            self._meta_cache.evict_col(col)

    def _visible_metadata_columns(self) -> List[str]:
        base_cols: Set[str] = (
            set(self._base.get_metadata_columns())
            if self._base is not None
            else set()
        )
        local_cols: Set[str] = (
            set(self._meta_delta.local.columns)
            if self._meta_delta.local is not None
            else set()
        )
        visible = (base_cols - self._meta_delta.deleted_cols) | local_cols
        return self._view_config.apply_meta_col_filter(list(visible))

    def get_metadata_columns(self) -> List[str]:
        return self._visible_metadata_columns()

    def _validate_metadata(self) -> None:
        return

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def _apply_materialized_deltas(self, concrete_backend: BaseStorageBackend) -> None:
        if self._meta_delta.deleted_cols:
            existing_cols = set(concrete_backend.get_metadata().columns)
            to_drop = list(self._meta_delta.deleted_cols & existing_cols)
            if to_drop:
                concrete_backend.drop_metadata_columns(to_drop)

        if self._meta_delta.local is not None and not self._meta_delta.local.empty:
            existing_cols = set(concrete_backend.get_metadata().columns)
            for col in self._meta_delta.local.columns:
                if col in self._meta_delta.deleted_cols:
                    continue
                val = self._meta_delta.local[col].values
                if col in existing_cols:
                    concrete_backend.update_metadata({col: val})
                else:
                    concrete_backend.add_metadata_column(col, val)

        super()._apply_materialized_deltas(concrete_backend)

    # ------------------------------------------------------------------
    # Commit  (delta only – cache is ignored)
    # ------------------------------------------------------------------

    def _commit_deltas(self, base_backend: BaseStorageBackend) -> None:
        if self._meta_delta.local is not None and not self._meta_delta.local.empty:
            base_cols = set(base_backend.get_metadata().columns)
            for col in self._meta_delta.local.columns:
                val = self._meta_delta.local[col].values
                if col in base_cols:
                    base_backend.update_metadata({col: val}, idx=self._index_map)
                else:
                    base_backend.add_metadata_column(col, val, idx=self._index_map)

        if self._meta_delta.deleted_cols:
            existing_base_cols = set(base_backend.get_metadata().columns)
            cols_to_drop = list(
                self._meta_delta.deleted_cols & existing_base_cols
            )
            if cols_to_drop:
                base_backend.drop_metadata_columns(cols_to_drop)

        self._meta_delta.clear()
        # Cache intentionally NOT cleared on commit.
        super()._commit_deltas(base_backend)

    # ------------------------------------------------------------------
    # Prefetch
    # ------------------------------------------------------------------

    def _prefetch_metadata(
        self,
        cols: List[str],
        overlay_positions: Optional[np.ndarray] = None,
    ) -> None:
        if self._base is None:
            raise DetachedStateError(
                "Cannot prefetch metadata: overlay is detached."
            )
        if overlay_positions is None:
            base_positions = self._index_map
        else:
            base_positions = self._translate(overlay_positions)

        df = self._base.get_metadata(idx=base_positions, cols=cols).reset_index(drop=True)
        self._meta_cache.put(df)


# ---------------------------------------------------------------------------
# OverlayObjectMixin
# ---------------------------------------------------------------------------

class OverlayObjectMixin(BaseObjectMixin, _OverlayLifecycleBase):
    """
    Object domain mixin for ``OverlayBackend``.

    State is encapsulated in ``_obj_delta : ObjectDelta`` (sparse CoW map).
    Objects are not cached (they are typically large/unpicklable); detach
    awareness is enforced at read time.
    """

    _base: Optional[BaseStorageBackend]
    _index_map: np.ndarray

    # ------------------------------------------------------------------

    def _initialize_overlay_context(self, **kwargs: Any) -> None:
        self._obj_delta = ObjectDelta()
        super()._initialize_overlay_context(**kwargs)

    # ------------------------------------------------------------------

    def _n_objects(self) -> int:
        return self._n_rows()

    def _base_required_for_object(self, overlay_idx: int) -> BaseStorageBackend:
        if self._base is None:
            raise DetachedStateError(
                f"Object at overlay index {overlay_idx} is not in the delta, "
                "and the overlay is detached.  Re-attach before reading."
            )
        return self._base

    # ------------------------------------------------------------------
    # BaseObjectMixin API
    # ------------------------------------------------------------------

    def get_objects(self, idx: Optional["INDEX_LIKE"] = None) -> Union[Any, List[Any]]:
        scalar = isinstance(idx, (int, np.integer))
        overlay_positions = self._resolve_overlay_idx(idx)

        results = []
        for oi in overlay_positions:
            oi_int = int(oi)
            if self._obj_delta.has(oi_int):
                results.append(self._obj_delta.get(oi_int))
            else:
                base = self._base_required_for_object(oi_int)
                base_pos = int(self._index_map[oi_int])
                results.append(base.get_objects(idx=base_pos))

        if scalar:
            return results[0] if results else None
        return results

    def update_objects(
        self,
        objs: Union[Any, List[Any]],
        idx: Optional["INDEX_LIKE"] = None,
        **kwargs,
    ) -> None:
        overlay_positions = self._resolve_overlay_idx(idx)

        if idx is None:
            if len(objs) != len(overlay_positions):
                raise ValueError("Length of objs must match overlay length.")
            for oi, obj in zip(overlay_positions, objs):
                self._obj_delta.set(int(oi), obj)
        elif isinstance(idx, (int, np.integer)):
            self._obj_delta.set(int(overlay_positions[0]), objs)
        else:
            if len(objs) != len(overlay_positions):
                raise ValueError("Length of objs must match index length.")
            for oi, obj in zip(overlay_positions, objs):
                self._obj_delta.set(int(oi), obj)

    def _validate_objects(self) -> None:
        return

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def _apply_materialized_deltas(self, concrete_backend: BaseStorageBackend) -> None:
        for oi, obj in self._obj_delta.local.items():
            concrete_backend.update_objects(obj, idx=int(oi))
        super()._apply_materialized_deltas(concrete_backend)

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def _commit_deltas(self, base_backend: BaseStorageBackend) -> None:
        for overlay_idx, obj in self._obj_delta.local.items():
            base_idx = int(self._index_map[overlay_idx])
            base_backend.update_objects(obj, idx=base_idx)
        self._obj_delta.clear()
        super()._commit_deltas(base_backend)