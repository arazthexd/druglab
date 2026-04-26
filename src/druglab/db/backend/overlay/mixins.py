"""Overlay domain stores for composition-based OverlayBackend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from ..base import BaseStorageBackend
from .deltas import FeatureCache, FeatureDelta, MetadataCache, MetadataDelta, ObjectDelta, ViewConfig
from .identity import DetachedStateError
from .protocol import OverlayContextProtocol

if TYPE_CHECKING:
    from druglab.db.indexing import INDEX_LIKE


class OverlayFeatureStore:
    def __init__(self, context: OverlayContextProtocol) -> None:
        self._ctx = context
        self._delta = FeatureDelta()
        self._cache = FeatureCache()

    @property
    def delta(self) -> FeatureDelta:
        return self._delta

    @property
    def cache(self) -> FeatureCache:
        return self._cache

    def _base_required(self, name: str) -> BaseStorageBackend:
        if self._ctx._base is None:
            raise DetachedStateError(
                f"Feature '{name}' is not in the delta or cache, and the overlay is detached."
            )
        return self._ctx._base

    def get_feature(self, name: str, idx: Optional["INDEX_LIKE"] = None) -> np.ndarray:
        self._ctx._view_config.check_feature(name)
        if self._delta.is_deleted(name):
            raise KeyError(f"Feature '{name}' has been deleted from this overlay.")
        overlay_positions = self._ctx._resolve_overlay_idx(idx)

        if self._delta.has(name):
            arr = self._delta.get(name)[overlay_positions]
            return self._apply_col_slice(name, arr)
        if self._cache.has(name):
            arr = self._cache.get(name)[overlay_positions]
            return self._apply_col_slice(name, arr)

        base_positions = self._ctx._translate(overlay_positions)
        arr = self._base_required(name).get_feature(name, idx=base_positions)
        return self._apply_col_slice(name, arr)

    def _apply_col_slice(self, name: str, arr: np.ndarray) -> np.ndarray:
        cs = self._ctx._view_config.get_col_slice(name)
        return cs.apply(arr) if cs is not None else arr

    def update_feature(self, name: str, array: np.ndarray, idx: Optional["INDEX_LIKE"] = None, na: Any = None) -> None:
        self._ctx._view_config.check_feature(name)
        if self._ctx._view_config.is_col_sliced(name):
            raise RuntimeError(f"Feature '{name}' is read-only due to column slicing.")
        self._delta.deleted.discard(name)

        n = self._ctx._n_rows()
        array = np.asarray(array)
        overlay_positions = self._ctx._resolve_overlay_idx(idx)

        if not self._delta.has(name):
            if idx is None:
                if array.shape[0] != n:
                    raise ValueError(f"Array length {array.shape[0]} does not match overlay length {n}.")
                self._delta.set(name, array.copy())
            else:
                if self._ctx._base is not None and name in self._ctx._base.get_feature_names() and not self._delta.is_deleted(name):
                    full = self._ctx._base.get_feature(name, idx=self._ctx._index_map).copy()
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
                    raise ValueError(f"Array length {array.shape[0]} does not match overlay length {n}.")
                self._delta.set(name, array.copy())
            else:
                existing[overlay_positions] = array

        self._cache.evict(name)

    def drop_feature(self, name: str) -> None:
        self._ctx._view_config.check_feature(name)
        base_names = self._ctx._base.get_feature_names() if self._ctx._base is not None else []
        if name not in base_names and not self._delta.has(name):
            raise KeyError(f"Feature '{name}' does not exist.")
        self._delta.delete(name)
        self._cache.evict(name)

    def get_feature_names(self) -> List[str]:
        base_names: Set[str] = set(self._ctx._base.get_feature_names()) if self._ctx._base is not None else set()
        visible = (base_names - self._delta.deleted) | set(self._delta.names())
        return self._ctx._view_config.apply_feature_filter(list(visible))

    def get_feature_shape(self, name: str) -> tuple:
        self._ctx._view_config.check_feature(name)
        if self._delta.is_deleted(name):
            raise KeyError(f"Feature '{name}' has been deleted.")
        raw_shape = self._delta.get(name).shape if self._delta.has(name) else self._base_required(name).get_feature_shape(name)
        cs = self._ctx._view_config.get_col_slice(name)
        if cs is not None:
            return (self._ctx._n_rows(), cs.stop - cs.start) + raw_shape[2:]
        return (self._ctx._n_rows(),) + raw_shape[1:]

    def prefetch(self, names: List[str], overlay_positions: Optional[np.ndarray] = None) -> None:
        if self._ctx._base is None:
            raise DetachedStateError("Cannot prefetch features: overlay is detached.")
        if overlay_positions is None:
            overlay_positions = np.arange(self._ctx._n_rows(), dtype=np.intp)
        base_positions = self._ctx._translate(overlay_positions)
        for name in names:
            if self._delta.has(name) or self._delta.is_deleted(name):
                continue
            self._cache.put(name, self._ctx._base.get_feature(name, idx=base_positions))

    def apply_materialized_deltas(self, concrete_backend: BaseStorageBackend) -> None:
        for name in self._delta.deleted:
            if name in concrete_backend.get_feature_names():
                concrete_backend.drop_feature(name)
        for name in self._delta.names():
            if name not in self._delta.deleted:
                concrete_backend.update_feature(name, self._delta.get(name))

    def commit(self, base_backend: BaseStorageBackend, index_map: np.ndarray) -> None:
        for name, arr in self._delta.local.items():
            base_backend.update_feature(name, arr, idx=index_map)
        for name in self._delta.deleted:
            if name in base_backend.get_feature_names():
                base_backend.drop_feature(name)
        self._delta.clear()


class OverlayMetadataStore:
    def __init__(self, context: OverlayContextProtocol) -> None:
        self._ctx = context
        self._delta = MetadataDelta()
        self._cache = MetadataCache()

    @property
    def delta(self) -> MetadataDelta:
        return self._delta

    @property
    def cache(self) -> MetadataCache:
        return self._cache

    def _base_required(self) -> BaseStorageBackend:
        if self._ctx._base is None:
            raise DetachedStateError("Metadata read missed delta/cache while detached.")
        return self._ctx._base

    def get_metadata(self, idx: Optional["INDEX_LIKE"] = None, cols: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        overlay_positions = self._ctx._resolve_overlay_idx(idx)
        base_df = self._base_required().get_metadata(idx=self._ctx._translate(overlay_positions)).reset_index(drop=True)

        drop_from_base = [c for c in self._delta.deleted_cols if c in base_df.columns]
        if drop_from_base:
            base_df = base_df.drop(columns=drop_from_base)

        if self._delta.local is not None:
            local_slice = self._delta.local.iloc[overlay_positions].reset_index(drop=True)
            for col in local_slice.columns:
                if col not in self._delta.deleted_cols:
                    base_df[col] = local_slice[col].values

        if self._ctx._view_config.allowed_meta_cols is not None:
            visible = [c for c in base_df.columns if c in self._ctx._view_config.allowed_meta_cols]
            base_df = base_df[visible]

        if cols is not None:
            if isinstance(cols, str):
                cols = [cols]
            for c in cols:
                self._ctx._view_config.check_meta_col(c)
            base_df = base_df[cols]

        return base_df.reset_index(drop=True)

    def _visible_metadata_columns(self) -> List[str]:
        base_cols: Set[str] = set(self._ctx._base.get_metadata_columns()) if self._ctx._base is not None else set()
        local_cols: Set[str] = set(self._delta.local.columns) if self._delta.local is not None else set()
        return self._ctx._view_config.apply_meta_col_filter(list((base_cols - self._delta.deleted_cols) | local_cols))

    def get_metadata_columns(self) -> List[str]:
        return self._visible_metadata_columns()

    def add_metadata_column(self, name: str, value: Union[pd.Series, np.ndarray, List[Any]], idx: Optional["INDEX_LIKE"] = None, na: Any = None) -> None:
        self._ctx._view_config.check_meta_col(name)
        self._delta.deleted_cols.discard(name)
        value = np.asarray(value)
        n = self._ctx._n_rows()
        self._delta.ensure_local(n)
        overlay_positions = self._ctx._resolve_overlay_idx(idx)
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

    def update_metadata(self, values: Union[pd.DataFrame, pd.Series, Dict[str, Any]], idx: Optional["INDEX_LIKE"] = None) -> None:
        if isinstance(values, pd.DataFrame):
            val_dict = {col: values[col].values for col in values.columns}
        elif isinstance(values, pd.Series):
            if values.name is None:
                raise ValueError("Series must have a name.")
            val_dict = {values.name: values.values}
        else:
            val_dict = dict(values)

        all_visible = self._visible_metadata_columns()
        for col in val_dict:
            self._ctx._view_config.check_meta_col(col)
            if col not in all_visible:
                raise KeyError(f"Column '{col}' does not exist. Use add_metadata_column.")

        n = self._ctx._n_rows()
        overlay_positions = self._ctx._resolve_overlay_idx(idx)
        self._delta.ensure_local(n)
        for col, val in val_dict.items():
            if col not in self._delta.local.columns:
                base_col = self._base_required().get_metadata(idx=self._ctx._index_map, cols=[col]).reset_index(drop=True)[col].values
                self._delta.local[col] = base_col
            if idx is None:
                self._delta.local[col] = val
            else:
                col_pos = self._delta.local.columns.get_loc(col)
                self._delta.local.iloc[overlay_positions, col_pos] = val
            self._cache.evict_col(col)

    def drop_metadata_columns(self, cols: Optional[Union[str, List[str]]] = None) -> None:
        if cols is None:
            self._delta.drop_all_visible(self._visible_metadata_columns())
            self._cache.clear()
            return
        if isinstance(cols, str):
            cols = [cols]
        for col in cols:
            self._delta.drop_col(col)
            self._cache.evict_col(col)

    def prefetch(self, cols: List[str], overlay_positions: Optional[np.ndarray] = None) -> None:
        if self._ctx._base is None:
            raise DetachedStateError("Cannot prefetch metadata: overlay is detached.")
        base_positions = self._ctx._index_map if overlay_positions is None else self._ctx._translate(overlay_positions)
        self._cache.put(self._ctx._base.get_metadata(idx=base_positions, cols=cols).reset_index(drop=True))

    def apply_materialized_deltas(self, concrete_backend: BaseStorageBackend) -> None:
        if self._delta.deleted_cols:
            existing_cols = set(concrete_backend.get_metadata().columns)
            to_drop = list(self._delta.deleted_cols & existing_cols)
            if to_drop:
                concrete_backend.drop_metadata_columns(to_drop)
        if self._delta.local is not None and not self._delta.local.empty:
            existing_cols = set(concrete_backend.get_metadata().columns)
            for col in self._delta.local.columns:
                if col in self._delta.deleted_cols:
                    continue
                val = self._delta.local[col].values
                if col in existing_cols:
                    concrete_backend.update_metadata({col: val})
                else:
                    concrete_backend.add_metadata_column(col, val)

    def commit(self, base_backend: BaseStorageBackend, index_map: np.ndarray) -> None:
        if self._delta.local is not None and not self._delta.local.empty:
            base_cols = set(base_backend.get_metadata().columns)
            for col in self._delta.local.columns:
                val = self._delta.local[col].values
                if col in base_cols:
                    base_backend.update_metadata({col: val}, idx=index_map)
                else:
                    base_backend.add_metadata_column(col, val, idx=index_map)
        if self._delta.deleted_cols:
            existing = set(base_backend.get_metadata().columns)
            cols_to_drop = list(self._delta.deleted_cols & existing)
            if cols_to_drop:
                base_backend.drop_metadata_columns(cols_to_drop)
        self._delta.clear()


class OverlayObjectStore:
    def __init__(self, context: OverlayContextProtocol) -> None:
        self._ctx = context
        self._delta = ObjectDelta()

    @property
    def delta(self) -> ObjectDelta:
        return self._delta

    def _base_required(self, overlay_idx: int) -> BaseStorageBackend:
        if self._ctx._base is None:
            raise DetachedStateError(f"Object at overlay index {overlay_idx} is not in the delta and overlay is detached.")
        return self._ctx._base

    def get_objects(self, idx: Optional["INDEX_LIKE"] = None):
        scalar = isinstance(idx, (int, np.integer))
        overlay_positions = self._ctx._resolve_overlay_idx(idx)
        results = []
        for oi in overlay_positions:
            oi_int = int(oi)
            if self._delta.has(oi_int):
                results.append(self._delta.get(oi_int))
            else:
                base_pos = int(self._ctx._index_map[oi_int])
                results.append(self._base_required(oi_int).get_objects(idx=base_pos))
        return results[0] if scalar else results

    def update_objects(self, objs, idx: Optional["INDEX_LIKE"] = None) -> None:
        overlay_positions = self._ctx._resolve_overlay_idx(idx)
        if idx is None:
            if len(objs) != len(overlay_positions):
                raise ValueError("Length of objs must match overlay length.")
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

    def apply_materialized_deltas(self, concrete_backend: BaseStorageBackend) -> None:
        for oi, obj in self._delta.local.items():
            concrete_backend.update_objects(obj, idx=int(oi))

    def commit(self, base_backend: BaseStorageBackend, index_map: np.ndarray) -> None:
        for overlay_idx, obj in self._delta.local.items():
            base_backend.update_objects(obj, idx=int(index_map[overlay_idx]))
        self._delta.clear()