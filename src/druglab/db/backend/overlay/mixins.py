
"""Overlay domain mixins with cooperative lifecycle hooks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from ...indexing import normalize_row_index
from ..base import BaseFeatureMixin, BaseMetadataMixin, BaseObjectMixin, BaseStorageBackend

if TYPE_CHECKING:
    from druglab.db.indexing import INDEX_LIKE


class _OverlayLifecycleBase:
    """Terminal hooks for cooperative overlay lifecycle methods."""

    def _initialize_overlay_context(self, **kwargs: Any) -> None:
        return

    def _apply_materialized_deltas(self, concrete_backend: BaseStorageBackend) -> None:
        return

    def _commit_deltas(self, base_backend: BaseStorageBackend) -> None:
        return


class OverlayFeatureMixin(BaseFeatureMixin, _OverlayLifecycleBase):
    def _initialize_overlay_context(self, **kwargs: Any) -> None:
        self._local_features: Dict[str, np.ndarray] = {}
        self._deleted_features: Set[str] = set()
        super()._initialize_overlay_context(**kwargs)

    def _n_feature_rows(self) -> int:
        return self._n_rows()

    def get_feature(self, name: str, idx: Optional[INDEX_LIKE] = None) -> np.ndarray:
        if name in self._deleted_features:
            raise KeyError(f"Feature '{name}' has been deleted.")

        overlay_positions = self._resolve_overlay_idx(idx)
        if name in self._local_features:
            return self._local_features[name][overlay_positions]

        base_positions = self._translate(overlay_positions)
        return self._base.get_feature(name, idx=base_positions)

    def update_feature(
        self,
        name: str,
        array: np.ndarray,
        idx: Optional[INDEX_LIKE] = None,
        na: Any = None,
        **kwargs,
    ) -> None:
        self._deleted_features.discard(name)

        n = self._n_rows()
        array = np.asarray(array)
        overlay_positions = self._resolve_overlay_idx(idx)

        if name not in self._local_features:
            if idx is None:
                if array.shape[0] != n:
                    raise ValueError(f"Array length {array.shape[0]} does not match overlay length {n}.")
                self._local_features[name] = array.copy()
            else:
                if name in self._base.get_feature_names() and name not in self._deleted_features:
                    full = self._base.get_feature(name, idx=self._index_map).copy()
                else:
                    shape = (n,) + array.shape[1:]
                    if na is None and np.issubdtype(array.dtype, np.floating):
                        na = np.nan
                    elif na is None:
                        na = 0
                    full = np.full(shape, na, dtype=array.dtype)
                full[overlay_positions] = array
                self._local_features[name] = full
        else:
            if idx is None:
                if array.shape[0] != n:
                    raise ValueError(f"Array length {array.shape[0]} does not match overlay length {n}.")
                self._local_features[name] = array.copy()
            else:
                self._local_features[name][overlay_positions] = array

    def drop_feature(self, name: str) -> None:
        base_names = self._base.get_feature_names()
        if name not in base_names and name not in self._local_features:
            raise KeyError(f"Feature '{name}' does not exist.")
        self._deleted_features.add(name)
        self._local_features.pop(name, None)

    def get_feature_names(self) -> List[str]:
        base_names = set(self._base.get_feature_names())
        return list((base_names - self._deleted_features) | set(self._local_features.keys()))

    def get_feature_shape(self, name: str) -> tuple:
        if name in self._deleted_features:
            raise KeyError(f"Feature '{name}' has been deleted.")
        if name in self._local_features:
            return self._local_features[name].shape
        base_shape = self._base.get_feature_shape(name)
        return (self._n_rows(),) + base_shape[1:]

    def _validate_features(self) -> None:
        return

    def _apply_materialized_deltas(self, concrete_backend: BaseStorageBackend) -> None:
        for name in self._deleted_features:
            if name in concrete_backend.get_feature_names():
                concrete_backend.drop_feature(name)

        for name, arr in self._local_features.items():
            if name not in self._deleted_features:
                concrete_backend.update_feature(name, arr)

        super()._apply_materialized_deltas(concrete_backend)

    def _commit_deltas(self, base_backend: BaseStorageBackend) -> None:
        for name, arr in self._local_features.items():
            base_backend.update_feature(name, arr, idx=self._index_map)

        for name in self._deleted_features:
            if name in base_backend.get_feature_names():
                base_backend.drop_feature(name)

        self._local_features.clear()
        self._deleted_features.clear()
        super()._commit_deltas(base_backend)


class OverlayMetadataMixin(BaseMetadataMixin, _OverlayLifecycleBase):
    def _initialize_overlay_context(self, **kwargs: Any) -> None:
        self._local_metadata: Optional[pd.DataFrame] = None
        self._deleted_metadata_cols: Set[str] = set()
        super()._initialize_overlay_context(**kwargs)

    def _n_metadata_rows(self) -> int:
        return self._n_rows()

    def get_metadata(
        self,
        idx: Optional[INDEX_LIKE] = None,
        cols: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        overlay_positions = self._resolve_overlay_idx(idx)
        base_positions = self._translate(overlay_positions)

        base_cols = self._base.get_metadata(idx=base_positions).reset_index(drop=True)
        drop_cols = [c for c in self._deleted_metadata_cols if c in base_cols.columns]
        if drop_cols:
            base_cols = base_cols.drop(columns=drop_cols)

        if self._local_metadata is not None:
            local_slice = self._local_metadata.iloc[overlay_positions].reset_index(drop=True)
            for col in local_slice.columns:
                if col not in self._deleted_metadata_cols:
                    base_cols[col] = local_slice[col].values

        if cols is not None:
            if isinstance(cols, str):
                cols = [cols]
            base_cols = base_cols[cols]

        return base_cols.reset_index(drop=True)

    def add_metadata_column(
        self,
        name: str,
        value: Union[pd.Series, np.ndarray, List[Any]],
        idx: Optional[INDEX_LIKE] = None,
        na: Any = None,
        **kwargs,
    ) -> None:
        self._deleted_metadata_cols.discard(name)

        value = np.asarray(value)
        n = self._n_rows()

        if self._local_metadata is None:
            self._local_metadata = pd.DataFrame(index=range(n))

        overlay_positions = self._resolve_overlay_idx(idx)
        if idx is None:
            self._local_metadata[name] = value
        else:
            if name not in self._local_metadata.columns:
                if na is None and np.issubdtype(value.dtype, np.floating):
                    na = np.nan
                elif na is None:
                    na = 0
                self._local_metadata[name] = np.full(n, na, dtype=value.dtype)
            col_idx = self._local_metadata.columns.get_loc(name)
            self._local_metadata.iloc[overlay_positions, col_idx] = value

    def update_metadata(
        self,
        values: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
        idx: Optional[INDEX_LIKE] = None,
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

        overlay_positions = self._resolve_overlay_idx(idx)
        all_visible_cols = self._visible_metadata_columns()
        for col in val_dict:
            if col not in all_visible_cols:
                raise KeyError(f"Column '{col}' does not exist. Use add_metadata_column.")

        if self._local_metadata is None:
            self._local_metadata = pd.DataFrame(index=range(n))

        for col, val in val_dict.items():
            if col not in self._local_metadata.columns:
                base_all = self._base.get_metadata(idx=self._index_map, cols=[col]).reset_index(drop=True)
                self._local_metadata[col] = base_all[col].values

            if idx is None:
                self._local_metadata[col] = val
            else:
                col_idx = self._local_metadata.columns.get_loc(col)
                self._local_metadata.iloc[overlay_positions, col_idx] = val

    def drop_metadata_columns(self, cols: Optional[Union[str, List[str]]] = None) -> None:
        if cols is None:
            visible = self._visible_metadata_columns()
            self._deleted_metadata_cols.update(visible)
            if self._local_metadata is not None:
                self._local_metadata = pd.DataFrame(index=self._local_metadata.index)
            return

        if isinstance(cols, str):
            cols = [cols]
        for col in cols:
            self._deleted_metadata_cols.add(col)
            if self._local_metadata is not None and col in self._local_metadata.columns:
                self._local_metadata.drop(columns=[col], inplace=True)

    def _visible_metadata_columns(self) -> List[str]:
        base_cols = set(self._base.get_metadata().columns) - self._deleted_metadata_cols
        local_cols = set(self._local_metadata.columns) if self._local_metadata is not None else set()
        return list(base_cols | local_cols)

    def _validate_metadata(self) -> None:
        return

    def _apply_materialized_deltas(self, concrete_backend: BaseStorageBackend) -> None:
        if self._deleted_metadata_cols:
            existing_cols = set(concrete_backend.get_metadata().columns)
            to_drop = list(self._deleted_metadata_cols & existing_cols)
            if to_drop:
                concrete_backend.drop_metadata_columns(to_drop)

        if self._local_metadata is not None and not self._local_metadata.empty:
            existing_cols = set(concrete_backend.get_metadata().columns)
            for col in self._local_metadata.columns:
                if col in self._deleted_metadata_cols:
                    continue
                val = self._local_metadata[col].values
                if col in existing_cols:
                    concrete_backend.update_metadata({col: val})
                else:
                    concrete_backend.add_metadata_column(col, val)

        super()._apply_materialized_deltas(concrete_backend)

    def _commit_deltas(self, base_backend: BaseStorageBackend) -> None:
        if self._local_metadata is not None and not self._local_metadata.empty:
            base_cols = set(base_backend.get_metadata().columns)
            for col in self._local_metadata.columns:
                val = self._local_metadata[col].values
                if col in base_cols:
                    base_backend.update_metadata({col: val}, idx=self._index_map)
                else:
                    base_backend.add_metadata_column(col, val, idx=self._index_map)

        if self._deleted_metadata_cols:
            existing_base_cols = set(base_backend.get_metadata().columns)
            cols_to_drop = list(self._deleted_metadata_cols & existing_base_cols)
            if cols_to_drop:
                base_backend.drop_metadata_columns(cols_to_drop)

        self._local_metadata = None
        self._deleted_metadata_cols.clear()
        super()._commit_deltas(base_backend)


class OverlayObjectMixin(BaseObjectMixin, _OverlayLifecycleBase):
    def _initialize_overlay_context(self, **kwargs: Any) -> None:
        self._local_objects: Dict[int, Any] = {}
        super()._initialize_overlay_context(**kwargs)

    def _n_objects(self) -> int:
        return self._n_rows()

    def get_objects(self, idx: Optional[INDEX_LIKE] = None) -> Union[Any, List[Any]]:
        scalar = isinstance(idx, (int, np.integer))
        overlay_positions = self._resolve_overlay_idx(idx)

        results = []
        for oi in overlay_positions:
            if oi in self._local_objects:
                results.append(self._local_objects[oi])
            else:
                base_pos = int(self._index_map[oi])
                results.append(self._base.get_objects(idx=base_pos))

        if scalar:
            return results[0] if results else None
        return results

    def update_objects(
        self,
        objs: Union[Any, List[Any]],
        idx: Optional[INDEX_LIKE] = None,
        **kwargs,
    ) -> None:
        overlay_positions = self._resolve_overlay_idx(idx)

        if idx is None:
            if len(objs) != len(overlay_positions):
                raise ValueError("Length of objs must match overlay length.")
            for oi, obj in zip(overlay_positions, objs):
                self._local_objects[int(oi)] = obj
        elif isinstance(idx, (int, np.integer)):
            self._local_objects[int(overlay_positions[0])] = objs
        else:
            if len(objs) != len(overlay_positions):
                raise ValueError("Length of objs must match index length.")
            for oi, obj in zip(overlay_positions, objs):
                self._local_objects[int(oi)] = obj

    def _validate_objects(self) -> None:
        return

    def _apply_materialized_deltas(self, concrete_backend: BaseStorageBackend) -> None:
        for oi, obj in self._local_objects.items():
            concrete_backend.update_objects(obj, idx=int(oi))

        super()._apply_materialized_deltas(concrete_backend)

    def _commit_deltas(self, base_backend: BaseStorageBackend) -> None:
        for overlay_idx, obj in self._local_objects.items():
            base_idx = int(self._index_map[overlay_idx])
            base_backend.update_objects(obj, idx=base_idx)

        self._local_objects.clear()
        super()._commit_deltas(base_backend)