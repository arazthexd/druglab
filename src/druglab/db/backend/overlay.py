"""
druglab.db.backend.overlay
~~~~~~~~~~~~~~~~~~~~~~~~~~~
OverlayBackend: a zero-copy proxy wrapper around any BaseStorageBackend.
 
Implements:
- Zero-copy row filtering via index mapping
- Copy-on-Write (CoW) semantics for mutations (deltas stored locally)
- Tombstones for deletions (never touches the base backend)
- ``.materialize(target_path)`` → concrete backend 
- ``.commit()`` → flush deltas to base backend
- ``.save(path)`` → delegates to materialize().save(path)

Materialization Note
--------------------
``materialize(target_path=None)`` operates in two clean phases:
 
Phase A - Clone the base
    ``concrete = self._base.clone_concrete(target_path, index_map)``
    This calls the cooperative ``_gather_materialized_state`` MRO chain on
    the *base* backend to build a new instance of the **same concrete class**
    (e.g. ``EagerMemoryBackend``).  No hardcoding of ``EagerMemoryBackend``.
 
Phase B - Apply local deltas
    Iterate over the CoW state (local features, metadata, objects, tombstones)
    and apply them to *concrete* via the public backend API.
 
This design means that future out-of-core backends (e.g. ``ZarrBackend``)
can materialise correctly just by implementing ``_gather_materialized_state``,
without any changes to ``OverlayBackend``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from .base import (
    BaseStorageBackend,
    BaseMetadataMixin,
    BaseObjectMixin,
    BaseFeatureMixin,
)
from druglab.db.indexing import RowSelection, normalize_row_index

if TYPE_CHECKING:
    from druglab.db.backend import EagerMemoryBackend
    from druglab.db.indexing import INDEX_LIKE

__all__ = ["OverlayBackend"]


class OverlayBackend(BaseStorageBackend):
    """
    A zero-copy proxy backend that sits in front of any BaseStorageBackend.

    Initialization flattens nested overlays: if *base_backend* is already
    an OverlayBackend, the index maps are composed and the new instance
    points directly at the absolute base, preventing deep call stacks.

    Reads are delegated to the base (dynamic — no snapshot isolation).
    Writes are intercepted and stored in local delta state (CoW).
    Tombstones (dropped features / metadata columns) shadow the base.
    """

    BACKEND_NAME = "OverlayBackend"

    def __init__(
        self,
        base_backend: BaseStorageBackend,
        index_map: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        # ----------------------------------------------------------------
        # Flatten nested overlays
        # ----------------------------------------------------------------
        if isinstance(base_backend, OverlayBackend):
            # Compose index maps: self._index_map[i] → outer._index_map[i] → absolute base
            outer_map = base_backend._index_map
            if index_map is None:
                # Full view of the outer overlay
                composed = outer_map.copy()
            else:
                idx = np.asarray(index_map, dtype=np.intp)
                composed = outer_map[idx]
            self._base = base_backend._base
            self._index_map = composed
        else:
            self._base = base_backend
            if index_map is None:
                self._index_map = np.arange(len(base_backend), dtype=np.intp)
            else:
                self._index_map = np.asarray(index_map, dtype=np.intp)

        # ----------------------------------------------------------------
        # CoW delta state
        # ----------------------------------------------------------------
        self._local_features: Dict[str, np.ndarray] = {}
        self._local_metadata: Optional[pd.DataFrame] = None   # full overlay-sized df of changes
        self._local_objects: Dict[int, Any] = {}              # overlay-index → object

        # Tombstones
        self._deleted_features: Set[str] = set()
        self._deleted_metadata_cols: Set[str] = set()

        # Do NOT call super().__init__() because that fires lifecycle hooks
        # which expect the cooperative mixin chain — OverlayBackend bypasses
        # that chain entirely and manages its own state above.
        # We skip the lifecycle hooks intentionally.

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index_map)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _translate(self, overlay_idx: np.ndarray) -> np.ndarray:
        """Map overlay-local positions to base positions."""
        return self._index_map[overlay_idx]

    def _n_rows(self) -> int:
        return len(self._index_map)

    def _resolve_overlay_idx(self, idx: Optional[INDEX_LIKE]) -> np.ndarray:
        """Return an array of overlay-local positions (never None)."""
        positions = normalize_row_index(idx, self._n_rows())
        if positions is None:
            return np.arange(self._n_rows(), dtype=np.intp)
        return positions

    # ------------------------------------------------------------------
    # BaseMetadataMixin
    # ------------------------------------------------------------------

    def _n_metadata_rows(self) -> int:
        return self._n_rows()

    def get_metadata(
        self,
        idx: Optional[INDEX_LIKE] = None,
        cols: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        overlay_positions = self._resolve_overlay_idx(idx)
        base_positions = self._translate(overlay_positions)

        # Fetch from base (excluding deleted columns)
        base_cols = self._base.get_metadata(idx=base_positions).reset_index(drop=True)

        # Remove tombstoned columns
        drop_cols = [c for c in self._deleted_metadata_cols if c in base_cols.columns]
        if drop_cols:
            base_cols = base_cols.drop(columns=drop_cols)

        # Apply local metadata overrides
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
        # Remove from tombstone if re-adding
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
            self._local_metadata.iloc[overlay_positions, self._local_metadata.columns.get_loc(name)] = value

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

        # Check all columns exist (in base or local) and are not tombstoned
        all_visible_cols = self._visible_metadata_columns()
        for col in val_dict:
            if col not in all_visible_cols:
                raise KeyError(
                    f"Column '{col}' does not exist. Use add_metadata_column."
                )

        if self._local_metadata is None:
            self._local_metadata = pd.DataFrame(index=range(n))

        for col, val in val_dict.items():
            if col not in self._local_metadata.columns:
                # Seed from base for all overlay rows
                base_all = self._base.get_metadata(
                    idx=self._index_map, cols=[col]
                ).reset_index(drop=True)
                self._local_metadata[col] = base_all[col].values

            if idx is None:
                self._local_metadata[col] = val
            else:
                self._local_metadata.iloc[
                    overlay_positions,
                    self._local_metadata.columns.get_loc(col)
                ] = val

    def drop_metadata_columns(
        self,
        cols: Optional[Union[str, List[str]]] = None,
    ) -> None:
        if cols is None:
            # Tombstone all visible columns
            visible = self._visible_metadata_columns()
            self._deleted_metadata_cols.update(visible)
            if self._local_metadata is not None:
                self._local_metadata = pd.DataFrame(index=self._local_metadata.index)
        else:
            if isinstance(cols, str):
                cols = [cols]
            for col in cols:
                self._deleted_metadata_cols.add(col)
                if self._local_metadata is not None and col in self._local_metadata.columns:
                    self._local_metadata.drop(columns=[col], inplace=True)

    def _visible_metadata_columns(self) -> List[str]:
        """Return columns visible through this overlay (base minus tombstones plus local)."""
        base_cols = set(self._base.get_metadata().columns) - self._deleted_metadata_cols
        local_cols = set(self._local_metadata.columns) if self._local_metadata is not None else set()
        return list(base_cols | local_cols)

    def _validate_metadata(self) -> None:
        pass

    # ------------------------------------------------------------------
    # BaseObjectMixin
    # ------------------------------------------------------------------

    def _n_objects(self) -> int:
        return self._n_rows()

    def get_objects(
        self,
        idx: Optional[INDEX_LIKE] = None,
    ) -> Union[Any, List[Any]]:
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
            oi = int(overlay_positions[0])
            self._local_objects[oi] = objs
        else:
            if len(objs) != len(overlay_positions):
                raise ValueError("Length of objs must match index length.")
            for oi, obj in zip(overlay_positions, objs):
                self._local_objects[int(oi)] = obj

    def _validate_objects(self) -> None:
        pass

    # ------------------------------------------------------------------
    # BaseFeatureMixin
    # ------------------------------------------------------------------

    def _n_feature_rows(self) -> int:
        return self._n_rows()

    def get_feature(
        self,
        name: str,
        idx: Optional[INDEX_LIKE] = None,
    ) -> np.ndarray:
        # Check tombstone first
        if name in self._deleted_features:
            raise KeyError(f"Feature '{name}' has been deleted.")

        overlay_positions = self._resolve_overlay_idx(idx)

        if name in self._local_features:
            arr = self._local_features[name]
            return arr[overlay_positions]

        # Delegate to base with translated indices
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
        # Remove from tombstone if re-adding
        self._deleted_features.discard(name)

        n = self._n_rows()
        array = np.asarray(array)
        overlay_positions = self._resolve_overlay_idx(idx)

        if name not in self._local_features:
            if idx is None:
                # Full replacement
                if array.shape[0] != n:
                    raise ValueError(
                        f"Array length {array.shape[0]} does not match overlay length {n}."
                    )
                self._local_features[name] = array.copy()
            else:
                # Partial: seed from base if exists, else zeros/nan
                if name in self._base.get_feature_names() and name not in self._deleted_features:
                    base_all = self._base.get_feature(name, idx=self._index_map)
                    full = base_all.copy()
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
                    raise ValueError(
                        f"Array length {array.shape[0]} does not match overlay length {n}."
                    )
                self._local_features[name] = array.copy()
            else:
                self._local_features[name][overlay_positions] = array

    def drop_feature(self, name: str) -> None:
        # Check it exists (in base or local)
        base_names = self._base.get_feature_names()
        if name not in base_names and name not in self._local_features:
            raise KeyError(f"Feature '{name}' does not exist.")
        self._deleted_features.add(name)
        # Remove from local delta if present
        self._local_features.pop(name, None)

    def get_feature_names(self) -> List[str]:
        base_names = set(self._base.get_feature_names())
        visible = (base_names - self._deleted_features) | set(self._local_features.keys())
        return list(visible)

    def get_feature_shape(self, name: str) -> tuple:
        if name in self._deleted_features:
            raise KeyError(f"Feature '{name}' has been deleted.")
        if name in self._local_features:
            return self._local_features[name].shape
        # Shape from base but with overlay row count
        base_shape = self._base.get_feature_shape(name)
        return (self._n_rows(),) + base_shape[1:]

    def _validate_features(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Lifecycle hooks (no-ops for OverlayBackend)
    # ------------------------------------------------------------------

    def initialize_storage_context(self, **kwargs) -> None:
        pass

    def bind_capabilities(self) -> None:
        pass

    def post_initialize_validate(self) -> None:
        pass

    def validate(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def materialize(
        self,
        target_path: Optional[Path] = None,
    ) -> BaseStorageBackend:
        """
        Collapse the overlay proxy tree into a concrete backend of the same
        class as ``self._base``.
 
        The Director operates in two phases:
 
        **Phase A - Clone the base**
            ``concrete = self._base.clone_concrete(target_path, self._index_map)``
            This calls the cooperative ``_gather_materialized_state`` MRO chain
            on the *actual base backend*, building a new instance of the **same
            concrete type** (e.g. ``EagerMemoryBackend``).  No class is hardcoded.
 
        **Phase B - Apply local deltas**
            Iterate over ``_deleted_features``, ``_local_features``,
            ``_deleted_metadata_cols``, ``_local_metadata``, and
            ``_local_objects``, applying each to *concrete* via its public API.
 
        Parameters
        ----------
        target_path : Path, optional
            Forwarded to ``clone_concrete``.  Reserved for future out-of-core
            backends that stream sliced state to disk in Phase A.
 
        Returns
        -------
        BaseStorageBackend
            A new concrete backend of type ``type(self._base)`` with the
            overlay's logical state fully materialised.
        """
        # ---- Phase A: Clone the base, restricted to our index_map ----
        concrete: BaseStorageBackend = self._base.clone_concrete(
            target_path=target_path,
            index_map=self._index_map,
        )
 
        # ---- Phase B: Apply CoW deltas --------------------------------
 
        # 1. Tombstoned features (delete from the cloned backend)
        for name in self._deleted_features:
            if name in concrete.get_feature_names():
                concrete.drop_feature(name)
 
        # 2. Local feature overrides (add or update)
        for name, arr in self._local_features.items():
            if name in self._deleted_features:
                continue  # re-tombstoned after re-add, skip
            if name in concrete.get_feature_names():
                concrete.update_feature(name, arr)
            else:
                concrete.update_feature(name, arr)  # add new feature
 
        # 3. Tombstoned metadata columns
        if self._deleted_metadata_cols:
            existing_cols = set(concrete.get_metadata().columns)
            to_drop = list(self._deleted_metadata_cols & existing_cols)
            if to_drop:
                concrete.drop_metadata_columns(to_drop)
 
        # 4. Local metadata overrides
        if self._local_metadata is not None and not self._local_metadata.empty:
            existing_cols = set(concrete.get_metadata().columns)
            for col in self._local_metadata.columns:
                if col in self._deleted_metadata_cols:
                    continue  # tombstoned
                val = self._local_metadata[col].values
                if col in existing_cols:
                    concrete.update_metadata({col: val})
                else:
                    concrete.add_metadata_column(col, val)
 
        # 5. Local object overrides
        #    _local_objects keys are overlay-local indices (0..n-1), which
        #    correspond directly to rows in the cloned concrete backend.
        for oi, obj in self._local_objects.items():
            concrete.update_objects(obj, idx=int(oi))
 
        return concrete
    
    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def commit(self) -> None:
        """
        Flush all local deltas down to the base backend, then clear local state.
        Tombstoned features/columns are propagated to the base as deletions.
        """
        # Flush feature updates
        for name, arr in self._local_features.items():
            self._base.update_feature(name, arr, idx=self._index_map)

        # Flush metadata updates
        if self._local_metadata is not None and not self._local_metadata.empty:
            for col in self._local_metadata.columns:
                val = self._local_metadata[col].values
                if col not in self._base.get_metadata().columns:
                    self._base.add_metadata_column(col, val, idx=self._index_map)
                else:
                    self._base.update_metadata({col: val}, idx=self._index_map)

        # Flush object updates
        for overlay_idx, obj in self._local_objects.items():
            base_idx = int(self._index_map[overlay_idx])
            self._base.update_objects(obj, idx=base_idx)

        # Propagate tombstones to base
        for name in self._deleted_features:
            if name in self._base.get_feature_names():
                self._base.drop_feature(name)

        if self._deleted_metadata_cols:
            existing_base_cols = set(self._base.get_metadata().columns)
            cols_to_drop = list(self._deleted_metadata_cols & existing_base_cols)
            if cols_to_drop:
                self._base.drop_metadata_columns(cols_to_drop)

        # Clear local delta state
        self._local_features.clear()
        self._local_metadata = None
        self._local_objects.clear()
        self._deleted_features.clear()
        self._deleted_metadata_cols.clear()

    def save(
        self,
        path: Path,
        object_writer: Optional[Callable[[List[Any], Path], None]] = None,
        **kwargs
    ) -> None:
        """
        Save as a .dlb bundle by materialising first, then delegating to
        the concrete backend's save().  This ensures a clean, unified state.
        """
        self.materialize().save(path, object_writer=object_writer, **kwargs)
    
    def get_name(self) -> str:
        return self._base.__class__.__name__
    
    def get_module(self) -> str:
        return self._base.__class__.__module__