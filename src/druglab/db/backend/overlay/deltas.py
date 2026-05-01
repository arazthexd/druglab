"""
druglab.db.backend.overlay.deltas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Isolated, memory-efficient delta state containers for overlay domain mixins.

Each delta object is *owned* by exactly one domain mixin and encapsulates all
mutable state for that domain.  The design intentionally avoids storing domain
data directly on the ``OverlayBackend`` instance, which previously caused a
flat namespace collision and made testing harder.

Classes
-------
FeatureDelta
    Tracks locally added/updated feature arrays plus a tombstone set of
    dropped feature names.  Stores only the rows the overlay sees.

MetadataDelta
    Tracks a local ``pd.DataFrame`` patch plus a tombstone set of dropped
    column names.  The local DataFrame has the same number of rows as the
    overlay (``n_overlay_rows``).

ObjectDelta
    Tracks individually mutated objects as a sparse ``{overlay_idx: obj}``
    dict, avoiding a full copy of the object list.

ViewConfig
    Immutable allowlist configuration produced by ``OverlayBackend.set_view()``.
    Stores optional feature allowlist, optional metadata-column allowlist, and
    per-feature column slice constraints.

FeatureCache / MetadataCache
    Prefetched read-through caches.  Mutations write to the *delta*, not to
    the cache; the cache is evicted lazily on next prefetch.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# FeatureDelta
# ---------------------------------------------------------------------------

@dataclass
class FeatureDelta:
    """
    Encapsulates all mutable feature state for a single overlay instance.

    Attributes
    ----------
    local : Dict[str, np.ndarray]
        Feature arrays keyed by name.  Each array has shape
        ``(n_overlay_rows, ...)`` matching the overlay length.
    deleted : Set[str]
        Names of features that have been logically deleted from this overlay.
        Deleted names shadow any same-named feature in the base backend.
    """

    local: Dict[str, np.ndarray] = field(default_factory=dict)
    deleted: Set[str] = field(default_factory=set)

    # ------------------------------------------------------------------
    def has(self, name: str) -> bool:
        return name in self.local

    def is_deleted(self, name: str) -> bool:
        return name in self.deleted

    def get(self, name: str) -> np.ndarray:
        return self.local[name]

    def set(self, name: str, array: np.ndarray) -> None:
        self.deleted.discard(name)
        self.local[name] = array

    def delete(self, name: str) -> None:
        self.deleted.add(name)
        self.local.pop(name, None)

    def clear(self) -> None:
        self.local.clear()
        self.deleted.clear()

    def names(self) -> List[str]:
        return list(self.local.keys())

    def deep_copy(self) -> "FeatureDelta":
        return FeatureDelta(
            local={k: v.copy() for k, v in self.local.items()},
            deleted=set(self.deleted),
        )


# ---------------------------------------------------------------------------
# MetadataDelta
# ---------------------------------------------------------------------------

@dataclass
class MetadataDelta:
    """
    Encapsulates all mutable metadata state for a single overlay instance.

    Attributes
    ----------
    local : pd.DataFrame | None
        Patch DataFrame with the same number of rows as the overlay.  Only
        columns that were explicitly written are present; ``None`` means no
        writes have occurred yet.
    deleted_cols : Set[str]
        Column names that have been logically dropped from this overlay.
    """

    local: Optional[pd.DataFrame] = None
    deleted_cols: Set[str] = field(default_factory=set)

    # ------------------------------------------------------------------
    def ensure_local(self, n_rows: int) -> None:
        if self.local is None:
            self.local = pd.DataFrame(index=range(n_rows))

    def drop_col(self, col: str) -> None:
        self.deleted_cols.add(col)
        if self.local is not None and col in self.local.columns:
            self.local.drop(columns=[col], inplace=True)

    def drop_all_visible(self, visible_cols: List[str]) -> None:
        self.deleted_cols.update(visible_cols)
        if self.local is not None:
            self.local = pd.DataFrame(index=self.local.index)

    def clear(self) -> None:
        self.local = None
        self.deleted_cols.clear()

    def deep_copy(self) -> "MetadataDelta":
        return MetadataDelta(
            local=None if self.local is None else self.local.copy(deep=True),
            deleted_cols=set(self.deleted_cols),
        )


# ---------------------------------------------------------------------------
# ObjectDelta
# ---------------------------------------------------------------------------

@dataclass
class ObjectDelta:
    """
    Sparse overlay-index → object mapping for individually mutated objects.

    Attributes
    ----------
    local : Dict[int, Any]
        Maps overlay row positions (int) to the mutated object at that
        position.  Unmodified rows are *not* stored here; they are read
        from the base backend on demand.
    """

    local: Dict[int, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def has(self, overlay_idx: int) -> bool:
        return overlay_idx in self.local

    def get(self, overlay_idx: int) -> Any:
        return self.local[overlay_idx]

    def set(self, overlay_idx: int, obj: Any) -> None:
        self.local[int(overlay_idx)] = obj

    def clear(self) -> None:
        self.local.clear()

    def deep_copy(self) -> "ObjectDelta":
        return ObjectDelta(local=copy.deepcopy(self.local))


# ---------------------------------------------------------------------------
# ViewConfig  (allowlist + column slicing)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnSlice:
    """
    Describes a column-range restriction on a single feature array.

    Attributes
    ----------
    start : int
    stop : int
        Half-open range ``[start, stop)`` applied to axis-1 of the feature.
    read_only : bool
        Always ``True`` for column-sliced features – partial-column mutations
        are not yet supported.
    """

    start: int
    stop: int
    read_only: bool = True

    def apply(self, arr: np.ndarray) -> np.ndarray:
        return arr[:, self.start:self.stop]


@dataclass
class ViewConfig:
    """
    Allowlist + column-slice constraints produced by ``set_view()``.

    ``None`` means "no restriction" (all items visible).

    Attributes
    ----------
    allowed_features : Set[str] | None
        If set, only these feature names are accessible.
    allowed_meta_cols : Set[str] | None
        If set, only these metadata columns are accessible.
    feature_col_slices : Dict[str, ColumnSlice]
        Per-feature column restrictions (always read-only for now).
    """

    allowed_features: Optional[Set[str]] = None
    allowed_meta_cols: Optional[Set[str]] = None
    feature_col_slices: Dict[str, ColumnSlice] = field(default_factory=dict)

    def check_feature(self, name: str) -> None:
        if self.allowed_features is not None and name not in self.allowed_features:
            raise KeyError(
                f"Feature '{name}' is not in the view allowlist. "
                f"Allowed: {sorted(self.allowed_features)}"
            )

    def check_meta_col(self, col: str) -> None:
        if self.allowed_meta_cols is not None and col not in self.allowed_meta_cols:
            raise KeyError(
                f"Metadata column '{col}' is not in the view allowlist. "
                f"Allowed: {sorted(self.allowed_meta_cols)}"
            )

    def is_col_sliced(self, name: str) -> bool:
        return name in self.feature_col_slices

    def get_col_slice(self, name: str) -> Optional[ColumnSlice]:
        return self.feature_col_slices.get(name)

    def apply_feature_filter(self, names: List[str]) -> List[str]:
        if self.allowed_features is None:
            return names
        return [n for n in names if n in self.allowed_features]

    def apply_meta_col_filter(self, cols: List[str]) -> List[str]:
        if self.allowed_meta_cols is None:
            return cols
        return [c for c in cols if c in self.allowed_meta_cols]


# ---------------------------------------------------------------------------
# FeatureCache
# ---------------------------------------------------------------------------

@dataclass
class FeatureCache:
    """
    Prefetched feature data.  Mutations bypass this cache; commits ignore it.

    Attributes
    ----------
    data : Dict[str, np.ndarray]
        Maps feature name → prefetched array (overlay-row-ordered).
    """

    data: Dict[str, np.ndarray] = field(default_factory=dict)

    def has(self, name: str) -> bool:
        return name in self.data

    def get(self, name: str) -> np.ndarray:
        return self.data[name]

    def put(self, name: str, array: np.ndarray) -> None:
        self.data[name] = array

    def evict(self, name: str) -> None:
        self.data.pop(name, None)

    def clear(self) -> None:
        self.data.clear()


# ---------------------------------------------------------------------------
# MetadataCache
# ---------------------------------------------------------------------------

@dataclass
class MetadataCache:
    """
    Prefetched metadata data.  Commits ignore this cache entirely.

    Attributes
    ----------
    data : pd.DataFrame | None
        Prefetched slice of base metadata aligned to overlay rows.
    cached_cols : Set[str]
        Which columns are currently cached.
    """

    data: Optional[pd.DataFrame] = None
    cached_cols: Set[str] = field(default_factory=set)

    def has_col(self, col: str) -> bool:
        return col in self.cached_cols

    def get(self, cols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        if self.data is None:
            return None
        if cols is None:
            return self.data
        available = [c for c in cols if c in self.data.columns]
        if not available:
            return None
        return self.data[available]

    def put(self, df: pd.DataFrame) -> None:
        if self.data is None:
            self.data = df.copy()
        else:
            for col in df.columns:
                self.data[col] = df[col].values
        self.cached_cols.update(df.columns.tolist())

    def evict_col(self, col: str) -> None:
        if self.data is not None and col in self.data.columns:
            self.data.drop(columns=[col], inplace=True)
        self.cached_cols.discard(col)

    def clear(self) -> None:
        self.data = None
        self.cached_cols.clear()