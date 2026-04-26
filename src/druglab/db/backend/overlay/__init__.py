"""
druglab.db.backend.overlay
~~~~~~~~~~~~~~~~~~~~~~~~~~~
``OverlayBackend`` - a zero-copy, detachable, caching, column-aware proxy
backend with Copy-on-Write semantics.

Architecture
------------
``OverlayBackend`` is composed of three domain mixins that each own their
state via isolated delta/cache dataclasses:

    OverlayObjectMixin   → ObjectDelta
    OverlayMetadataMixin → MetadataDelta + MetadataCache
    OverlayFeatureMixin  → FeatureDelta  + FeatureCache

Notes
-----
- Architecture: mixins reference the context via ``OverlayContextProtocol``;
    delta state is encapsulated in dataclasses rather than flat ``self`` attrs.

- View Config: ``set_view()`` installs an allowlist + column slices.
    Column-sliced features are read-only; mutations raise ``RuntimeError``.

- Cache: ``prefetch()`` fills ``FeatureCache`` / ``MetadataCache``.
    Read order: Delta → Cache → Base.  Mutations invalidate cache.
    ``commit()`` ignores cached data.

- Detach/Attach: ``detach()`` sets ``_base = None``; reads that miss
    Delta and Cache raise ``DetachedStateError``.  ``attach()`` validates
    UUID + dimensions before restoring ``_base``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import uuid as _uuid_mod

import numpy as np

from ...indexing import normalize_row_index
from ..base import BaseStorageBackend
from .deltas import ColumnSlice, ViewConfig
from .identity import DetachedStateError, SchemaIdentity
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
    """
    A zero-copy proxy backend with Copy-on-Write semantics, view bounding,
    prefetch caching, and safe detach/attach for cross-process transport.

    Parameters
    ----------
    base_backend : BaseStorageBackend
        The concrete backend this overlay wraps.
    index_map : np.ndarray of dtype np.intp, optional
        Maps overlay row positions → base row positions.
        ``None`` → identity map over all base rows.
    """

    BACKEND_NAME = "OverlayBackend"

    def __init__(
        self,
        base_backend: BaseStorageBackend,
        index_map: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        # Flatten nested overlays: compose index maps, reuse the concrete base.
        if isinstance(base_backend, OverlayBackend):
            outer_map = base_backend._index_map
            if index_map is None:
                composed = outer_map.copy()
            else:
                composed = outer_map[np.asarray(index_map, dtype=np.intp)]
            self._base = base_backend._base
            self._index_map = composed
        else:
            self._base = base_backend
            self._index_map = (
                np.arange(len(base_backend), dtype=np.intp)
                if index_map is None
                else np.asarray(index_map, dtype=np.intp)
            )

        # Capture schema identity at creation time so attach() can validate.
        self._base_identity: Optional[SchemaIdentity] = (
            SchemaIdentity.capture(self._base) if self._base is not None else None
        )

        # View configuration: starts unrestricted.
        self._view_config = ViewConfig()

        # Initialise all domain mixin delta/cache state.
        self._initialize_overlay_context(**kwargs)

    # ------------------------------------------------------------------
    # OverlayContextProtocol implementation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Suppress the BaseStorageBackend lifecycle hooks – OverlayBackend
    # manages its own initialisation via _initialize_overlay_context.
    # ------------------------------------------------------------------

    def initialize_storage_context(self, **kwargs) -> None:
        return

    def bind_capabilities(self) -> None:
        return

    def post_initialize_validate(self) -> None:
        return

    def validate(self) -> None:
        return

    # ------------------------------------------------------------------
    # View Configuration
    # ------------------------------------------------------------------

    def set_view(
        self,
        features: Optional[List[str]] = None,
        meta_cols: Optional[List[str]] = None,
        feature_col_slices: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> None:
        """
        Restrict this overlay's visible features and metadata columns.

        Parameters
        ----------
        features : List[str] | None
            Allowlist of feature names.  ``None`` means no restriction.
        meta_cols : List[str] | None
            Allowlist of metadata columns.  ``None`` means no restriction.
        feature_col_slices : Dict[str, Tuple[int, int]] | None
            Per-feature column restrictions as ``(start, stop)`` pairs.
            Sliced features are automatically **read-only**.

        Notes
        -----
        * Accessing a feature/column not in the allowlist raises ``KeyError``.
        * Mutating a column-sliced feature raises ``RuntimeError``.
        * Calling ``set_view()`` again replaces the previous config entirely.
        """
        col_slices: Dict[str, ColumnSlice] = {}
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

    # ------------------------------------------------------------------
    # Prefetch / Caching
    # ------------------------------------------------------------------

    def prefetch(
        self,
        features: Optional[List[str]] = None,
        meta_cols: Optional[List[str]] = None,
        rows: Optional["INDEX_LIKE"] = None,
    ) -> None:
        """
        Eagerly load data from the base backend into the overlay's read cache.

        After a prefetch the overlay can answer read requests for the given
        features/columns **without** touching the base backend.  This is the
        preparation step before ``detach()``.

        Parameters
        ----------
        features : List[str] | None
            Feature names to prefetch.  ``None`` → prefetch all base features.
        meta_cols : List[str] | None
            Metadata columns to prefetch.  ``None`` → prefetch all base columns.
        rows : INDEX_LIKE | None
            Row subset to prefetch.  ``None`` → all rows in the overlay.

        Notes
        -----
        * Prefetching a name that is already in the delta is a no-op.
        * Mutations after prefetch will invalidate the corresponding cache entry.
        * ``commit()`` only flushes delta data; cache is ignored.
        """
        if self._base is None:
            raise DetachedStateError("Cannot prefetch: overlay is already detached.")

        overlay_positions: Optional[np.ndarray]
        if rows is not None:
            overlay_positions = self._resolve_overlay_idx(rows)
        else:
            overlay_positions = None

        # Feature prefetch
        feat_names = features if features is not None else self._base.get_feature_names()
        self._prefetch_features(feat_names, overlay_positions)

        # Metadata prefetch
        all_base_cols = self._base.get_metadata_columns()
        col_list = meta_cols if meta_cols is not None else all_base_cols
        if col_list:
            self._prefetch_metadata(col_list, overlay_positions)

    # ------------------------------------------------------------------
    # Detach / Attach
    # ------------------------------------------------------------------

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

        incoming_identity = SchemaIdentity.capture(base_backend)
        self._base_identity.validate_compatible(incoming_identity)
        self._base = base_backend

    @property
    def is_detached(self) -> bool:
        """``True`` when the overlay has no base backend reference."""
        return self._base is None

    # ------------------------------------------------------------------
    # Clone (deep-copy of the overlay shell, sharing the same base)
    # ------------------------------------------------------------------

    def clone(
        self,
        target_path: Optional[Path] = None,
        index_map: Optional[np.ndarray] = None,
    ) -> "OverlayBackend":
        clone_index_map = (
            self._index_map.copy() if index_map is None
            else np.asarray(index_map, dtype=np.intp)
        )
        # Build a new shell without calling __init__ (to avoid base-flattening)
        cloned = object.__new__(self.__class__)
        cloned._base = self._base
        cloned._base_identity = self._base_identity
        cloned._index_map = clone_index_map
        cloned._view_config = ViewConfig(
            allowed_features=(
                set(self._view_config.allowed_features)
                if self._view_config.allowed_features is not None
                else None
            ),
            allowed_meta_cols=(
                set(self._view_config.allowed_meta_cols)
                if self._view_config.allowed_meta_cols is not None
                else None
            ),
            feature_col_slices=dict(self._view_config.feature_col_slices),
        )
        # Deep-copy all delta state
        cloned._feat_delta = self._feat_delta.deep_copy()
        cloned._feat_cache = FeatureCache()  # do not share cache
        cloned._meta_delta = self._meta_delta.deep_copy()
        cloned._meta_cache = MetadataCache()  # do not share cache
        cloned._obj_delta = self._obj_delta.deep_copy()
        return cloned

    # ------------------------------------------------------------------
    # Materialize – collapse overlay into a concrete backend
    # ------------------------------------------------------------------

    def materialize(self, target_path: Optional[Path] = None) -> BaseStorageBackend:
        """
        Collapse the overlay into a standalone concrete backend.

        Applies the full delta stack on top of the (optionally row-sliced)
        base data.  The result is an ``EagerMemoryBackend`` (or whatever class
        ``self._base`` is) that is completely independent of this overlay.
        """
        if self._base is None:
            raise DetachedStateError(
                "Cannot materialize a detached overlay.  Re-attach the base "
                "backend first."
            )
        concrete: BaseStorageBackend = self._base.clone(
            target_path=target_path,
            index_map=self._index_map,
        )
        self._apply_materialized_deltas(concrete)
        return concrete

    # ------------------------------------------------------------------
    # Commit – flush deltas to base
    # ------------------------------------------------------------------

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
            raise DetachedStateError(
                "Cannot commit: overlay is detached.  Re-attach before committing."
            )
        self._commit_deltas(self._base)

    # ------------------------------------------------------------------
    # Save (delegates to materialize → save)
    # ------------------------------------------------------------------

    def save(
        self,
        path: Path,
        object_writer: Optional[Callable[[list, Path], None]] = None,
        **kwargs,
    ) -> None:
        self.materialize().save(path, object_writer=object_writer, **kwargs)

    # ------------------------------------------------------------------
    # Identity helpers for BaseTable.save / load
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return self._base.__class__.__name__ if self._base is not None else "OverlayBackend"

    def get_module(self) -> str:
        return self._base.__class__.__module__ if self._base is not None else __name__


# ---------------------------------------------------------------------------
# Lazy imports to avoid circular dependencies
# ---------------------------------------------------------------------------

from .deltas import FeatureCache, MetadataCache  # noqa: E402