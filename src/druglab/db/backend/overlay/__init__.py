"""
druglab.db.backend.overlay
~~~~~~~~~~~~~~~~~~~~~~~~~~
``OverlayBackend`` — a zero-copy, CoW view over any ``BaseStorageBackend``.

Architecture
------------
``OverlayBackend`` now inherits from ``CompositeStorageBackend`` so all
standard domain routing (``get_feature`` → ``self._feature_store``, etc.)
is handled by the parent class without duplication.

The three slots (``self._feature_store``, ``self._metadata_store``,
``self._object_store``) are filled with domain-specific *proxy* stores
(``OverlayFeatureStore``, ``OverlayMetadataStore``, ``OverlayObjectStore``)
that each wrap exactly their matching base store.  Cross-domain access is
structurally impossible: a feature store cannot reach the metadata store.

Index map composition (nested overlays)
----------------------------------------
When *base_backend* is itself an ``OverlayBackend``, the constructor
collapses the chain by composing the two index maps and pointing at the
concrete base directly.  The resulting ``OverlayBackend`` always has
``isinstance(self._base, EagerMemoryBackend)`` (or another concrete type),
never ``OverlayBackend``.

Detach / attach
---------------
``detach()`` sets ``self._base = None`` **and** nulls out ``_base_store``
on each proxy store.  Any subsequent read that cannot be satisfied from the
delta or cache raises ``DetachedStateError``.

``attach()`` validates schema compatibility and restores the base reference
in both ``self._base`` and each proxy store's ``_base_store``.

View configuration
------------------
``set_view()`` rebuilds a ``ViewConfig`` dataclass and pushes the new
reference into each proxy store.  ``clear_view()`` resets to unrestricted.

Thread-safety
-------------
Same as ``BaseStorageBackend``: concurrent writes are not safe.  Use the
scatter-gather pattern (prefetch → detach → worker mutation → attach →
commit) for multiprocessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ...indexing import normalize_row_index
from ..base import BaseStorageBackend
from ..composite import CompositeStorageBackend
from .deltas import ColumnSlice, ViewConfig
from .identity import DetachedStateError, SchemaIdentity
from .stores import OverlayFeatureStore, OverlayMetadataStore, OverlayObjectStore

if TYPE_CHECKING:
    from druglab.db.indexing import INDEX_LIKE

__all__ = ["OverlayBackend"]


class OverlayBackend(CompositeStorageBackend):
    """
    Zero-copy, CoW view over a ``BaseStorageBackend``.

    Parameters
    ----------
    base_backend : BaseStorageBackend
        The backend to view.  May itself be an ``OverlayBackend``; in that
        case the index maps are composed and the chain collapses.
    index_map : np.ndarray of dtype np.intp, optional
        Row positions (in *base_backend*) that this overlay exposes.
        ``None`` → expose all rows in order.
    """

    BACKEND_NAME = "OverlayBackend"

    def __init__(
        self,
        base_backend: BaseStorageBackend,
        index_map: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        # ------------------------------------------------------------------
        # 1. Resolve the concrete base and compose index maps.
        # ------------------------------------------------------------------
        if isinstance(base_backend, OverlayBackend):
            outer_map = base_backend._index_map
            if index_map is None:
                composed = outer_map.copy()
            else:
                composed = outer_map[np.asarray(index_map, dtype=np.intp)]
            concrete_base = base_backend._base   # already the concrete backend
        else:
            concrete_base = base_backend
            composed = (
                np.arange(len(base_backend), dtype=np.intp)
                if index_map is None
                else np.asarray(index_map, dtype=np.intp)
            )

        self._base: Optional[BaseStorageBackend] = concrete_base
        self._index_map: np.ndarray = composed
        self._base_identity = (
            SchemaIdentity.capture(concrete_base)
            if concrete_base is not None
            else None
        )
        self._view_config = ViewConfig()

        # ------------------------------------------------------------------
        # 2. Build proxy stores wrapping the *concrete* base's stores.
        # ------------------------------------------------------------------
        # Access the concrete stores only when the concrete base is a
        # CompositeStorageBackend (which EagerMemoryBackend is).  For any
        # other concrete backend we fall back to the backend-level API by
        # wrapping a thin adapter.  In practice all DrugLab concrete backends
        # are CompositeStorageBackend subclasses.
        if isinstance(concrete_base, CompositeStorageBackend):
            base_obj_store = concrete_base._object_store
            base_meta_store = concrete_base._metadata_store
            base_feat_store = concrete_base._feature_store
        else:
            # Fallback: use None (detached-like) — uncommon path.
            base_obj_store = None
            base_meta_store = None
            base_feat_store = None

        obj_proxy = OverlayObjectStore(base_obj_store, self._index_map)
        meta_proxy = OverlayMetadataStore(
            base_meta_store, self._index_map, self._view_config
        )
        feat_proxy = OverlayFeatureStore(
            base_feat_store, self._index_map, self._view_config
        )

        # ------------------------------------------------------------------
        # 3. Hand the proxies to CompositeStorageBackend.
        #    We bypass CompositeStorageBackend.__init__'s validate() call
        #    because an overlay's validate() is a no-op.
        # ------------------------------------------------------------------
        # Call BaseStorageBackend.__init__ (sets schema_uuid) then wire stores.
        BaseStorageBackend.__init__(self)
        self._object_store = obj_proxy
        self._metadata_store = meta_proxy
        self._feature_store = feat_proxy

    # ------------------------------------------------------------------
    # Length — overlay exposes exactly len(index_map) rows.
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index_map)

    # ------------------------------------------------------------------
    # Validation — overlays are always structurally valid by construction.
    # ------------------------------------------------------------------

    def validate(self) -> None:  # noqa: D102
        return

    # ------------------------------------------------------------------
    # View configuration
    # ------------------------------------------------------------------

    def set_view(
        self,
        features: Optional[List[str]] = None,
        meta_cols: Optional[List[str]] = None,
        feature_col_slices: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> None:
        """
        Restrict the overlay's visible features and/or metadata columns.

        Parameters
        ----------
        features : List[str], optional
            Allowlist of feature names.  Access to any other name raises
            ``KeyError``.
        meta_cols : List[str], optional
            Allowlist of metadata column names.
        feature_col_slices : Dict[str, (int, int)], optional
            Per-feature ``(start, stop)`` column restrictions (read-only).
        """
        col_slices: Dict[str, ColumnSlice] = {}
        if feature_col_slices:
            for name, (start, stop) in feature_col_slices.items():
                col_slices[name] = ColumnSlice(start=start, stop=stop, read_only=True)

        new_cfg = ViewConfig(
            allowed_features=set(features) if features is not None else None,
            allowed_meta_cols=set(meta_cols) if meta_cols is not None else None,
            feature_col_slices=col_slices,
        )
        self._view_config = new_cfg
        # Push the new config reference into every proxy store.
        self._feature_store._view_config = new_cfg
        self._metadata_store._view_config = new_cfg

    def clear_view(self) -> None:
        """Remove all view restrictions (reset to unrestricted access)."""
        self.set_view()

    # ------------------------------------------------------------------
    # Prefetch
    # ------------------------------------------------------------------

    def prefetch(
        self,
        features: Optional[List[str]] = None,
        meta_cols: Optional[List[str]] = None,
        rows: Optional["INDEX_LIKE"] = None,
    ) -> None:
        """
        Warm the proxy-store caches with data from the base backend.

        Parameters
        ----------
        features : List[str], optional
            Feature names to prefetch.  ``None`` → all base features.
        meta_cols : List[str], optional
            Metadata columns to prefetch.  ``None`` → all base columns.
        rows : INDEX_LIKE, optional
            Subset of overlay rows to prefetch.  ``None`` → all rows.
        """
        if self._base is None:
            raise DetachedStateError(
                "Cannot prefetch: overlay is already detached."
            )
        n = len(self._index_map)
        overlay_positions = (
            normalize_row_index(rows, n)
            if rows is not None
            else None
        )

        feat_names = (
            features
            if features is not None
            else self._base.get_feature_names()
        )
        self._feature_store.prefetch(feat_names, overlay_positions)

        all_base_cols = self._base.get_metadata_columns()
        col_list = meta_cols if meta_cols is not None else all_base_cols
        if col_list:
            self._metadata_store.prefetch(col_list, overlay_positions)

    # ------------------------------------------------------------------
    # Detach / attach
    # ------------------------------------------------------------------

    def detach(self) -> None:
        """
        Sever the reference to the base backend.

        After detaching the overlay is self-contained: any read satisfied from
        the delta or the prefetch cache succeeds; any read that requires the
        base raises ``DetachedStateError``.
        """
        self._base = None
        self._object_store._base_store = None
        self._metadata_store._base_store = None
        self._feature_store._base_store = None

    def attach(self, base_backend: BaseStorageBackend) -> None:
        """
        Re-attach a base backend after detaching or cross-process transport.

        The supplied backend must be schema-compatible with the identity
        captured at overlay creation time (same UUID, row count, feature
        schema, and metadata schema).

        Raises
        ------
        ValueError
            If the supplied backend is not schema-compatible.
        """
        if self._base_identity is None:
            self._base = base_backend
            self._base_identity = SchemaIdentity.capture(base_backend)
            self._rewire_proxy_stores(base_backend)
            return

        self._base_identity.validate_compatible(
            SchemaIdentity.capture(base_backend)
        )
        self._base = base_backend
        self._rewire_proxy_stores(base_backend)

    def _rewire_proxy_stores(self, base_backend: BaseStorageBackend) -> None:
        """Restore ``_base_store`` in every proxy after an attach."""
        if isinstance(base_backend, CompositeStorageBackend):
            self._object_store._base_store = base_backend._object_store
            self._metadata_store._base_store = base_backend._metadata_store
            self._feature_store._base_store = base_backend._feature_store
        else:
            # For non-composite concrete backends we cannot re-wire stores;
            # leave them None and surface errors on access (unusual path).
            pass

    @property
    def is_detached(self) -> bool:
        """``True`` when the overlay has no base backend reference."""
        return self._base is None

    # ------------------------------------------------------------------
    # Clone
    # ------------------------------------------------------------------

    def clone(
        self,
        target_path: Optional[Path] = None,
        index_map: Optional[np.ndarray] = None,
    ) -> "OverlayBackend":
        """
        Return a new ``OverlayBackend`` sharing the same concrete base but
        with independent delta state.

        Parameters
        ----------
        target_path
            Reserved for out-of-core backends; ignored here.
        index_map
            Override the cloned overlay's row selection.  ``None`` → use
            the same ``_index_map`` as this overlay (copied).
        """
        clone_map = (
            self._index_map.copy()
            if index_map is None
            else np.asarray(index_map, dtype=np.intp)
        )

        # Build a new overlay pointing at the same concrete base.
        # We use object.__new__ to avoid triggering __init__ (which would
        # re-read from base_backend and reset deltas).
        cloned = object.__new__(self.__class__)

        # Copy scalar / reference attributes.
        cloned._base = self._base
        cloned._base_identity = self._base_identity
        cloned._index_map = clone_map

        # Deep-copy the view config so the clone's config is independent.
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

        # New proxy stores with the same base stores but fresh (empty) caches.
        cloned._object_store = OverlayObjectStore(
            self._object_store._base_store, clone_map
        )
        cloned._metadata_store = OverlayMetadataStore(
            self._metadata_store._base_store, clone_map, cloned._view_config
        )
        cloned._feature_store = OverlayFeatureStore(
            self._feature_store._base_store, clone_map, cloned._view_config
        )

        # Deep-copy delta state so clone mutations don't affect the source.
        cloned._object_store._delta = self._object_store.deep_copy_delta()
        cloned._metadata_store._delta = self._metadata_store.deep_copy_delta()
        cloned._feature_store._delta = self._feature_store.deep_copy_delta()

        # Give the clone a fresh schema_uuid.
        import uuid as _uuid_mod
        cloned.schema_uuid = str(_uuid_mod.uuid4())

        return cloned

    # ------------------------------------------------------------------
    # Materialize
    # ------------------------------------------------------------------

    def materialize(
        self,
        target_path: Optional[Path] = None,
    ) -> BaseStorageBackend:
        """
        Collapse the overlay into a standalone concrete backend.

        Clones the base (optionally row-sliced), then replays all deltas.
        The result is completely independent of this overlay.

        Raises
        ------
        DetachedStateError
            If the overlay is currently detached.
        """
        if self._base is None:
            raise DetachedStateError(
                "Cannot materialize a detached overlay."
            )
        concrete = self._base.clone(
            target_path=target_path, index_map=self._index_map
        )
        self._feature_store.apply_materialized_deltas(concrete)
        self._metadata_store.apply_materialized_deltas(concrete)
        self._object_store.apply_materialized_deltas(concrete)
        return concrete

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def commit(self) -> None:
        """
        Flush all local deltas from this overlay down to the base backend.

        Only delta data is written; cached data is ignored.  After commit,
        all delta stores are cleared; subsequent reads fall through to the
        base (or cache) as normal.

        Raises
        ------
        DetachedStateError
            If the overlay is currently detached.
        """
        if self._base is None:
            raise DetachedStateError("Cannot commit: overlay is detached.")
        self._feature_store.commit(self._base)
        self._metadata_store.commit(self._base)
        self._object_store.commit(self._base)

    # ------------------------------------------------------------------
    # Persistence — overlays persist via materialize().save()
    # ------------------------------------------------------------------

    def save(
        self,
        path: Path,
        object_writer: Optional[Callable[[list, Path], None]] = None,
        **kwargs,
    ) -> None:
        self.materialize().save(path, object_writer=object_writer, **kwargs)

    def _gather_materialized_state(
        self,
        target_path: Optional[Path] = None,
        index_map: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        raise NotImplementedError(
            "OverlayBackend does not expose direct gathered state. "
            "Use materialize() or clone()."
        )

    def save_storage_context(self, path: Path, **kwargs) -> None:
        raise NotImplementedError(
            "OverlayBackend persists via materialize().save()."
        )

    @classmethod
    def load_storage_context(cls, path: Path, **kwargs) -> Dict[str, object]:
        raise NotImplementedError(
            "OverlayBackend cannot be loaded directly from storage context."
        )

    # ------------------------------------------------------------------
    # Name helpers (used by BaseTable.save for the config manifest)
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return self._base.__class__.__name__ if self._base is not None else "OverlayBackend"

    def get_module(self) -> str:
        return (
            self._base.__class__.__module__
            if self._base is not None
            else __name__
        )

    # ------------------------------------------------------------------
    # n_* helpers expected by CompositeStorageBackend.validate()
    # (validate is a no-op for overlays, but the parent may call these)
    # ------------------------------------------------------------------

    def _n_objects(self) -> int:
        return len(self._index_map)

    def _n_metadata_rows(self) -> int:
        return len(self._index_map)

    def _n_feature_rows(self) -> int:
        return len(self._index_map)