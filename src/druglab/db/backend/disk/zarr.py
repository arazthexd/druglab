"""
druglab.db.backend.disk.zarr_store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Out-of-core feature store backed by Zarr v3.

Design notes
------------
* Every feature is stored as a top-level Zarr array inside a ``zarr.Group``.
  Names that start with ``.`` (e.g. ``.journal``) are reserved for internal
  metadata and are never exposed as feature names.
* 1-D chunking (2 500 rows per chunk along axis 0) keeps cheminformatics
  scatter/gather operations fast without requiring manual schema configuration.
* The append path auto-creates arrays with appropriate chunk shapes; callers
  do not need to pre-declare a schema.
* Orthogonal indexing (``arr.oindex[idx]``) is used for both reads and writes
  because Zarr v3 does not support ``vindex`` for 1-D integer array selections.

Out-of-core rollback journal
-----------------------------
``begin_transaction``  creates a ``.journal`` sub-group that snaps the
affected rows to disk *before* any mutation.

``commit_transaction`` applies the delta, then deletes ``.journal``.

``rollback_transaction`` detects an existing ``.journal``, reads back the
saved values and indices, restores the arrays, and deletes the journal.
This is crash-safe: if the process dies between ``begin`` and ``commit``,
the next ``ZarrFeatureStore`` that opens the same path will find the
``.journal`` still on disk and can call ``rollback_transaction()`` to
restore a consistent state.

-1 safety guardrail
--------------------
Both ``get_feature`` and ``update_feature`` reject any index containing
``-1`` (virtual overlay rows) immediately, before touching disk. This
prevents NumPy / Zarr from silently returning the last row of an array.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

try:
    import zarr
    _ZARR_AVAILABLE = True
except ImportError:
    _ZARR_AVAILABLE = False

from ..base.stores import BaseFeatureStore

if TYPE_CHECKING:
    from druglab.db.backend.overlay.deltas import FeatureDelta
    from druglab.db.indexing import INDEX_LIKE

__all__ = ["ZarrFeatureStore"]

_CHUNK_ROWS = 2_500          # default chunk size along the row axis
_JOURNAL_KEY = ".journal"    # reserved group name


def _require_zarr() -> None:
    if not _ZARR_AVAILABLE:
        raise ImportError(
            "zarr is required for ZarrFeatureStore. "
            "Install it with: pip install zarr"
        )


def _validate_no_virtual(idx: np.ndarray) -> None:
    """Raise IndexError if any index is -1 (un-resolved overlay append)."""
    if idx is not None and np.any(idx < 0):
        raise IndexError(
            "Base stores cannot process virtual indices (-1). "
            "Ensure the OverlayBackend resolved all appended rows via "
            "commit() before calling the base store."
        )


class ZarrFeatureStore(BaseFeatureStore):
    """
    Out-of-core feature store backed by a ``zarr.Group``.

    Parameters
    ----------
    group : zarr.Group
        An open, writable Zarr group.  All feature arrays are created as
        direct children of this group.  Pass ``zarr.open_group(path, mode='w')``
        for a brand-new store or ``zarr.open_group(path, mode='r+')`` to
        attach to an existing one.
    """

    def __init__(self, group: "zarr.Group") -> None:
        _require_zarr()
        self._group = group
        self._journal: Optional["zarr.Group"] = None
        # Recover from a crash if a journal already exists on disk.
        if _JOURNAL_KEY in self._group:
            # Do NOT auto-rollback here — let the caller decide.
            # But keep the reference so rollback_transaction() can act.
            self._journal = self._group[_JOURNAL_KEY]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _feature_names_raw(self) -> List[str]:
        """All keys in the group that don't start with '.'."""
        return [k for k in self._group.keys() if not k.startswith(".")]

    def _require_array(self, name: str) -> "zarr.Array":
        if name not in self._group:
            raise KeyError(f"Feature '{name}' does not exist in this ZarrFeatureStore.")
        return self._group[name]

    # ------------------------------------------------------------------
    # BaseFeatureStore abstract interface
    # ------------------------------------------------------------------

    def get_feature(self, name: str, idx: Optional["INDEX_LIKE"] = None) -> np.ndarray:
        arr = self._require_array(name)
        if idx is None:
            return arr[:]
        idx_np = np.asarray(idx, dtype=np.intp)
        _validate_no_virtual(idx_np)
        return arr.oindex[idx_np]

    def update_feature(
        self,
        name: str,
        array: np.ndarray,
        idx: Optional["INDEX_LIKE"] = None,
        na: Any = None,
    ) -> None:
        array = np.asarray(array)
        if name not in self._group:
            # Auto-create on first write.
            if idx is None:
                shape = array.shape
            else:
                idx_np = np.asarray(idx, dtype=np.intp)
                _validate_no_virtual(idx_np)
                n_rows = self.n_rows() or (int(idx_np.max()) + 1 if idx_np.size else 0)
                shape = (n_rows,) + array.shape[1:]

            chunks = (_CHUNK_ROWS,) + array.shape[1:]
            new_arr = self._group.create_array(
                name, shape=shape, chunks=chunks, dtype=array.dtype
            )
            if idx is None:
                new_arr[:] = array
            else:
                if na is None:
                    na = np.nan if np.issubdtype(array.dtype, np.floating) else 0
                new_arr[:] = np.full(shape, na, dtype=array.dtype)
                new_arr.oindex[idx_np] = array
            return

        zarr_arr = self._group[name]
        if idx is None:
            if array.shape[0] != zarr_arr.shape[0]:
                raise ValueError(
                    f"Cannot update feature '{name}': array has {array.shape[0]} rows "
                    f"but existing feature has {zarr_arr.shape[0]} rows."
                )
            zarr_arr[:] = array
        else:
            idx_np = np.asarray(idx, dtype=np.intp)
            _validate_no_virtual(idx_np)
            zarr_arr.oindex[idx_np] = array

    def drop_feature(self, name: str) -> None:
        if name not in self._group:
            raise KeyError(f"Feature '{name}' does not exist.")
        del self._group[name]

    def get_feature_names(self) -> List[str]:
        return self._feature_names_raw()

    def get_feature_shape(self, name: str) -> tuple:
        return tuple(self._require_array(name).shape)

    def n_rows(self) -> int:
        names = self._feature_names_raw()
        if not names:
            return 0
        return self._group[names[0]].shape[0]

    def append(self, data: Dict[str, np.ndarray]) -> None:
        """
        Append new rows to the store, auto-creating arrays if needed.

        Each array in *data* must have the same number of new rows.
        Existing features not present in *data* are NOT extended — callers
        must ensure all features are provided to maintain row-count parity.
        """
        for name, new_arr in data.items():
            new_arr = np.asarray(new_arr)
            if name not in self._group:
                chunks = (_CHUNK_ROWS,) + new_arr.shape[1:]
                new_zarr = self._group.create_array(
                    name, shape=new_arr.shape, chunks=chunks, dtype=new_arr.dtype
                )
                new_zarr[:] = new_arr
            else:
                existing = self._group[name]
                old_n = existing.shape[0]
                new_n = old_n + new_arr.shape[0]
                new_shape = (new_n,) + existing.shape[1:]
                existing.resize(new_shape)
                existing[old_n:new_n] = new_arr

    # ------------------------------------------------------------------
    # Materialized-state gathering
    # ------------------------------------------------------------------

    def gather_materialized_state(
        self, index_map: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        # ZarrFeatureStore does not support in-memory materialization;
        # callers should clone via save/load or use OverlayBackend.materialize().
        raise NotImplementedError(
            "ZarrFeatureStore does not support in-memory state gathering. "
            "Use ZarrFeatureStore.save() to persist and load() to re-open."
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path, **kwargs) -> None:
        """
        No-op if this store's group already lives under *path*.

        If the group lives elsewhere, copy all arrays into a new group at
        ``path / "features.zarr"``.
        """
        import zarr
        expected = Path(path) / "features.zarr"
        store_path = Path(str(self._group.store.path)) if hasattr(self._group.store, "path") else None
        if store_path is not None and Path(store_path) == expected:
            return  # Already in the right place, nothing to do.
        # Copy to bundle location.
        target = zarr.open_group(str(expected), mode="w")
        for name in self._feature_names_raw():
            arr = self._group[name]
            data = arr[:]
            chunks = arr.chunks
            new_arr = target.create_array(name, shape=data.shape, chunks=chunks, dtype=data.dtype)
            new_arr[:] = data

    @classmethod
    def load(cls, path: Path, **kwargs) -> "ZarrFeatureStore":
        _require_zarr()
        import zarr
        zarr_path = Path(path) / "features.zarr"
        if not zarr_path.exists():
            # Return an empty store backed by a new in-memory group.
            return cls(zarr.open_group(str(zarr_path), mode="w"))
        return cls(zarr.open_group(str(zarr_path), mode="r+"))

    # ------------------------------------------------------------------
    # Out-of-core rollback journal (transaction protocol)
    # ------------------------------------------------------------------

    def begin_transaction(self, delta: "FeatureDelta", index_map: np.ndarray) -> None:
        """
        Snapshot rows that will be mutated or deleted to a on-disk journal.

        Journal layout inside ``.journal`` group::

            <feature_name>          : array of old values for the affected rows
            <feature_name>_indices  : the base-row positions that were snapped
        """
        import zarr
        # Overwrite any stale journal (e.g. from a previous crash).
        if _JOURNAL_KEY in self._group:
            del self._group[_JOURNAL_KEY]
        journal = self._group.require_group(_JOURNAL_KEY)
        self._journal = journal

        # Features that will be written (updated or added)
        for name in delta.local:
            if name in self._group:
                old_values = self._group[name].oindex[index_map]
                journal.create_array(name, shape=old_values.shape, chunks=old_values.shape, dtype=old_values.dtype)
                journal[name][:] = old_values
                journal.create_array(
                    f"{name}_indices", shape=index_map.shape, chunks=index_map.shape, dtype=index_map.dtype
                )
                journal[f"{name}_indices"][:] = index_map
                # Mark as pre-existing so rollback restores rather than deletes
                journal.attrs[f"{name}_existed"] = True
            else:
                # Brand-new feature — rollback must drop it.
                # Store a zero-byte sentinel array so this name appears in journal.keys().
                sentinel = journal.create_array(name, shape=(0,), dtype=np.uint8)
                journal.attrs[f"{name}_existed"] = False

        # Features that will be dropped
        for name in delta.deleted:
            if name in self._group:
                full = self._group[name][:]
                chunks = (_CHUNK_ROWS,) + full.shape[1:]
                backed = journal.create_array(name, shape=full.shape, chunks=chunks, dtype=full.dtype)
                backed[:] = full
                journal.attrs[f"{name}_existed"] = True
                journal.attrs[f"{name}_full_backup"] = True

    def commit_transaction(self, delta: "FeatureDelta", index_map: np.ndarray) -> None:
        """Apply the delta to disk; journal must already exist from begin_transaction."""
        if self._journal is None:
            raise RuntimeError("Cannot commit without beginning a transaction first.")

        # Drop deleted features
        for name in delta.deleted:
            if name in self._group:
                del self._group[name]

        # Apply updates / additions (reuse update_feature which handles auto-create)
        for name, arr in delta.local.items():
            if name in delta.deleted:
                continue
            self.update_feature(name, arr, idx=index_map)

        # NOTE: Do NOT delete the journal here.
        # CompositeStorageBackend.apply_deltas() clears _journal only after ALL
        # three stores commit successfully.  We rely on _clear_journal() below.

    def rollback_transaction(self) -> None:
        """Restore any rows backed up in begin_transaction from the on-disk journal."""
        journal = self._journal
        if journal is None:
            # Check if a crash left a journal on disk.
            if _JOURNAL_KEY in self._group:
                journal = self._group[_JOURNAL_KEY]
            else:
                return

        journal_attrs = dict(journal.attrs)
        # Collect feature names from the journal (skip _indices keys)
        journaled_names = [k for k in journal.keys() if not k.endswith("_indices")]

        for name in journaled_names:
            existed = journal_attrs.get(f"{name}_existed", True)
            is_full_backup = journal_attrs.get(f"{name}_full_backup", False)

            if not existed:
                # Feature was brand-new; remove any partial write.
                if name in self._group:
                    del self._group[name]
            elif is_full_backup:
                # Full-array backup (feature was going to be dropped).
                backed_up = journal[name][:]
                if name in self._group:
                    del self._group[name]
                chunks = (_CHUNK_ROWS,) + backed_up.shape[1:]
                new_arr = self._group.create_array(name, shape=backed_up.shape, chunks=chunks, dtype=backed_up.dtype)
                new_arr[:] = backed_up
            else:
                # Partial-row backup — restore just the affected rows.
                indices_key = f"{name}_indices"
                if name in self._group and indices_key in journal:
                    old_values = journal[name][:]
                    indices = journal[indices_key][:]
                    self._group[name].oindex[indices] = old_values

        # Delete the journal from disk.
        if _JOURNAL_KEY in self._group:
            del self._group[_JOURNAL_KEY]
        self._journal = None

    def _clear_journal(self) -> None:
        """Called by CompositeStorageBackend after a successful commit."""
        if _JOURNAL_KEY in self._group:
            del self._group[_JOURNAL_KEY]
        self._journal = None