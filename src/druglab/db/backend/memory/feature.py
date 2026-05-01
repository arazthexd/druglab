"""In-memory feature store."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from ...indexing import RowSelection
from ..base.stores import BaseFeatureStore

if TYPE_CHECKING:
    from druglab.db.backend.overlay.deltas import FeatureDelta
    from druglab.db.indexing import INDEX_LIKE


__all__ = ["MemoryFeatureStore"]

# Sentinel meaning "this feature did not exist before the transaction"
_MISSING = object()


class MemoryFeatureStore(BaseFeatureStore):
    def __init__(self, features: Optional[Dict[str, np.ndarray]] = None, *, n_rows_hint: int = 0) -> None:
        self._features = dict(features) if features is not None else {}
        self._n_rows_hint = int(n_rows_hint)
        self._journal: Optional[Dict[str, Any]] = None

    def get_feature(self, name: str, idx: Optional["INDEX_LIKE"] = None) -> np.ndarray:
        arr = self._features[name]
        sel = RowSelection.from_raw(idx, arr.shape[0])
        return sel.apply_to(arr)

    def update_feature(self, name: str, array: np.ndarray, idx: Optional["INDEX_LIKE"] = None, na: Any = None) -> None:
        if name not in self._features:
            if idx is None:
                if array.shape[0] != self.n_rows():
                    raise ValueError(
                        f"Length of array's first dimension ({array.shape[0]}) must match number of feature rows ({self.n_rows()})."
                    )
                self._features[name] = np.asarray(array).copy()
                return

            sel = RowSelection.from_raw(idx, self.n_rows() or len(array))
            n_rows = self.n_rows() or (int(sel.positions.max()) + 1 if len(sel.positions) > 0 else 0)
            shape = (n_rows, *np.asarray(array).shape[1:])
            if na is None and np.issubdtype(np.asarray(array).dtype, np.floating):
                na = np.nan
            elif na is None:
                na = 0
            full_arr = np.full(shape, na, dtype=np.asarray(array).dtype)
            full_arr[sel.positions] = array
            self._features[name] = full_arr
            return

        if idx is None:
            arr = np.asarray(array)
            if arr.shape[0] != self._features[name].shape[0]:
                raise ValueError(
                    f"Cannot update feature '{name}': array has {arr.shape[0]} rows but existing feature has {self._features[name].shape[0]} rows."
                )
            self._features[name] = arr.copy()
            return

        sel = RowSelection.from_raw(idx, self._features[name].shape[0])
        self._features[name][sel.positions] = array

    def drop_feature(self, name: str) -> None:
        del self._features[name]

    def get_feature_names(self) -> List[str]:
        return list(self._features.keys())

    def get_feature_shape(self, name: str) -> tuple:
        return self._features[name].shape

    def n_rows(self) -> int:
        if not self._features:
            return self._n_rows_hint
        first_name = next(iter(self._features))
        return self._features[first_name].shape[0]

    def gather_materialized_state(self, index_map: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if index_map is not None:
            return {"features": {k: v[index_map].copy() for k, v in self._features.items()}}
        return {"features": {k: v.copy() for k, v in self._features.items()}}

    def save(self, path: Path) -> None:
        feat_dir = path / "features"
        feat_dir.mkdir(exist_ok=True)
        for name, arr in self._features.items():
            safe_name = name.replace("/", "_").replace("\\", "_")
            np.save(str(feat_dir / f"{safe_name}.npy"), arr)

    @classmethod
    def load(
        cls,
        path: Path,
        mmap_features: bool = False,
    ) -> "MemoryFeatureStore":
        feat_dir = path / "features"
        features: Dict[str, np.ndarray] = {}
        if feat_dir.exists():
            for npy_path in sorted(feat_dir.glob("*.npy")):
                name = npy_path.stem
                if mmap_features:
                    features[name] = np.load(str(npy_path), mmap_mode="r")
                else:
                    features[name] = np.load(str(npy_path), allow_pickle=False)
        return cls(features=features)

    # ------------------------------------------------------------------
    # Transaction protocol
    # ------------------------------------------------------------------

    def begin_transaction(self, delta: "FeatureDelta", index_map: np.ndarray) -> None:
        """
        Snapshot the rows that will be mutated or deleted by *delta*.

        Journal structure per feature::

            {
                "<name>": {
                    "existed": True | False,        # False → new feature; rollback deletes it
                    "indices": np.ndarray | None,   # None when full array is replaced
                    "old_values": np.ndarray | None,
                }
            }
        """
        self._journal = {}

        # Features that will be written (updated or added)
        for name in delta.local:
            if name in self._features:
                self._journal[name] = {
                    "existed": True,
                    "indices": index_map.copy(),
                    "old_values": self._features[name][index_map].copy(),
                }
            else:
                # Brand-new feature; rollback must delete it
                self._journal[name] = {
                    "existed": False,
                    "indices": None,
                    "old_values": None,
                }

        # Features that will be dropped
        for name in delta.deleted:
            if name in self._features:
                self._journal[name] = {
                    "existed": True,
                    "indices": None,           # full array backed up
                    "old_values": self._features[name].copy(),
                }

    def commit_transaction(self, delta: "FeatureDelta", index_map: np.ndarray) -> None:
        """Apply *delta* to the store; journal must have been set by begin_transaction."""
        if self._journal is None:
            raise RuntimeError("Cannot commit without beginning a transaction first.")

        # Drop deleted features
        for name in delta.deleted:
            if name in self._features:
                del self._features[name]

        # Apply updates / additions
        for name, arr in delta.local.items():
            if name in delta.deleted:
                continue
            self.update_feature(name, arr, idx=index_map)

        # NOTE: Do NOT clear self._journal here.
        # apply_deltas() clears it explicitly only after ALL three stores commit
        # successfully.  If a later store fails, rollback_transaction() must
        # still find the journal populated.

    def rollback_transaction(self) -> None:
        """Restore rows that were backed up in begin_transaction."""
        if self._journal is None:
            return
        for name, entry in self._journal.items():
            if not entry["existed"]:
                # Feature was brand new — remove any partial write
                self._features.pop(name, None)
            elif entry["indices"] is None:
                # Full-array backup (e.g. a dropped feature)
                self._features[name] = entry["old_values"]
            else:
                # Partial-row backup
                if name in self._features:
                    self._features[name][entry["indices"]] = entry["old_values"]
        self._journal = None


MemoryFeatureMixin = MemoryFeatureStore