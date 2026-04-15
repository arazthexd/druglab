"""
druglab.db.backends.memory
~~~~~~~~~~~~~~~~~~~~~~~~~~
In-memory storage mixins and the EagerMemoryBackend concrete class.

Each mixin handles exactly one data dimension (metadata, objects, features).
They compose via multiple inheritance into ``EagerMemoryBackend``, the
default backend for new tables.

Persistence
-----------
``EagerMemoryBackend.save()`` writes inside a ``.dlb`` bundle directory:

    <bundle>/
        metadata.parquet      (or metadata.csv if pyarrow absent)
        objects/
            objects.pkl       (single pickle of the entire list)
        features/
            <name>.npy        (one file per feature array)

All reads via ``get_*`` methods support ``idx`` arguments (int, slice,
List[int]) so the same interface works identically whether data lives in
RAM or on disk.
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from druglab.db.backends.base import BaseStorageBackend, INDEX_LIKE


# ---------------------------------------------------------------------------
# Index normalisation helper
# ---------------------------------------------------------------------------

def _resolve_idx(idx: Optional[INDEX_LIKE], n: int) -> Union[np.ndarray, None]:
    """
    Convert ``idx`` (None / int / slice / List[int]) to a NumPy integer array
    of row indices, or return ``None`` to signal "all rows".
    """
    if idx is None:
        return None
    if isinstance(idx, int):
        # Negative indexing support
        if idx < 0:
            idx = n + idx
        return np.array([idx], dtype=np.intp)
    if isinstance(idx, slice):
        return np.arange(*idx.indices(n), dtype=np.intp)
    if isinstance(idx, (list, np.ndarray)):
        arr = np.asarray(idx, dtype=np.intp)
        # Handle negative indices
        arr = np.where(arr < 0, n + arr, arr)
        return arr
    raise TypeError(
        f"idx must be None, int, slice, or List[int]; got {type(idx).__name__}"
    )


# ---------------------------------------------------------------------------
# MemoryMetadataMixin
# ---------------------------------------------------------------------------

class MemoryMetadataMixin:
    """
    Manages tabular metadata in a Pandas DataFrame kept entirely in RAM.

    Relies on ``self._n`` being defined (provided by ``MemoryObjectMixin``
    or a concrete backend's ``__init__``).
    """

    _metadata: pd.DataFrame

    # ------------------------------------------------------------------
    # Metadata API
    # ------------------------------------------------------------------

    def get_metadata(
        self,
        idx: Optional[INDEX_LIKE] = None,
        cols: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """
        Fetch rows and/or columns with strict query pushdown.

        Query pushdown means the index selection happens directly on the
        underlying DataFrame without first materialising a full copy.

        Parameters
        ----------
        idx
            Row selector (None → all rows).
        cols
            Column selector (None → all columns).

        Returns
        -------
        pd.DataFrame
            Always a DataFrame with a reset integer index.
        """
        resolved = _resolve_idx(idx, len(self._metadata))

        # Row selection
        if resolved is None:
            df = self._metadata
        else:
            df = self._metadata.iloc[resolved]

        # Column selection
        if cols is not None:
            if isinstance(cols, str):
                cols = [cols]
            df = df[cols]

        return df.reset_index(drop=True)

    def update_metadata(
        self,
        value: pd.DataFrame,
        idx: Optional[INDEX_LIKE] = None,
    ) -> None:
        """
        Perform a partial, in-place update of the metadata with strict query pushdown.

        Parameters
        ----------
        value : pd.DataFrame
            The new data to insert.
        idx : Optional[INDEX_LIKE], default None
            Row selector. ``None`` → apply to all rows.
            Accepts ``int``, ``slice``, or ``List[int]``.
        """
        resolved = _resolve_idx(idx, len(self._metadata))
        new_cols = [col for col in value.columns if col not in self._metadata.columns]
        if resolved is None:
            self._metadata.update(value)
            if len(new_cols) > 0:
                self._metadata[new_cols] = value[new_cols].values
            return
        self._metadata.iloc[resolved].update(value)
        if len(new_cols) > 0:
            self._metadata[new_cols].iloc[resolved] = value[new_cols].values

    def drop_metadata(
        self,
        cols: Optional[List[str]] = None
    ) -> None:
        """
        Drop metadata columns given as *cols*. If *cols* is None, drop all columns.

        Parameters
        ----------
        cols : Optional[List[str]], default None
            List of columns to drop. If None, drop all columns.
        """
        if cols is None:
            self._metadata = pd.DataFrame(index=self._metadata.index)
        else:
            self._metadata.drop(columns=cols, inplace=True)

    def try_numerize_metadata(
        self,
        columns: Optional[List[str]] = None,
    ) -> None:
        meta = self.get_metadata()
        if columns is None:
            columns = meta.columns
        columns_set = set(columns)

        for col in meta.select_dtypes(include=["object", "string"]).columns:
            if col not in columns_set:
                continue
            try:
                meta[col] = pd.to_numeric(meta[col])
            except (ValueError, TypeError):
                pass

        self._metadata = meta


# ---------------------------------------------------------------------------
# MemoryObjectMixin
# ---------------------------------------------------------------------------

class MemoryObjectMixin:
    """
    Manages a flat Python list of objects (RDKit Mols, reactions, dicts, …).

    This mixin owns ``self._objects`` and ``self._n`` (derived from the list
    length).
    """

    _objects: List[Any]

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._objects)

    # ------------------------------------------------------------------
    # Object API
    # ------------------------------------------------------------------

    def get_objects(self, idx: Optional[INDEX_LIKE] = None) -> Union[Any, List[Any]]:
        """
        Fetch one or multiple objects.

        Parameters
        ----------
        idx
            ``int``        → returns the single object (not wrapped in a list).
            ``slice``      → returns a list.
            ``List[int]``  → returns a list in the specified order.
            ``None``       → returns all objects.
        """
        if idx is None:
            return self._objects.copy()
        
        if isinstance(idx, int):
            n = len(self._objects)
            if idx < 0:
                idx = n + idx
            return self._objects[idx]

        resolved = _resolve_idx(idx, len(self._objects))
        return [self._objects[i] for i in resolved]

    def put_object(self, index: int, obj: Any) -> None:
        """Overwrite the object at *index*."""
        if index < 0:
            index = len(self._objects) + index
        self._objects[index] = obj


# ---------------------------------------------------------------------------
# MemoryFeatureMixin
# ---------------------------------------------------------------------------

class MemoryFeatureMixin:
    """
    Manages named NumPy arrays in a Python dictionary kept entirely in RAM.

    Query pushdown is implemented via NumPy fancy indexing, which avoids
    making a full copy when only a slice is needed (NumPy views for slices).
    """

    _features: Dict[str, np.ndarray]

    # ------------------------------------------------------------------
    # Feature API
    # ------------------------------------------------------------------

    def get_feature(
        self,
        name: str,
        idx: Optional[INDEX_LIKE] = None,
    ) -> np.ndarray:
        """
        Fetch a feature array, pushing the index selection into NumPy.

        For slices NumPy returns a *view* (zero-copy).  For list/int
        indices NumPy performs fancy indexing (copy, but no full-array
        materialisation in the caller).

        Parameters
        ----------
        name
            Feature key.
        idx
            Row selector.  ``None`` → return full array.

        Returns
        -------
        np.ndarray
        """
        arr = self._features[name]
        resolved = _resolve_idx(idx, arr.shape[0])
        if resolved is None:
            return arr
        # Prefer slice-based indexing (returns a view) when possible
        if len(resolved) == 0:
            return arr[0:0]  # empty array with correct dtype/shape
        return arr[resolved]

    def add_feature(self, name: str, array: np.ndarray) -> None:
        """Add or overwrite a feature array."""
        self._features[name] = array

    def drop_feature(self, name: str) -> None:
        """Remove a feature by name."""
        del self._features[name]

    def get_feature_names(self) -> List[str]:
        """Return the list of stored feature keys."""
        return list(self._features.keys())


# ---------------------------------------------------------------------------
# EagerMemoryBackend
# ---------------------------------------------------------------------------

class EagerMemoryBackend(
    MemoryMetadataMixin,
    MemoryObjectMixin,
    MemoryFeatureMixin,
    BaseStorageBackend,
):
    """
    Fully in-memory backend.

    All data lives in RAM:
    * ``_objects``  - Python list
    * ``_metadata`` - Pandas DataFrame
    * ``_features`` - dict of NumPy arrays

    Saves to a ``.dlb`` bundle directory as Parquet + pickle + .npy files.
    Restoring from disk is handled by ``EagerMemoryBackend.load()``.

    Parameters
    ----------
    objects
        Initial object list.
    metadata
        Initial metadata DataFrame.
    features
        Initial feature dictionary.
    """

    BACKEND_NAME = "EagerMemoryBackend"

    def __init__(
        self,
        objects: Optional[List[Any]] = None,
        metadata: Optional[pd.DataFrame] = None,
        features: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self._objects: List[Any] = objects if objects is not None else []
        self._metadata: pd.DataFrame = (
            metadata.reset_index(drop=True) if metadata is not None else pd.DataFrame()
        )
        self._features: Dict[str, np.ndarray] = features if features is not None else {}

    # ------------------------------------------------------------------
    # create_view (query pushdown for subsetting)
    # ------------------------------------------------------------------

    def create_view(self, indices: Sequence[int]) -> "EagerMemoryBackend":
        """
        Return an independent, synchronised view restricted to *indices*.

        All three data stores (objects, metadata, features) are sliced at
        the backend level and deep-copied so mutations do not propagate.

        Parameters
        ----------
        indices
            Ordered integer row indices.

        Returns
        -------
        EagerMemoryBackend
        """
        idx_arr = np.asarray(indices, dtype=np.intp)
        n = len(self._objects)

        # Resolve negatives
        idx_arr = np.where(idx_arr < 0, n + idx_arr, idx_arr)

        # Objects: deep-copy selected items
        new_objects = [copy.deepcopy(self._objects[i]) for i in idx_arr]

        # Metadata: iloc + reset (no full copy needed; .copy() after iloc is minimal)
        new_metadata = (
            self._metadata.iloc[idx_arr].reset_index(drop=True).copy()
        )

        # Features: NumPy fancy-index (copies, but only the selected rows)
        new_features = {k: v[idx_arr].copy() for k, v in self._features.items()}

        return EagerMemoryBackend(
            objects=new_objects,
            metadata=new_metadata,
            features=new_features,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path, serializer: Optional[Callable] = None) -> None:
        """
        Save backend data inside the ``.dlb`` bundle directory at *path*.

        Layout
        ------
        ::

            <path>/
                metadata.parquet   (fallback: metadata.csv)
                objects/
                    objects.pkl    (single pickle of the full list with serializer applied)
                features/
                    <name>.npy
        """
        path = Path(path)

        # --- metadata ---
        if not self._metadata.empty:
            try:
                self._metadata.to_parquet(path / "metadata.parquet", index=False)
            except Exception:
                self._metadata.to_csv(path / "metadata.csv", index=False)

        # --- objects ---
        obj_dir = path / "objects"
        obj_dir.mkdir(exist_ok=True)
        if serializer is not None:
            serialised = [serializer(obj) for obj in self._objects]
        else:
            serialised = self._objects
        (obj_dir / "objects.pkl").write_bytes(pickle.dumps(serialised))

        # --- features ---
        feat_dir = path / "features"
        feat_dir.mkdir(exist_ok=True)
        for name, arr in self._features.items():
            safe_name = name.replace("/", "_").replace("\\", "_")
            np.save(str(feat_dir / f"{safe_name}.npy"), arr)

    @classmethod
    def load(
        cls,
        path: Path,
        deserializer: Optional[Callable] = None,
        mmap_features: bool = False,
    ) -> "EagerMemoryBackend":
        """
        Load backend state from a ``.dlb`` bundle directory.

        Parameters
        ----------
        path
            Bundle directory written by ``save()``.
        deserializer
            Optional callable ``(raw) -> obj`` to reconstruct objects.
        mmap_features
            If ``True``, load feature arrays as memory-mapped ``np.memmap``.

        Returns
        -------
        EagerMemoryBackend
        """
        path = Path(path)

        # --- metadata ---
        parquet_path = path / "metadata.parquet"
        csv_path = path / "metadata.csv"
        if parquet_path.exists():
            metadata = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            metadata = pd.read_csv(csv_path)
        else:
            metadata = pd.DataFrame()

        # --- objects ---
        obj_path = path / "objects" / "objects.pkl"
        if obj_path.exists():
            raw_list = pickle.loads(obj_path.read_bytes())  # noqa: S301
            if deserializer is not None:
                objects = [deserializer(r) for r in raw_list]
            else:
                objects = raw_list
        else:
            objects = []

        # --- features ---
        feat_dir = path / "features"
        features: Dict[str, np.ndarray] = {}
        if feat_dir.exists():
            for npy_path in sorted(feat_dir.glob("*.npy")):
                name = npy_path.stem
                if mmap_features:
                    features[name] = np.load(str(npy_path), mmap_mode="r")
                else:
                    features[name] = np.load(str(npy_path), allow_pickle=False)

        return cls(objects=objects, metadata=metadata, features=features)
    
    def validate(self):
        super().validate()
        n = len(self)
        meta = self.get_metadata()

        if not meta.empty and len(meta) != n and len(meta) != 0:
            raise ValueError(
                f"metadata has {len(meta)} rows but objects has {n}. "
                "They must be the same length."
            )
        # If metadata is empty DataFrame, pad it to the right shape
        if meta.empty and n > 0:
            self._metadata = pd.DataFrame(index=range(n))

        for key in self.get_feature_names():
            arr = self.get_feature(key)
            if arr.shape[0] != n:
                raise ValueError(
                    f"Feature '{key}' has {arr.shape[0]} rows but objects has {n}."
                )