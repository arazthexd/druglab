"""
druglab.db.backend.memory
~~~~~~~~~~~~~~~~~~~~~~~~~~
In-memory storage mixins and the EagerMemoryBackend concrete class.

Each mixin handles exactly one data dimension (metadata, objects, features).
They compose via multiple inheritance into ``EagerMemoryBackend``, the
default backend for new tables.

Index normalisation is handled by ``druglab.db.indexing.normalize_row_index``
(via the ``_resolve_idx`` shim defined at the bottom of this module for
internal backward-compatibility).

Persistence
-----------
``EagerMemoryBackend.save()`` writes inside a ``.dlb`` bundle directory:

    <bundle>/
        metadata.parquet      (or metadata.csv if pyarrow absent)
        objects/
            objects.pkl       (single pickle of the entire list)
        features/
            <name>.npy        (one file per feature array)

All reads/writes via ``get_*`` and ``update_*`` methods support ``idx`` 
arguments which is handled by the ``druglab.db.indexing`` module. The same 
interface works identically whether data lives in RAM or on disk.
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .base import (
    BaseStorageBackend,
    BaseMetadataMixin,
    BaseObjectMixin,
    BaseFeatureMixin,
)
from druglab.db.indexing import (
    INDEX_LIKE,
    RowSelection,
    normalize_row_index,
)

# ---------------------------------------------------------------------------
# Backward-compatibility shim
# ---------------------------------------------------------------------------

def _resolve_idx(idx: Optional[INDEX_LIKE], n: int) -> Optional[np.ndarray]:
    """
    Thin compatibility shim that delegates to ``normalize_row_index``.

    All new code should call ``normalize_row_index`` directly.  This shim
    exists so that any internal callers that still reference ``_resolve_idx``
    (e.g. existing tests) continue to work without modification.
    """
    return normalize_row_index(idx, n)


# ---------------------------------------------------------------------------
# MemoryMetadataMixin
# ---------------------------------------------------------------------------


class MemoryMetadataMixin(BaseMetadataMixin):
    """
    In-memory metadata storage mixin utilizing Pandas DataFrames.
    
    This mixin manages tabular metadata natively in RAM. Operations rely on 
    Pandas `.iloc` and `.loc` indexing to enforce strict query pushdown and 
    avoid unnecessary DataFrame copies.
    """

    def __init__(self, metadata: Optional[pd.DataFrame] = None, **kwargs):
        super().__init__(**kwargs)
        self._metadata = metadata if metadata is not None else pd.DataFrame(index=range(len(self)))

    def get_metadata(
        self,
        idx: Optional[INDEX_LIKE] = None,
        cols: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        sel = RowSelection.from_raw(idx, len(self._metadata))

        if sel.is_full:
            df = self._metadata
        else:
            df = self._metadata.iloc[sel.positions]

        if cols is not None:
            if isinstance(cols, str):
                cols = [cols]
            df = df[cols]

        return df.reset_index(drop=True)

    def add_metadata_column(
        self,
        name: str,
        value: Union[pd.Series, np.ndarray, List[Any]],
        idx: Optional[INDEX_LIKE] = None,
        na: Any = None,
        **kwargs
    ) -> None:
        sel = RowSelection.from_raw(idx, len(self._metadata))
        value = np.asarray(value)

        if sel.is_full:
            self._metadata[name] = value
        else:
            resolved = sel.positions
            if na is None and np.issubdtype(value.dtype, np.integer):
                raise ValueError(
                    "na must be provided when populating integer columns"
                )
            if na is None and np.issubdtype(value.dtype, np.floating):
                na = np.nan
            if value.shape[0] != len(resolved):
                raise ValueError(
                    f"Expected {len(resolved)} values, got {value.shape[0]}"
                )
            
            # Create a full array of `na` and populate the targeted indices
            arr = np.full(len(self._metadata), na, dtype=np.asarray(value).dtype)
            arr[resolved] = value
            self._metadata[name] = arr

    def update_metadata(
        self,
        values: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
        idx: Optional[INDEX_LIKE] = None,
        **kwargs
    ) -> None:
        sel = RowSelection.from_raw(idx, len(self._metadata))

        if isinstance(values, pd.DataFrame):
            val_dict = {col: values[col].values for col in values.columns}
        elif isinstance(values, pd.Series):
            if values.name is None:
                raise ValueError("Series must have a name to update metadata.")
            val_dict = {values.name: values.values}
        else:
            val_dict = values

        for col, val in val_dict.items():
            if col not in self._metadata.columns:
                raise KeyError(
                    f"Column '{col}' does not exist in metadata. "
                    "Use add_metadata_column."
                )
            if sel.is_full:
                self._metadata[col] = val
            else:
                self._metadata.iloc[
                    sel.positions,
                    self._metadata.columns.get_loc(col)
                ] = val

    def drop_metadata_columns(
        self,
        cols: Optional[Union[str, List[str]]] = None
    ) -> None:
        if cols is None:
            self._metadata = pd.DataFrame(index=self._metadata.index)
        else:
            if isinstance(cols, str):
                cols = [cols]
            self._metadata = self._metadata.drop(columns=cols)

    def _n_metadata_rows(self) -> int:
        """
        Get the total number of rows in the metadata DataFrame.

        Returns
        -------
        int
            Length of the internal metadata DataFrame. If the DataFrame is empty
            (0 rows, 0 columns), this function returns the length of the object 
            store (i.e., the number of objects stored in the backend).

        Notes
        -----
        The implementation first checks if the DataFrame is empty (0 rows, 0 columns).
        If it is, the function returns the length of the object store. Otherwise, it returns
        the length of the DataFrame.
        """
        n_rows, n_columns = self._metadata.shape
        if n_rows == 0 and n_columns == 0:
            # Genuinely empty backend
            return len(self)
        return n_rows

    def _validate_metadata(self) -> None:
        """
        Validate internal metadata consistency. 
        (No-op for in-memory as Pandas inherently enforces rectangular integrity).
        """
        # In-memory pandas dataframes inherently enforce structural integrity
        pass

    def try_numerize_metadata(self, columns: Optional[List[str]] = None) -> None:
        """
        Attempt to numerize columns in the metadata DataFrame.

        NOTE: This is just a utility function for in-memory metadata saved as pandas dataframes.
        It is not a general utility for any storage backend.

        Parameters
        ----------
        columns : Optional[List[str]], default None
            The column(s) to numerize. If None, numerize all columns.
        """
        if columns is None:
            columns = self._metadata.columns.tolist()
        else:
            if isinstance(columns, str):
                columns = [columns]

        for col in columns:
            if col in self._metadata.columns:
                try:
                    self._metadata[col] = pd.to_numeric(self._metadata[col])
                except (ValueError, TypeError):
                    pass


# ---------------------------------------------------------------------------
# MemoryObjectMixin
# ---------------------------------------------------------------------------

class MemoryObjectMixin(BaseObjectMixin):
    """
    In-memory object storage mixin utilizing a standard Python list.

    Optimized for rapid, transactional point-lookups and vector updates 
    for generic Python objects (e.g., RDKit Mol instances).
    """

    def __init__(self, objects: Optional[List[Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self._objects = objects if objects is not None else []

    def get_objects(self, idx: Optional[INDEX_LIKE] = None) -> Union[Any, List[Any]]:
        """
        Retrieve one or more objects from RAM.

        Parameters
        ----------
        idx : Optional[INDEX_LIKE], default None
            Row selector. 

        Returns
        -------
        Union[Any, List[Any]]
            Returns a single object if `idx` is an integer. Otherwise, returns 
            a list of objects.
        """

        if idx is None:
            return self._objects.copy()

        # Scalar short-circuit: return a single object (not a list)
        if isinstance(idx, (int, np.integer)):
            n = len(self._objects)
            i = int(idx)
            if i >= n or i < -n:
                raise IndexError(
                    f"index {idx} is out of bounds for axis 0 with size {n}"
                )
            index = n + i if i < 0 else i
            return self._objects[index]

        sel = RowSelection.from_raw(idx, len(self._objects))
        return sel.apply_to_list(self._objects)

    def update_objects(
        self,
        objs: Union[Any, List[Any]],
        idx: Optional[INDEX_LIKE] = None,
        **kwargs
    ) -> None:
        """
        Perform an in-place update of stored objects.

        Parameters
        ----------
        objs : Union[Any, List[Any]]
            The object or sequence of objects to insert.
        idx : Optional[INDEX_LIKE], default None
            The specific index/indices to overwrite. If None, the entire 
            internal list is replaced by `objs`.

        Raises
        ------
        ValueError
            If `idx` is a sequence but its length does not match `objs`.
        """

        if idx is None:
            self._objects = list(objs)
            return

        if isinstance(idx, (int, np.integer)):
            n = len(self._objects)
            i = int(idx)
            index = n + i if i < 0 else i
            self._objects[index] = objs
            return

        sel = RowSelection.from_raw(idx, len(self._objects))
        if len(sel.positions) != len(objs):
            raise ValueError(
                "Length of objs sequence must match length of resolved index."
            )
        for i, obj in zip(sel.positions, objs):
            self._objects[i] = obj

    def _n_objects(self) -> int:
        """
        Get the total number of stored objects.

        Returns
        -------
        int
            Length of the internal object list.
        """
        return len(self._objects)

    def _validate_objects(self) -> None:
        """
        Validate internal object consistency.
        (No-op for in-memory list storage).
        """
        pass


# ---------------------------------------------------------------------------
# MemoryFeatureMixin
# ---------------------------------------------------------------------------


class MemoryFeatureMixin(BaseFeatureMixin):
    """
    In-memory feature storage mixin utilizing a dictionary of NumPy arrays.

    Implements query pushdown via NumPy fancy indexing and slicing, returning 
    zero-copy views where possible.
    """

    def __init__(self, features: Optional[Dict[str, np.ndarray]] = None, **kwargs):
        super().__init__(**kwargs)
        self._features = features if features is not None else {}

    def get_feature(self, name: str, idx: Optional[INDEX_LIKE] = None) -> np.ndarray:
        """
        Fetch a feature array or a specific subset of it.

        Parameters
        ----------
        name : str
            The name of the feature to retrieve.
        idx : Optional[INDEX_LIKE], default None
            Row selector. If None, returns the full array. If a slice is 
            provided, attempts to return a memory view.

        Returns
        -------
        np.ndarray
            The requested feature array subset.
        """
        arr = self._features[name]
        sel = RowSelection.from_raw(idx, arr.shape[0])
        return sel.apply_to(arr)

    def update_feature(
        self,
        name: str,
        array: np.ndarray,
        idx: Optional[INDEX_LIKE] = None,
        na: Any = None,
        **kwargs
    ) -> None:
        """
        Add or update a feature array in-place.

        If the feature does not exist and `idx` is provided, a new array is 
        initialized filled with `na` values, and the target indices are populated.

        Parameters
        ----------
        name : str
            The name of the feature to update/create.
        array : np.ndarray
            The incoming feature data.
        idx : Optional[INDEX_LIKE], default None
            Row selector targeting exactly where `array` should be written.
        na : Any, default None
            Fill value for un-targeted rows when creating a new partial array. 
            Defaults to np.nan for floats, 0 for integers.
        """
        if name not in self._features:
            if idx is None:
                if array.shape[0] != self._n_feature_rows():
                    raise ValueError(
                        f"Length of array's first dimension ({array.shape[0]}) must "
                        f"match number of feature rows ({self._n_feature_rows()})."
                    )
                self._features[name] = np.asarray(array).copy()
            else:
                sel = RowSelection.from_raw(idx, self._n_feature_rows() or len(array))
                n_rows = self._n_feature_rows() or (
                    sel.positions.max() + 1 if len(sel.positions) > 0 else 0
                )
                shape = (n_rows, *np.asarray(array).shape[1:])

                if na is None and np.issubdtype(np.asarray(array).dtype, np.floating):
                    na = np.nan
                elif na is None:
                    na = 0

                full_arr = np.full(shape, na, dtype=np.asarray(array).dtype)
                full_arr[sel.positions] = array
                self._features[name] = full_arr
        else:
            if idx is None:
                arr = np.asarray(array)
                if arr.shape[0] != self._features[name].shape[0]:
                    raise ValueError(
                        f"Cannot update feature '{name}': array has {arr.shape[0]} rows "
                        f"but existing feature has {self._features[name].shape[0]} rows."
                    )
                self._features[name] = arr.copy()
            else:
                sel = RowSelection.from_raw(idx, self._features[name].shape[0])
                self._features[name][sel.positions] = array

    def drop_feature(self, name: str) -> None:
        """
        Remove a feature array from memory.

        Parameters
        ----------
        name : str
            The name of the feature to delete.
        """
        del self._features[name]

    def get_feature_names(self) -> List[str]:
        """
        List all stored feature names.

        Returns
        -------
        List[str]
            A list containing the dictionary keys of stored feature arrays.
        """
        return list(self._features.keys())

    def get_feature_shape(self, name: str) -> tuple:
        """
        Quickly inspect the shape of a feature array.

        Parameters
        ----------
        name : str
            The target feature.

        Returns
        -------
        tuple
            The shape of the requested NumPy array.
        """
        return self._features[name].shape


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
    Fully eager, in-memory unified storage backend.

    Inherits and orchestrates Metadata (Pandas), Objects (Lists), and 
    Features (NumPy). This is the default backend for small to medium 
    cheminformatics datasets that fit comfortably in RAM.
    """

    BACKEND_NAME = "EagerMemoryBackend"

    def __len__(self) -> int:
        """
        Return the global, official length of the dataset.

        Returns
        -------
        int
            The authoritative row count (delegated to object count).
        """
        return self._n_objects()

    def create_view(self, indices: Sequence[int]) -> "EagerMemoryBackend":
        """
        Return an independent, deep-copied view restricted to specific indices.
        """
        sel = RowSelection.from_raw(
            np.asarray(indices, dtype=np.intp) if indices else np.array([], dtype=np.intp),
            len(self),
        )

        if sel.is_empty:
            return EagerMemoryBackend()

        new_objects = [copy.deepcopy(self._objects[i]) for i in sel.positions]
        new_metadata = self._metadata.iloc[sel.positions].reset_index(drop=True).copy()
        new_features = {k: v[sel.positions].copy() for k, v in self._features.items()}

        return EagerMemoryBackend(
            objects=new_objects,
            metadata=new_metadata,
            features=new_features,
        )

    def save(self, path: Path, serializer: Optional[Callable] = None) -> None:
        """
        Persist backend state into a '.dlb' bundle directory.

        Writes metadata as Parquet (or CSV), serializes the entire object 
        list into a single pickle file, and saves feature arrays natively 
        as `.npy` files.

        Parameters
        ----------
        path : Path
            The target `.dlb` directory path (pre-created by the Orchestrator).
        serializer : Optional[Callable], default None
            An optional function `(obj) -> bytes` to serialize generic objects.
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

        # stream_v2: Stream all object payloads (serialized or raw) to prevent list-level pickle OOM spikes.
        with open(obj_dir / "objects.pkl", "wb") as f:
            pickle.dump(
                {
                    "format": "stream_v2",
                    "count": len(self._objects),
                    "serialized": serializer is not None,
                },
                f,
            )
            for obj in self._objects:
                payload = serializer(obj) if serializer is not None else obj
                pickle.dump(payload, f)

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
        Reconstruct the backend from a '.dlb' bundle directory.

        Parameters
        ----------
        path : Path
            The location of the `.dlb` bundle.
        deserializer : Optional[Callable], default None
            An optional function `(bytes) -> obj` to reconstruct stored objects.
        mmap_features : bool, default False
            If True, loads `.npy` feature files as memory-mapped arrays rather 
            than fully pulling them into RAM.

        Returns
        -------
        EagerMemoryBackend
            A fully populated instance of the in-memory backend.
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
            with open(obj_path, "rb") as f:
                raw_payload = pickle.load(f)

                if isinstance(raw_payload, dict) and raw_payload.get("format") in {
                    "stream_v1", "stream_v2"
                }:
                    count = int(raw_payload["count"])
                    raw_list = [pickle.load(f) for _ in range(count)]
                    payload_is_serialized = raw_payload.get(
                        "format"
                    ) == "stream_v1" or bool(raw_payload.get("serialized", False))
                else:
                    raw_list = raw_payload
                    payload_is_serialized = deserializer is not None
                if deserializer is not None and payload_is_serialized:
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