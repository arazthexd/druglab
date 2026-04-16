"""
druglab.db.backend.memory
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

All reads/writes via ``get_*`` and ``update_*`` methods support ``idx`` 
arguments (int, slice, List[int], np.ndarray) so the same interface works 
identically whether data lives in RAM or on disk.
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
    INDEX_LIKE,
    BaseMetadataMixin,
    BaseObjectMixin,
    BaseFeatureMixin
)

# ---------------------------------------------------------------------------
# Index normalisation helper
# ---------------------------------------------------------------------------

def _resolve_idx(idx: Optional[INDEX_LIKE], n: int) -> Union[np.ndarray, None]:
    """
    Normalize various index representations into a standard NumPy integer array.

    Handles standard integers, slices, lists of integers, and boolean masks.
    Automatically resolves negative indices relative to the backend length `n`.

    Parameters
    ----------
    idx : Optional[INDEX_LIKE]
        The user-provided index (int, slice, list, or ndarray).
    n : int
        The total number of rows/objects currently in the domain to resolve 
        negative indices against.

    Returns
    -------
    Union[np.ndarray, None]
        A 1D numpy array of integer indices (dtype=np.intp), or None if `idx` 
        was None (signaling "all rows").

    Raises
    ------
    TypeError
        If the provided index type is unsupported.
    """
    if idx is None:
        return None
    if isinstance(idx, int):
        # Negative indexing support
        if idx < 0:
            if idx + n < 0:
                raise IndexError(
                    f"index {idx} is out of bounds for axis 0 with size {n}"
                )
            idx = n + idx
        return np.array([idx], dtype=np.intp)
    if isinstance(idx, slice):
        return np.arange(*idx.indices(n), dtype=np.intp)
    if isinstance(idx, (list, np.ndarray)):
        arr = np.asarray(idx)
        # Handle boolean masks
        if arr.dtype == bool:
            return np.where(arr)[0].astype(np.intp)
        
        arr = arr.astype(np.intp)
        # Handle negative indices
        arr = np.where(arr < 0, n + arr, arr)
        if np.any((arr < 0) | (arr >= n)):
            raise IndexError(
                f"index {arr} is out of bounds for axis 0 with size {n}"
            )
        return arr
    
    raise TypeError(
        f"idx must be None, int, slice, List[int], or np.ndarray; got {type(idx).__name__}"
    )


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
        super().__init__(**kwargs)  # Passes leftovers down the chain
        self._metadata = metadata if metadata is not None else pd.DataFrame(index=range(len(self)))

    def get_metadata(
        self,
        idx: Optional[INDEX_LIKE] = None,
        cols: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """
        Fetch a subset of the metadata DataFrame.

        Parameters
        ----------
        idx : Optional[INDEX_LIKE], default None
            Row selector. None indicates all rows.
        cols : Optional[Union[str, List[str]]], default None
            Column selector. None indicates all columns.

        Returns
        -------
        pd.DataFrame
            A new DataFrame containing the requested subset. The index is 
            always reset to standard integers.
        """
        resolved = _resolve_idx(idx, len(self._metadata))

        if resolved is None:
            df = self._metadata
        else:
            df = self._metadata.iloc[resolved]

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
        """
        Add a new metadata column, optionally populating only specific rows.

        Pandas indices on incoming Series are explicitly ignored; data is 
        aligned strictly by position.

        Parameters
        ----------
        name : str
            The name of the new column.
        value : Union[pd.Series, np.ndarray, List[Any]]
            The data to populate the new column.
        idx : Optional[INDEX_LIKE], default None
            Specific rows to populate with `value`. If provided, all other 
            rows will be filled with `na`.
        na : Any, default None
            The fill value used for rows not included in `idx`.
        """
        resolved = _resolve_idx(idx, len(self._metadata))
        value = np.asarray(value)
        
        if resolved is None:
            # Positional assignment ignoring Pandas index
            self._metadata[name] = value
        else:
            if na is None and value.dtype == int:
                raise ValueError(
                    "na must be provided when populating integer columns"
                )
            if na is None and value.dtype == float:
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
        """
        Perform an in-place update of existing metadata columns.

        Parameters
        ----------
        values : Union[pd.DataFrame, pd.Series, Dict[str, Any]]
            The new values to insert. Keys/column names must match existing 
            metadata columns.
        idx : Optional[INDEX_LIKE], default None
            Specific rows to update. None applies the update to all rows.

        Raises
        ------
        KeyError
            If any column in `values` does not already exist in the metadata.
        ValueError
            If a Series without a name is provided.
        """
        resolved = _resolve_idx(idx, len(self._metadata))
        
        # Standardize inputs to a dictionary of numpy arrays
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
                raise KeyError(f"Column '{col}' does not exist in metadata. Use add_metadata_column.")
            
            if resolved is None:
                self._metadata[col] = val
            else:
                self._metadata.iloc[resolved, self._metadata.columns.get_loc(col)] = val

    def drop_metadata_columns(self, cols: Optional[Union[str, List[str]]] = None) -> None:
        """
        Remove metadata columns from the internal DataFrame.

        Parameters
        ----------
        cols : Optional[Union[str, List[str]]], default None
            The column(s) to remove. If None, the entire DataFrame is wiped 
            and replaced with an empty DataFrame maintaining the same index.
        """
        if cols is None:
            self._metadata = pd.DataFrame(index=self._metadata.index)
        else:
            if isinstance(cols, str):
                cols = [cols]
            self._metadata.drop(columns=cols, inplace=True)

    def _n_metadata_rows(self) -> int:
        """
        Get the total number of rows in the metadata DataFrame.

        If case metadata is empty (no access to row number), we return `len(self)`, assuming
        that the number of rows is equal to the number of objects or any number that the
        final storage backend will provide (refer to the backend in use for details).

        Returns
        -------
        int
            Length of the internal metadata DataFrame.
        """
        if self._metadata.empty:
            return len(self)
        return len(self._metadata)

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
                    pass # Leave untouched if it can't be safely converted

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
        super().__init__(**kwargs)  # Passes leftovers down the chain
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
        
        if isinstance(idx, int):
            n = len(self._objects)
            index = n + idx if idx < 0 else idx
            return self._objects[index]

        resolved = _resolve_idx(idx, len(self._objects))
        return [self._objects[i] for i in resolved]

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
            
        if isinstance(idx, int):
            n = len(self._objects)
            index = n + idx if idx < 0 else idx
            self._objects[index] = objs
            return
            
        resolved = _resolve_idx(idx, len(self._objects))
        if len(resolved) != len(objs):
            raise ValueError("Length of objs sequence must match length of resolved index.")
            
        for i, obj in zip(resolved, objs):
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
        super().__init__(**kwargs)  # Passes leftovers down the chain
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
        resolved = _resolve_idx(idx, arr.shape[0])
        
        if resolved is None:
            return arr
            
        if len(resolved) == 0:
            return arr[0:0] 
            
        return arr[resolved]

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
                self._features[name] = np.asarray(array)
            else:
                resolved = _resolve_idx(idx, self._n_feature_rows() or len(array))
                n_rows = self._n_feature_rows() or (resolved.max() + 1 if len(resolved) > 0 else 0)
                shape = (n_rows, *np.asarray(array).shape[1:])
                
                # Default numeric 'na' to np.nan if not provided and dtype is float
                if na is None and np.issubdtype(np.asarray(array).dtype, np.floating):
                    na = np.nan
                elif na is None:
                    na = 0
                    
                full_arr = np.full(shape, na, dtype=np.asarray(array).dtype)
                full_arr[resolved] = array
                self._features[name] = full_arr
        else:
            if idx is None:
                arr = np.asarray(array)
                if arr.shape[0] != self._features[name].shape[0]:
                    raise ValueError(
                        f"Cannot update feature '{name}': array has {arr.shape[0]} rows "
                        f"but existing feature has {self._features[name].shape[0]} rows."
                    )
                self._features[name] = arr
            else:
                resolved = _resolve_idx(idx, self._features[name].shape[0])
                self._features[name][resolved] = array

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

        Used internally by table subsetting to ensure mutations in the child 
        table do not affect the parent table's data in RAM.

        Parameters
        ----------
        indices : Sequence[int]
            The exact row numbers to extract into the new backend.

        Returns
        -------
        EagerMemoryBackend
            A completely new backend instance containing only the requested rows.
        """
        idx_arr = _resolve_idx(indices, len(self))

        if idx_arr is None or len(idx_arr) == 0:
            return EagerMemoryBackend()

        new_objects = [copy.deepcopy(self._objects[i]) for i in idx_arr]
        new_metadata = self._metadata.iloc[idx_arr].reset_index(drop=True).copy()
        new_features = {k: v[idx_arr].copy() for k, v in self._features.items()}

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
        if serializer is not None:
            with open(obj_dir / "objects.pkl", "wb") as f:
                pickle.dump({"format": "stream_v1", "count": len(self._objects)}, f)
                for obj in self._objects:
                    pickle.dump(serializer(obj), f)
        else:
            with open(obj_dir / "objects.pkl", "wb") as f:
                pickle.dump(self._objects, f)

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
                if isinstance(raw_payload, dict) and raw_payload.get("format") == "stream_v1":
                    count = int(raw_payload["count"])
                    raw_list = [pickle.load(f) for _ in range(count)]
                else:
                    raw_list = raw_payload
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