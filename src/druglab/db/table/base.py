"""
druglab.db.base
~~~~~~~~~~~~~~~
BaseTable: abstract orchestrator enforcing the four-property contract.
HistoryEntry: immutable record written by pipe blocks.

Architecture
------------
``BaseTable`` now delegates all data storage to a ``BaseStorageBackend``.
The default backend is ``EagerMemoryBackend`` (fully in-memory).  Future
backends (SQLite, Zarr, HDF5) can be swapped in without touching this class.

Advanced multi-axis indexing with strict query pushdown::

    META = 'metadata'
    OBJ  = 'object'
    FEAT = 'feature'

    table[0:10]                         # returns a new table (10 rows)
    table[FEAT, 'fps', 0:100]           # pushes slice to backend
    table[META, ['MolWt', 'LogP'], 5]   # pushes col+row request to backend
    table[OBJ, 3]                       # single object from backend

Backwards-compatible dot-notation still works:
    table.metadata          -> pd.DataFrame (full)
    table.objects           -> List[T]      (full)
    table.features          -> dict         (full)
"""

from __future__ import annotations

import re
import copy
import json
import pickle
import shutil
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

# Updated imports based on standard project structure
from ..backend import EagerMemoryBackend, BaseStorageBackend, INDEX_LIKE

# ---------------------------------------------------------------------------
# Type variables
# ---------------------------------------------------------------------------

OT = TypeVar("OT") # object type
MT = TypeVar("MT", bound=pd.DataFrame) # metadata type (not used yet)
FT = TypeVar("FT", bound=np.ndarray) # feature type (not used yet)
BT = TypeVar("BT", bound=BaseStorageBackend) # backend type

# ---------------------------------------------------------------------------
# Multi-axis index axis constants
# ---------------------------------------------------------------------------

META = M = "metadata"
OBJ  = O = "object"
FEAT = F = "feature"

# ---------------------------------------------------------------------------
# HistoryEntry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HistoryEntry:
    """
    Immutable record appended by a pipe block after it runs.

    Fields
    ------
    block_name   : fully-qualified class name of the block
    config       : JSON-serialisable snapshot of the block's config
    timestamp    : UTC ISO-8601 string
    rows_in      : number of rows in the table when the block started
    rows_out     : number of rows in the table when the block finished
    extra        : optional free-form dict for block-specific metadata
    """

    block_name: str
    config: Dict[str, Any]
    timestamp: str
    rows_in: int
    rows_out: int
    extra: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def now(
        block_name: str,
        config: Dict[str, Any],
        rows_in: int,
        rows_out: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "HistoryEntry":
        return HistoryEntry(
            block_name=block_name,
            config=config,
            timestamp=datetime.now(timezone.utc).isoformat(),
            rows_in=rows_in,
            rows_out=rows_out,
            extra=extra or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "HistoryEntry":
        return HistoryEntry(**d)


# ---------------------------------------------------------------------------
# BaseTable
# ---------------------------------------------------------------------------

class BaseTable(ABC, Generic[OT]): # TODO: add BT
    """
    Abstract base class for all DrugLab table types.

    Data is delegated to a ``BaseStorageBackend`` (default: EagerMemoryBackend).
    All public properties proxy to the backend, maintaining backwards
    compatibility with the previous attribute-based API.

    Invariant (always enforced):
        len(objects) == len(metadata)
        features[k].shape[0] == len(objects)  for every key k

    Subclasses must implement:
        _serialize_object(obj)   -> bytes
        _deserialize_object(raw) -> OT
        _object_type_name()      -> str

    Multi-axis indexing (strict query pushdown)::

        table[0:5]                       # new 5-row table
        table[FEAT, 'fps', 0:10]         # ndarray (FT), only rows 0-9 loaded
        table[META, ['MolWt'], 0]        # single-row DataFrame (MT)
        table[OBJ, 3]                    # single object (OT)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        objects: Optional[List[OT]] = None,
        metadata: Optional[pd.DataFrame] = None,
        features: Optional[Dict[str, np.ndarray]] = None,
        history: Optional[List[HistoryEntry]] = None,
        *, 
        _backend: Optional[BT] = None,
    ) -> None:
        # Allow callers to pass a pre-built backend (used internally by load())
        if _backend is not None:
            self._backend = _backend
        else:
            # Build the default EagerMemoryBackend from raw arguments
            obj_list  = objects  if objects  is not None else []
            meta_df   = metadata if metadata is not None else pd.DataFrame(index=range(len(obj_list)))
            feat_dict = features if features is not None else {}

            if not isinstance(meta_df, pd.DataFrame):
                meta_df = pd.DataFrame(meta_df)

            meta_df = meta_df.reset_index(drop=True)

            self._backend = EagerMemoryBackend(
                objects=obj_list,
                metadata=meta_df,
                features=feat_dict,
            )

        self._history: List[HistoryEntry] = history if history is not None else []
        self._validate()

    # ------------------------------------------------------------------
    # Property wrappers
    # ------------------------------------------------------------------

    @property
    def backend(self) -> BT:
        return self._backend

    @property
    def objects(self) -> List[OT]:
        """Full list of objects (proxied from backend)."""
        return self._backend.get_objects()  # direct access for performance
    
    @objects.setter
    def objects(self, objs: List[OT]):
        self._mutate_with_validation(
            domain="object",
            method="objects.setter",
            mutate=lambda: self._backend.set_objects(objs),
        )

    @property
    def metadata(self) -> pd.DataFrame:
        """Full metadata DataFrame (proxied from backend)."""
        return self._backend.get_metadata()
    
    @metadata.setter
    def metadata(self, meta: pd.DataFrame):
        self._mutate_with_validation(
            domain="metadata",
            method="metadata.setter",
            mutate=lambda: self._backend.set_metadata(meta),
        )

    @property
    def metadata_columns(self) -> List[str]:
        """List of all metadata column names."""
        return self._backend.get_metadata().columns.tolist()

    @property
    def features(self) -> Dict[str, np.ndarray]:
        """Full feature dictionary (proxied from backend)."""
        return self._backend.get_features() # Updated to use the batch reader
    
    @features.setter
    def features(self, feats: Dict[str, np.ndarray]):
        # Clear existing features, then batch update new ones to simulate a total reset
        def _mutate() -> None:
            for key in self._backend.get_feature_names():
                self._backend.drop_feature(key)
            self._backend.update_features(feats)

        self._mutate_with_validation(
            domain="feature",
            method="features.setter",
            mutate=_mutate,
        )
    
    @property
    def feature_names(self) -> List[str]:
        """List of all feature names."""
        return self._backend.get_feature_names()

    @property
    def history(self) -> List[HistoryEntry]:
        return self._history

    @property
    def n(self) -> int:
        """Number of rows / objects in the table."""
        return len(self._backend)

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        feat_names = self._backend.get_feature_names()
        feat_summary = ", ".join(
            f"{k}:{self._backend.get_feature_shape(k)}" # Updated to use shape getter
            for k in feat_names
        )
        return (
            f"{self.__class__.__name__}("
            f"n={self.n}, "
            f"metadata_cols={self._backend.get_metadata().columns.tolist()}, "
            f"features=[{feat_summary}])"
        )

    # ------------------------------------------------------------------
    # Invariant enforcement
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        # Backend natively orchestrates all dimensional checks now!
        self._backend.validate()

    def _run_post_mutation_validation(
        self,
        *,
        domain: str,
        method: str,
    ) -> None:
        """
        Run strict backend-level validation after a domain mutation.

        The error message always includes the domain/method boundary that
        introduced the mismatch for easier debugging.
        """
        try:
            self._backend.validate()
        except Exception as exc:
            raise ValueError(
                f"Post-mutation validation failed in domain='{domain}', method='{method}': {exc}"
            ) from exc

    def _mutate_with_validation(
        self,
        *,
        domain: str,
        method: str,
        mutate: Callable[[], None],
    ) -> None:
        """
        Execute a mutating backend operation and enforce final validation.
        """
        try:
            mutate()
        except Exception as exc:
            raise type(exc)(
                f"Mutation failed in domain='{domain}', method='{method}': {exc}"
            ) from exc

        self._run_post_mutation_validation(domain=domain, method=method)

    # ------------------------------------------------------------------
    # Abstract interface (subclasses implement object serialisation)
    # ------------------------------------------------------------------

    @abstractmethod
    def _serialize_object(self, obj: OT) -> bytes:
        """Serialise a single object to bytes for disk storage."""

    @abstractmethod
    def _deserialize_object(self, raw: bytes) -> OT:
        """Deserialise bytes back to an object."""

    @abstractmethod
    def _object_type_name(self) -> str:
        """Short human-readable name for the object type (e.g. 'Mol')."""

    # ------------------------------------------------------------------
    # Backend Delegation API
    # ------------------------------------------------------------------

    def get_metadata(
        self,
        idx: Optional[INDEX_LIKE] = None,
        cols: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """
        Fetch a subset of the metadata DataFrame.
        For full documentation, please refer to the active storage backend (e.g., `BaseStorageBackend`).
        """
        meta = self._backend.get_metadata(idx=idx, cols=cols)
        return meta

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
        For full documentation, please refer to the active storage backend.
        """
        self._mutate_with_validation(
            domain="metadata",
            method="add_metadata_column",
            mutate=lambda: self._backend.add_metadata_column(name=name, value=value, idx=idx, na=na, **kwargs),
        )

    def add_metadata_columns(
        self,
        columns: Dict[str, Union[pd.Series, np.ndarray, List[Any]]],
        idx: Optional[INDEX_LIKE] = None,
        na: Any = None,
        **kwargs
    ) -> None:
        """
        Add multiple new metadata columns simultaneously.
        For full documentation, please refer to the active storage backend.
        """
        self._mutate_with_validation(
            domain="metadata",
            method="add_metadata_columns",
            mutate=lambda: self._backend.add_metadata_columns(columns=columns, idx=idx, na=na, **kwargs),
        )

    def update_metadata(
        self,
        values: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
        idx: Optional[INDEX_LIKE] = None,
        **kwargs
    ) -> None:
        """
        Perform a partial, in-place update of *existing* metadata columns.
        For full documentation, please refer to the active storage backend.
        """
        self._mutate_with_validation(
            domain="metadata",
            method="update_metadata",
            mutate=lambda: self._backend.update_metadata(values=values, idx=idx, **kwargs),
        )

    def drop_metadata_columns(
        self,
        cols: Optional[Union[str, List[str]]] = None
    ) -> None:
        """
        Remove specific metadata columns, or all columns if None.
        For full documentation, please refer to the active storage backend.
        """
        self._mutate_with_validation(
            domain="metadata",
            method="drop_metadata_columns",
            mutate=lambda: self._backend.drop_metadata_columns(cols=cols),
        )

    def get_feature(
        self,
        name: str,
        idx: Optional[INDEX_LIKE] = None,
    ) -> np.ndarray:
        """
        Fetch a feature array or a specific subset of it.
        For full documentation, please refer to the active storage backend.
        """
        return self._backend.get_feature(name=name, idx=idx)

    def get_features(
        self,
        names: Optional[List[str]] = None,
        idx: Optional[INDEX_LIKE] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Fetch multiple feature arrays or specific subsets of them.
        For full documentation, please refer to the active storage backend.
        """
        return self._backend.get_features(names=names, idx=idx)

    def update_feature(
        self,
        name: str,
        array: np.ndarray,
        idx: Optional[INDEX_LIKE] = None,
        na: Any = None,
        **kwargs
    ) -> None:
        """
        Add or perform a partial, in-place update of a feature array.
        For full documentation, please refer to the active storage backend.
        """
        # Strict validation: Only allow alphanumeric characters and underscores
        if not re.match(r"^[a-zA-Z0-9_=]+$", name):
            raise ValueError(
                f"Invalid feature name '{name}'. Feature names must be alphanumeric "
                f"and contain no spaces or special characters (e.g., use 'ecfp_4' instead of 'ecfp/4')."
            )
            
        self._mutate_with_validation(
            domain="feature",
            method="update_feature",
            mutate=lambda: self._backend.update_feature(name=name, array=array, idx=idx, na=na, **kwargs),
        )

    def update_features(
        self,
        arrays: Dict[str, np.ndarray],
        idx: Optional[INDEX_LIKE] = None,
        na: Any = None,
        **kwargs
    ) -> None:
        """
        Add or perform a partial, in-place update of multiple feature arrays.
        For full documentation, please refer to the active storage backend.
        """
        self._mutate_with_validation(
            domain="feature",
            method="update_features",
            mutate=lambda: self._backend.update_features(arrays=arrays, idx=idx, na=na, **kwargs),
        )

    def drop_feature(self, name: str) -> None:
        """
        Remove a feature array from the table.
        For full documentation, please refer to the active storage backend.
        """
        self._mutate_with_validation(
            domain="feature",
            method="drop_feature",
            mutate=lambda: self._backend.drop_feature(name=name),
        )

    def get_feature_shape(self, name: str) -> Tuple:
        """
        Inspect the shape of a feature array without loading it entirely into RAM.
        For full documentation, please refer to the active storage backend.
        """
        return self._backend.get_feature_shape(name=name)
    
    def get_feature_names(self) -> List[str]:
        """
        List of all feature names.
        For full documentation, please refer to the active storage backend.
        """
        return self._backend.get_feature_names()

    def get_objects(self, idx: Optional[INDEX_LIKE] = None) -> Union[OT, List[OT]]:
        """
        Retrieve one or more objects (e.g., Molecules) from the table.
        For full documentation, please refer to the active storage backend.
        """
        return self._backend.get_objects(idx=idx)

    def update_objects(
        self,
        objs: Union[OT, List[OT]],
        idx: Optional[INDEX_LIKE] = None,
        **kwargs
    ) -> None:
        """
        Perform a partial or full update of stored objects.
        For full documentation, please refer to the active storage backend.
        """
        self._mutate_with_validation(
            domain="object",
            method="update_objects",
            mutate=lambda: self._backend.update_objects(objs=objs, idx=idx, **kwargs),
        )

    # ------------------------------------------------------------------
    # Backend API Helpers
    # ------------------------------------------------------------------

    def has_feature(self, name: str) -> bool:
        return name in self.get_feature_names()

    def merge_metadata(self, df: pd.DataFrame, on: str) -> None:
        """
        Merge external metadata into the table in-place using a relational join key.
        This modifies the schema by appending new columns.
        """
        current_cols = set(self.metadata_columns)
        
        if on not in current_cols:
            raise ValueError(f"Join column '{on}' not found in table metadata.")
        if on not in df.columns:
            raise ValueError(f"Join column '{on}' not found in provided DataFrame.")

        # Fetch ONLY the join column from the backend (Query Pushdown!)
        target_keys = self.get_metadata(cols=[on])
        target_keys['_orig_idx'] = np.arange(self.n)
        
        aligned_df: pd.DataFrame = target_keys.merge(df, on=on, how="left")
        aligned_df = aligned_df.sort_values('_orig_idx').reset_index(drop=True)
        aligned_df.drop(columns=[on, '_orig_idx'], inplace=True)
        
        if len(aligned_df) != self.n:
            raise ValueError(
                f"Merge resulted in {len(aligned_df)} rows, but table has {self.n}. "
                "Ensure the join key does not create duplicates."
            )
        
        # Route the aligned data to the appropriate backend methods
        to_update = {}
        to_add = {}

        for col in aligned_df.columns:
            if col in current_cols:
                to_update[col] = aligned_df[col].values
            else:
                to_add[col] = aligned_df[col].values

        if to_update:
            self.update_metadata(to_update)
        if to_add:
            self.add_metadata_columns(to_add)

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def append_history(self, entry: HistoryEntry) -> None:
        """Append one entry to the audit log."""
        self._history.append(entry)

    # ------------------------------------------------------------------
    # Advanced multi-axis indexing with strict query pushdown
    # ------------------------------------------------------------------

    def __getitem__(
        self,
        key: Union[int, slice, List[int], np.ndarray, pd.Series, Tuple],
    ) -> Any:
        """
        Advanced multi-axis indexing with backend query pushdown.

        Single-axis row selection (returns a new table)::

            table[0]           # single row table
            table[0:10]        # 10-row table
            table[[1, 3, 5]]   # 3-row table
            table[bool_array]  # filtered table

        Multi-axis selection (returns data, NOT a table)::

            table[FEAT, 'fps', 0:10]            # ndarray rows 0-9
            table[META, ['MolWt', 'LogP'], 5]   # single-row DataFrame
            table[OBJ, 3]                        # single object
            table[FEAT, 'fps']                   # full feature array (no idx)
        """
        if not isinstance(key, tuple):
            # Standard row-level subsetting → new table
            return self.subset(key)

        attr = key[0]

        # Safely parse variable-length tuple without IndexError.
        if len(key) == 2:
            if attr in (FEAT, F) and isinstance(key[1], str):
                names = key[1]
                idx = None
            else:
                names = None
                idx = key[1]
        elif len(key) == 3:
            names = key[1]
            idx = key[2]
        else:
            raise ValueError(
                "Invalid slicing format. Expected [ATTR, IDX] or [ATTR, NAMES, IDX]."
            )

        # Route to backend with strict query pushdown
        if attr in (META, M):
            return self._backend.get_metadata(idx=idx, cols=names)

        elif attr in (OBJ, O):
            if names is not None:
                raise ValueError("Objects do not have named columns.")
            return self._backend.get_objects(idx=idx)

        elif attr in (FEAT, F):
            if isinstance(names, str):
                return self._backend.get_feature(name=names, idx=idx)
            # Utilize the new batch-fetching capability in the backend!
            elif isinstance(names, list) or names is None:
                return self._backend.get_features(names=names, idx=idx)
            else:
                raise ValueError(
                    f"Expected str, List[str], or None for feature names; "
                    f"got {type(names).__name__}."
                )

        else:
            raise ValueError(f"Unknown attribute identifier: {attr!r}.")

    # ------------------------------------------------------------------
    # Slicing / subsetting
    # ------------------------------------------------------------------

    def subset(
        self,
        indices: Union[List[int], np.ndarray, pd.Series, slice, int],
        *,
        copy_objects: bool = True,
    ) -> "BaseTable[OT]":
        """
        Return a new table containing only the rows at ``indices``.

        The backend's ``create_view`` method is called to synchronise
        all three data stores at the storage layer.

        Parameters
        ----------
        indices
            Integer indices, boolean mask, slice, or single int.
        copy_objects
            Whether to deep-copy objects (default True).
            ``create_view`` always copies; set False only for internal
            in-place operations where a view is acceptable.
        """
        n = self.n

        if isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, slice):
            indices = list(range(*indices.indices(n)))

        idx = np.asarray(indices)
        if idx.dtype == bool:
            if len(idx) != n:
                raise ValueError(
                    f"Boolean mask length {len(idx)} must match table length {n}."
                )
            idx = np.where(idx)[0]

        idx = idx.astype(np.intp)

        # Push selection down to the backend
        new_backend = self._backend.create_view(idx.tolist())

        hist = list(self._history) + [
            HistoryEntry.now(
                block_name="BaseTable.subset",
                config={"n_indices": len(idx)},
                rows_in=n,
                rows_out=len(idx),
            )
        ]
        return self._new_instance_from_backend(new_backend, hist)

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy(self) -> "BaseTable[OT]":
        """Return a fully independent deep copy of this table."""
        new_backend = self._backend.create_view(list(range(len(self._backend))))
        return self._new_instance_from_backend(new_backend, list(self._history))

    # ------------------------------------------------------------------
    # Concatenation
    # ------------------------------------------------------------------

    @classmethod
    def concat(
        cls,
        tables: List["BaseTable[OT]"],
        *,
        handle_missing_features: str = "zeros",
    ) -> "BaseTable[OT]":
        """
        Row-wise concatenation of multiple tables of the same subclass.

        Parameters
        ----------
        tables
            List of tables to concatenate. All must be the same subclass.
        handle_missing_features
            What to do when a feature key is present in some tables but not
            others. Options:
            - ``'zeros'``  : fill missing rows with zeros (default)
            - ``'nan'``    : fill with NaN (float arrays only)
            - ``'raise'``  : raise ValueError on mismatch
        """
        if not tables:
            raise ValueError("concat() requires at least one table.")

        types = {type(t) for t in tables}
        if len(types) > 1:
            raise TypeError(
                f"All tables must be the same type; got {types}."
            )

        # objects
        all_objects: List[Any] = []
        for t in tables:
            all_objects.extend(t._backend.get_objects())

        # metadata
        meta_frames = [t._backend.get_metadata() for t in tables]
        combined_meta = pd.concat(meta_frames, ignore_index=True, sort=False)

        # features — union of all keys
        all_keys: set = set()
        for t in tables:
            all_keys.update(t._backend.get_feature_names())

        key_templates: Dict[str, np.ndarray] = {}
        for key in all_keys:
            for table in tables:
                if table.has_feature(key):
                    key_templates[key] = table._backend.get_feature(key)
                    break

        combined_features: Dict[str, np.ndarray] = {}
        for key in all_keys:
            parts = []
            present_arr = key_templates[key]
            for t in tables:
                if t.has_feature(key):
                    parts.append(t._backend.get_feature(key))
                else:
                    if handle_missing_features == "raise":
                        raise ValueError(
                            f"Feature '{key}' missing in at least one table."
                        )
                    shape = (len(t),) + present_arr.shape[1:]
                    if handle_missing_features == "nan":
                        if not np.issubdtype(present_arr.dtype, np.floating):
                            raise ValueError(
                                f"Feature '{key}' has dtype {present_arr.dtype}, which cannot represent NaN."
                            )
                        fill = np.full(shape, np.nan, dtype=present_arr.dtype)
                    else:
                        fill = np.zeros(shape, dtype=present_arr.dtype)
                    parts.append(fill)
            combined_features[key] = np.concatenate(parts, axis=0)

        # history
        combined_history: List[HistoryEntry] = []
        for t in tables:
            combined_history.extend(t._history)
        combined_history.append(
            HistoryEntry.now(
                block_name="BaseTable.concat",
                config={"n_tables": len(tables)},
                rows_in=sum(len(t) for t in tables),
                rows_out=len(all_objects),
            )
        )

        new_backend = EagerMemoryBackend(
            objects=all_objects,
            metadata=combined_meta,
            features=combined_features,
        )
        return tables[0]._new_instance_from_backend(new_backend, combined_history)

    # ------------------------------------------------------------------
    # Persistence — "Directory as a Bundle" (.dlb)
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path], overwrite: bool = False) -> Path:
        """
        Save the table to a ``.dlb`` directory bundle.

        Parameters
        ----------
        path
            The destination path for the ``.dlb`` bundle.
        overwrite
            If False (default), raises a FileExistsError if the target 
            path already exists. If True, atomically replaces the existing bundle.

        Layout
        ------
        ::

            <path>/
                config.json          (manifest: class, backend, history, schema)
                metadata.parquet     (or metadata.csv)
                objects/
                    objects.pkl
                features/
                    <name>.npy
        """
        import tempfile
        import shutil
        import json
        
        root = Path(path)
        
        # --- Safety Check: Prevent accidental overwrites ---
        if root.exists():
            if not overwrite:
                raise FileExistsError(
                    f"A bundle already exists at '{root}'. "
                    "Pass `overwrite=True` to explicitly replace it."
                )
        
        # Ensure the parent directory exists so we can create a temp dir next to it
        root.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. Create a secure temporary directory on the same filesystem
        temp_dir = Path(tempfile.mkdtemp(dir=root.parent, prefix=".tmp_dlb_"))

        try:
            # 2. Delegate data writing to the backend into the temporary directory
            self._backend.save(
                temp_dir,
                serializer=self._serialize_object,
            )

            # 3. Write config.json manifest
            history_data = [e.to_dict() for e in self._history]
            config = {
                "table_class": self.__class__.__name__,
                "object_type": self._object_type_name(),
                "backend_class": type(self._backend).__name__,
                "schema_version": 2,
                "n": self.n,
                "history": history_data,
            }
            (temp_dir / "config.json").write_text(
                json.dumps(config, indent=2), encoding="utf-8"
            )

            # 4. Atomic Swap: Remove the old bundle (if overwriting) and move the new one into place
            if root.exists():
                shutil.rmtree(root)
            temp_dir.replace(root)

        except Exception:
            # If ANYTHING fails during the save, clean up the temp directory 
            # and leave the original bundle (if it existed) completely untouched.
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise

        return root

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        mmap_features: bool = False,
    ) -> "BaseTable[OT]":
        """
        Load a table saved with ``save()``.

        Reads the ``config.json`` manifest to reconstruct the exact backend
        type and history, then delegates data loading to the backend.

        Parameters
        ----------
        path
            Bundle directory written by ``save()``.
        mmap_features
            If True, load feature arrays as ``numpy.memmap``.
        """
        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"No table found at '{root}'.")

        config_path = root / "config.json"
        if not config_path.exists():
            # Fallback: attempt to load legacy format (schema_version 1)
            return cls._load_legacy(root, mmap_features=mmap_features)

        config = json.loads(config_path.read_text(encoding="utf-8"))

        # Reconstruct history
        history = [HistoryEntry.from_dict(e) for e in config.get("history", [])]

        # --- THE FIX: Allocate the instance early so we can use its instance methods! ---
        instance = object.__new__(cls)

        backend_class_name = config.get("backend_class", "EagerMemoryBackend")
        if backend_class_name == "EagerMemoryBackend":
            backend = EagerMemoryBackend.load(
                root,
                deserializer=instance._deserialize_object,  # <-- Using the abstract instance contract!
                mmap_features=mmap_features,
            )
        else:
            raise ValueError(f"Unknown backend class '{backend_class_name}'.")

        # Now properly initialize the instance
        BaseTable.__init__(
            instance,
            _backend=backend,
            history=history,
        )
        return instance

    @classmethod
    def _load_legacy(
        cls,
        root: Path,
        *,
        mmap_features: bool = False,
    ) -> "BaseTable[OT]":
        """
        Fallback loader for tables saved with the old per-object-pickle format
        (schema_version 1).  Reads ``_meta.json`` instead of ``config.json``.
        """
        import pickle as _pickle

        meta_info = json.loads((root / "_meta.json").read_text())
        n = meta_info["n"]

        # FIX: Allocate instance early
        instance = object.__new__(cls)

        # objects (one file per object)
        obj_dir = root / "objects"
        objects = []
        for i in range(n):
            raw = (obj_dir / f"{i:07d}.pkl").read_bytes()
            objects.append(instance._deserialize_object(raw))

        # metadata
        parquet_path = root / "metadata.parquet"
        csv_path = root / "metadata.csv"
        if parquet_path.exists():
            metadata = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            metadata = pd.read_csv(csv_path)
        else:
            metadata = pd.DataFrame(index=range(n))

        # features
        feat_dir = root / "features"
        features: Dict[str, np.ndarray] = {}
        if feat_dir.exists():
            for npy_path in sorted(feat_dir.glob("*.npy")):
                feat_name = npy_path.stem
                if mmap_features:
                    features[feat_name] = np.load(str(npy_path), mmap_mode="r")
                else:
                    features[feat_name] = np.load(str(npy_path), allow_pickle=False)

        # history
        history_path = root / "history.json"
        history = []
        if history_path.exists():
            raw_history = json.loads(history_path.read_text())
            history = [HistoryEntry.from_dict(e) for e in raw_history]

        backend = EagerMemoryBackend(
            objects=objects,
            metadata=metadata,
            features=features,
        )
        
        # Initialize
        BaseTable.__init__(
            instance,
            _backend=backend,
            history=history,
        )
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_instance(
        self,
        objects: List[OT],
        metadata: pd.DataFrame,
        features: Dict[str, np.ndarray],
        history: List[HistoryEntry],
    ) -> "BaseTable[OT]":
        """
        Create a new instance of *this* subclass with raw data.
        Used by legacy internal callers (conformer/reaction modules).
        """
        backend = EagerMemoryBackend(
            objects=objects,
            metadata=metadata,
            features=features,
        )
        return self._new_instance_from_backend(backend, history)

    def _new_instance_from_backend(
        self,
        backend: BaseStorageBackend,
        history: List[HistoryEntry],
    ) -> "BaseTable[OT]":
        """
        Create a new instance of *this* subclass with a pre-built backend.
        This is the canonical internal factory used by subset, copy, concat.
        """
        instance = object.__new__(self.__class__)
        BaseTable.__init__(
            instance,
            _backend=backend,
            history=history,
        )
        return instance