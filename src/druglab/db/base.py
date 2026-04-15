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

import copy
import json
import pickle
import shutil
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

from .backends import EagerMemoryBackend, BaseStorageBackend

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
        auto_convert_numeric: bool = False, # TODO: Reimplmenet try_numerize_metadata
        _backend: Optional[BT] = None,
    ) -> None:
        # Allow callers to pass a pre-built backend (used internally by load())
        if _backend is not None:
            self._backend = _backend
        else:
            # Build the default EagerMemoryBackend from raw arguments
            obj_list  = objects  if objects  is not None else []
            meta_df   = metadata if metadata is not None else pd.DataFrame()
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
        self._backend.set_objects(objs)

    @property
    def metadata(self) -> pd.DataFrame:
        """Full metadata DataFrame (proxied from backend)."""
        return self._backend.get_metadata()
    
    @metadata.setter
    def metadata(self, meta: pd.DataFrame):
        return self._backend.set_metadata(meta)

    @property
    def metadata_columns(self) -> List[str]:
        """List of all metadata column names."""
        return self._backend.get_metadata().columns.tolist()

    @property
    def features(self) -> Dict[str, np.ndarray]:
        """Full feature dictionary (proxied from backend)."""
        return {k: self._backend.get_feature(k) for k in self._backend.get_feature_names()}
    
    @features.setter
    def features(self, feats: Dict[str, np.ndarray]):
        return self._backend.set_features(feats)
    
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
            f"{k}:{self._backend.get_feature(k).shape}"
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

        # --------------------------------------------------------------
        # n = len(self._backend)
        # meta = self._backend.get_metadata()

        # if not meta.empty and len(meta) != n and len(meta) != 0:
        #     raise ValueError(
        #         f"metadata has {len(meta)} rows but objects has {n}. "
        #         "They must be the same length."
        #     )
        # # If metadata is empty DataFrame, pad it to the right shape
        # if meta.empty and n > 0:
        #     self._backend.update_metadata(pd.DataFrame(index=range(n)))

        # for key in self._backend.get_feature_names():
        #     arr = self._backend.get_feature(key)
        #     if arr.shape[0] != n:
        #         raise ValueError(
        #             f"Feature '{key}' has {arr.shape[0]} rows but objects has {n}."
        #         )
        # --------------------------------------------------------------

        self._backend.validate()

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
    # Metadata helpers (backwards-compatible public API)
    # ------------------------------------------------------------------

    def add_metadata_column(self, name: str, values: Sequence) -> None:
        """Add (or overwrite) a metadata column."""
        if len(values) != self.n:
            raise ValueError(
                f"values length {len(values)} != table length {self.n}"
            )
        new_df = pd.DataFrame({name: values})
        self._backend.update_metadata(new_df, idx=None)

    def drop_metadata_columns(self, names: Union[str, List[str]]) -> None:
        """Drop one or more metadata columns in-place."""
        if isinstance(names, str):
            names = [names]
        self._backend.drop_metadata(cols=names)

    # def rename_metadata_columns(self, columns: Dict[str, str]) -> None:
    #     """Rename metadata columns in-place using a mapping dict."""
    #     meta = self._backend.get_metadata()
    #     meta.rename(columns=columns, inplace=True)
    #     self._backend.update_metadata(meta)

    def update_metadata(self, df: pd.DataFrame, on: Optional[str] = None) -> None:
        """Merge external metadata into the table in-place."""
        current = self._backend.get_metadata()

        if on is None:
            if len(df) != self.n:
                raise ValueError(
                    f"Length of update DataFrame ({len(df)}) must match "
                    f"table length ({self.n}) when joining by index."
                )
            for col in df.columns:
                current[col] = df[col].values
            self._backend.update_metadata(current)
        else:
            if on not in current.columns:
                raise ValueError(f"Join column '{on}' not found in table metadata.")
            if on not in df.columns:
                raise ValueError(f"Join column '{on}' not found in provided DataFrame.")

            temp_col = "__orig_idx__"
            while temp_col in current.columns or temp_col in df.columns:
                temp_col = f"_{temp_col}"

            overlap = [c for c in df.columns if c != on and c in current.columns]
            left: pd.DataFrame = (
                current.drop(columns=overlap).assign(**{temp_col: np.arange(self.n)})
            )
            merged: pd.DataFrame = left.merge(df, on=on, how="left")
            merged = merged.sort_values(temp_col).reset_index(drop=True)
            merged.drop(columns=[temp_col], inplace=True)

            if len(merged) != self.n:
                raise ValueError(
                    f"Merge resulted in {len(merged)} rows, but table has {self.n}. "
                    "Ensure the join key does not create duplicates."
                )
            self._backend.update_metadata(merged)

    # ------------------------------------------------------------------
    # Feature helpers (backwards-compatible public API)
    # ------------------------------------------------------------------

    def add_feature(self, name: str, array: np.ndarray) -> None:
        """Add (or overwrite) a feature array."""
        if array.shape[0] != self.n:
            raise ValueError(
                f"Feature array '{name}' has {array.shape[0]} rows; "
                f"table has {self.n}."
            )
        self._backend.add_feature(name, array)

    def drop_feature(self, name: str) -> None:
        """Remove a feature by name."""
        self._backend.drop_feature(name)

    def has_feature(self, name: str) -> bool:
        """Return True if the feature exists."""
        return name in self._backend.get_feature_names()

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
        # For FEAT with 2 elements: table[FEAT, 'fp'] means name='fp', idx=None.
        # For META/OBJ with 2 elements: table[META, 0:5] means names=None, idx=0:5.
        if len(key) == 2:
            if attr in (FEAT, F) and isinstance(key[1], str):
                # table[FEAT, 'fp'] — feature name only, no index
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
            elif isinstance(names, list):
                return {
                    feat_name: self._backend.get_feature(name=feat_name, idx=idx)
                    for feat_name in names
                }
            elif names is None:
                all_feats = self._backend.get_feature_names()
                return {
                    feat_name: self._backend.get_feature(name=feat_name, idx=idx)
                    for feat_name in all_feats
                }
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

        # If copy_objects=False we skip the deep-copy that create_view already did.
        # For now create_view always copies; this flag is preserved for API compat.

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

        combined_features: Dict[str, np.ndarray] = {}
        for key in all_keys:
            parts = []
            for t in tables:
                if t.has_feature(key):
                    parts.append(t._backend.get_feature(key))
                else:
                    if handle_missing_features == "raise":
                        raise ValueError(
                            f"Feature '{key}' missing in at least one table."
                        )
                    # Find shape from the first table that has this feature
                    present_arr = next(
                        t._backend.get_feature(key)
                        for t in tables
                        if t.has_feature(key)
                    )
                    shape = (len(t),) + present_arr.shape[1:]
                    fill = (
                        np.full(shape, np.nan, dtype=present_arr.dtype)
                        if handle_missing_features == "nan"
                        else np.zeros(shape, dtype=present_arr.dtype)
                    )
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

    def save(self, path: Union[str, Path]) -> Path:
        """
        Save the table to a ``.dlb`` directory bundle.

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
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)

        # Delegate data writing to the backend
        self._backend.save(
            root,
            serializer=self._serialize_object,
        )

        # Write config.json manifest
        history_data = [e.to_dict() for e in self._history]
        config = {
            "table_class": self.__class__.__name__,
            "object_type": self._object_type_name(),
            "backend_class": type(self._backend).__name__,
            "schema_version": 2,
            "n": self.n,
            "history": history_data,
        }
        (root / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

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

        # Load backend (currently only EagerMemoryBackend supported)
        backend_class_name = config.get("backend_class", "EagerMemoryBackend")
        if backend_class_name == "EagerMemoryBackend":
            backend = EagerMemoryBackend.load(
                root,
                deserializer=cls._deserialize_object_static,
                mmap_features=mmap_features,
            )
        else:
            raise ValueError(
                f"Unknown backend class '{backend_class_name}'. "
                "Only 'EagerMemoryBackend' is supported in this release."
            )

        instance = object.__new__(cls)
        BaseTable.__init__(
            instance,
            _backend=backend,
            history=history,
            auto_convert_numeric=False,
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

        # objects (one file per object)
        obj_dir = root / "objects"
        objects = []
        for i in range(n):
            raw = (obj_dir / f"{i:07d}.pkl").read_bytes()
            objects.append(cls._deserialize_object_static(raw))

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
        instance = object.__new__(cls)
        BaseTable.__init__(
            instance,
            _backend=backend,
            history=history,
            auto_convert_numeric=False,
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
            auto_convert_numeric=False,
        )
        return instance

    @staticmethod
    def _deserialize_object_static(raw: bytes) -> Any:
        """
        Fallback static deserialiser used by ``load()``.
        Subclasses that need a different strategy should override ``load()``
        or provide their own ``_deserialize_object``.
        """
        return pickle.loads(raw)  # noqa: S301