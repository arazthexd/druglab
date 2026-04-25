"""
druglab.db.table.base
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
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar, Union, Type

import numpy as np
import pandas as pd

from ..backend import EagerMemoryBackend, BaseStorageBackend
from ..backend.overlay import OverlayBackend
from ..indexing import INDEX_LIKE, RowSelection, normalize_row_index

# ---------------------------------------------------------------------------
# Type variables
# ---------------------------------------------------------------------------

OT = TypeVar("OT")
MT = TypeVar("MT", bound=pd.DataFrame)
FT = TypeVar("FT", bound=np.ndarray)
BT = TypeVar("BT", bound=BaseStorageBackend)

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

class BaseTable(ABC, Generic[OT]):
    """
    Abstract base class for all DrugLab table types.

    Data is delegated to a ``BaseStorageBackend`` (default: EagerMemoryBackend).
    All public properties proxy to the backend, maintaining backwards
    compatibility with the previous attribute-based API.

    Row addressing uses the shared ``druglab.db.indexing`` module, which
    provides ``normalize_row_index`` and ``RowSelection`` as the canonical
    index-resolution primitives across all backends and the table layer.

    Invariant (always enforced):
        len(objects) == len(metadata)
        features[k].shape[0] == len(objects)  for every key k

    Subclasses must implement either:
        1) Single object versions: 
        
            A) ``_serialize_object(obj) -> bytes``
            B) ``_deserialize_object(raw) -> Any``

        2) Batched, more generalized versions (RECOMMENDED):

            A) ``_make_object_writer() -> Callable[[List[Any], Path], None]``
            B) ``_make_object_reader() -> Callable[[Path], List[Any]]``

    Multi-axis indexing (strict query pushdown):
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
        **kwargs: Any
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
    # Property wrappers (TODO: SERIOUS CHECKS REQUIRED)
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
        # OverlayBackend skips the full validate() (no-op), which is fine.
        if not isinstance(self._backend, OverlayBackend):
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
            self._validate()            
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
    # Persistence
    # ------------------------------------------------------------------

    def _serialize_object(self, obj: OT) -> bytes:
        """Serialise a single object to bytes for disk storage."""
        raise NotImplementedError()

    @classmethod
    def _deserialize_object(cls, raw: bytes) -> OT:
        """Deserialise bytes back to an object."""
        raise NotImplementedError()

    # @abstractmethod
    # def _object_type_name(self) -> str:
    #     """Short human-readable name for the object type (e.g. 'Mol')."""

    def _make_object_writer(self) -> Callable[[List[Any], Path], None]:
        """
        Build a bulk ``object_writer`` callable from ``_serialize_object``.
 
        The writer streams each serialised object into a ``stream_v2`` pickle
        bundle under ``<dir_path>/objects.pkl``, marking ``serialized=True``
        so the reader knows to call ``_deserialize_object`` on load.
 
        Subclasses may override this to supply an entirely custom bulk format
        (e.g. a RDKit SDF writer that writes all molecules in one pass).
        """
        serialize = self._serialize_object
 
        def _writer(objects: List[Any], dir_path: Path) -> None:
            with open(dir_path / "objects.pkl", "wb") as f:
                pickle.dump(
                    {
                        "format": "stream_v2",
                        "count": len(objects),
                        "serialized": True,
                    },
                    f,
                )
                for obj in objects:
                    pickle.dump(serialize(obj), f)
 
        return _writer
 
    @classmethod
    def _make_object_reader(cls) -> Callable[[Path], List[Any]]:
        """
        Build a bulk ``object_reader`` callable from ``_deserialize_object``.
 
        Reads a ``stream_v2`` pickle bundle from ``<dir_path>/objects.pkl``.
        When the bundle header says ``serialized=True``, applies
        ``_deserialize_object`` to each raw payload; otherwise returns
        the payloads as-is (for bundles written without a serialiser).
 
        Subclasses may override this to supply a matching bulk format reader.
        """
        deserialize = cls._deserialize_object
 
        def _reader(dir_path: Path) -> List[Any]:
            obj_path = dir_path / "objects.pkl"
            if not obj_path.exists():
                return []
            with open(obj_path, "rb") as f:
                raw_payload = pickle.load(f)
 
                count = int(raw_payload["count"])
                is_serialized = raw_payload.get("serialized", False)
                raw_list = [pickle.load(f) for _ in range(count)]
 
            if is_serialized:
                return [deserialize(r) for r in raw_list]
            return raw_list
 
        return _reader
    
    def save(self, path: Union[str, Path], overwrite: bool = False) -> Path:
        """
        Save the table to a ``.dlb`` directory bundle.
 
        Phase 1 change: delegates object writing via a batch ``object_writer``
        callable built from ``_make_object_writer()`` rather than the old
        per-object ``serializer`` kwarg.
 
        Parameters
        ----------
        path
            The destination path for the ``.dlb`` bundle.
        overwrite
            If False (default), raises a FileExistsError if the target
            path already exists.  If True, atomically replaces it.
        """
        import tempfile
 
        root = Path(path)
 
        if root.exists():
            if not overwrite:
                raise FileExistsError(
                    f"A bundle already exists at '{root}'. "
                    "Pass `overwrite=True` to explicitly replace it."
                )
 
        root.parent.mkdir(parents=True, exist_ok=True)
 
        # Build the batch writer from per-object _serialize_object hook.
        object_writer = self._make_object_writer()
 
        temp_dir = Path(tempfile.mkdtemp(dir=root.parent, prefix=".tmp_dlb_"))
 
        try:
            # Delegate data writing to the backend (or overlay → materialize).
            self._backend.save(temp_dir, object_writer=object_writer)
 
            n = len(self._backend)
            history_data = [e.to_dict() for e in self._history]
            config = {
                "table_class": self.__class__.__name__,
                "table_module": self.__class__.__module__,
                # "object_type": self._object_type_name(),
                "backend_class": self._backend.get_name(),
                "backend_module": self._backend.get_module(),
                "schema_version": 2,
                "n": n,
                "history": history_data,
            }
            (temp_dir / "config.json").write_text(
                json.dumps(config, indent=2), encoding="utf-8"
            )
 
            if root.exists():
                shutil.rmtree(root)
            temp_dir.replace(root)
 
        except Exception as e:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise e
 
        return root
 
    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        **kwargs: Any
    ) -> "BaseTable[OT]":
        """
        Load a table saved with ``save()``.
 
        Phase 1 change: builds a batch ``object_reader`` from
        ``_make_object_reader()`` and passes it to ``EagerMemoryBackend.load``
        instead of the old per-object ``deserializer``.
        """
        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"No table found at '{root}'.")
 
        config_path = root / "config.json" 
        config: dict = json.loads(config_path.read_text(encoding="utf-8"))
        history = [HistoryEntry.from_dict(e) for e in config.pop("history", [])]
 
        backend_name = config.pop("backend_class", "EagerMemoryBackend")
        backed_module = config.pop("backend_module", "druglab.db.backend")
        table_name = config.pop("table_class", "BaseTable")
        table_module = config.pop("table_module", "druglab.db.table")

        import importlib
        backend_class: Type[BaseStorageBackend] = getattr(
            importlib.import_module(backed_module), 
            backend_name
        )
        table_class: Type[BaseTable] = getattr(
            importlib.import_module(table_module), 
            table_name
        )

        object_reader = cls._make_object_reader()
        backend = backend_class.load(root, object_reader=object_reader, **kwargs)
        table = table_class(_backend=backend, history=history, **config)
        return table

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
    # Materialization
    # ------------------------------------------------------------------

    def materialize(
        self,
        target_path: Optional[Path] = None,
    ) -> "BaseTable[OT]":
        """
        Collapse the backend proxy tree into a concrete backend of the same
        class as the underlying base.
 
        Delegates to ``backend.materialize(target_path)`` for both concrete and
        proxy backends. Concrete backends deep-copy themselves, while proxies
        can collapse deferred state into concrete backends.
 
        Parameters
        ----------
        target_path : Path, optional
            Forwarded to the backend's materialise call.  Reserved for future
            out-of-core backends.
        """
        concrete_backend = self._backend.materialize(target_path=target_path)
        return self._new_instance_from_backend(concrete_backend, list(self._history))
    
    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def commit(self) -> None:
        """
        Flush all local deltas from an OverlayBackend down to the base backend.
 
        Raises TypeError if the backend is not an OverlayBackend.
        """
        if not isinstance(self._backend, OverlayBackend):
            raise TypeError(
                "commit() is only available on tables backed by OverlayBackend."
            )
        self._backend.commit()

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
 
        Creates an ``OverlayBackend`` proxy over the current backend using the
        resolved index map. This preserves zero-copy subset behavior.
 
        Parameters
        ----------
        indices
            Integer indices, boolean mask, slice, or single int.
        copy_objects
            Kept for API compatibility; ignored (OverlayBackend is zero-copy).
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
 
        new_backend = OverlayBackend(self._backend, idx)
 
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
        if isinstance(self._backend, OverlayBackend):
            new_backend = self._backend.materialize()
        else:
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

    # def save(self, path: Union[str, Path], overwrite: bool = False) -> Path:
    #     """
    #     Save the table to a ``.dlb`` directory bundle.

    #     If the backend is an OverlayBackend, it is materialized first so the
    #     saved bundle represents the clean, unified state.

    #     Parameters
    #     ----------
    #     path
    #         The destination path for the ``.dlb`` bundle.
    #     overwrite
    #         If False (default), raises a FileExistsError if the target 
    #         path already exists. If True, atomically replaces the existing bundle.

    #     Layout
    #     ------
    #     ::

    #         <path>/
    #             config.json          (manifest: class, backend, history, schema)
    #             metadata.parquet     (or metadata.csv)
    #             objects/
    #                 objects.pkl
    #             features/
    #                 <name>.npy
    #     """
    #     import tempfile
    #     import shutil
    #     import json
        
    #     root = Path(path)
        
    #     # --- Safety Check: Prevent accidental overwrites ---
    #     if root.exists():
    #         if not overwrite:
    #             raise FileExistsError(
    #                 f"A bundle already exists at '{root}'. "
    #                 "Pass `overwrite=True` to explicitly replace it."
    #             )
        
    #     # Ensure the parent directory exists so we can create a temp dir next to it
    #     root.parent.mkdir(parents=True, exist_ok=True)
        
    #     # 1. Create a secure temporary directory on the same filesystem
    #     temp_dir = Path(tempfile.mkdtemp(dir=root.parent, prefix=".tmp_dlb_"))

    #     try:
    #         # 2. Delegate data writing to the backend into the temporary directory
    #         #    OverlayBackend.save() internally calls materialize().save()
    #         self._backend.save(
    #             temp_dir,
    #             serializer=self._serialize_object,
    #         )

    #         # 3. Write config.json manifest
    #         # Use the materialized length for config
    #         n = len(self._backend)
    #         history_data = [e.to_dict() for e in self._history]
    #         config = {
    #             "table_class": self.__class__.__name__,
    #             "object_type": self._object_type_name(),
    #             "backend_class": "EagerMemoryBackend",  # always save as concrete
    #             "schema_version": 2,
    #             "n": n,
    #             "history": history_data,
    #         }
    #         (temp_dir / "config.json").write_text(
    #             json.dumps(config, indent=2), encoding="utf-8"
    #         )

    #         # 4. Atomic Swap
    #         if root.exists():
    #             shutil.rmtree(root)
    #         temp_dir.replace(root)

    #     except Exception:
    #         if temp_dir.exists():
    #             shutil.rmtree(temp_dir)
    #         raise

    #     return root

    # @classmethod
    # def load(
    #     cls,
    #     path: Union[str, Path],
    #     *,
    #     mmap_features: bool = False,
    # ) -> "BaseTable[OT]":
    #     """
    #     Load a table saved with ``save()``.
    #     """
    #     root = Path(path)
    #     if not root.is_dir():
    #         raise FileNotFoundError(f"No table found at '{root}'.")

    #     config_path = root / "config.json"
    #     if not config_path.exists():
    #         return cls._load_legacy(root, mmap_features=mmap_features)

    #     config = json.loads(config_path.read_text(encoding="utf-8"))

    #     history = [HistoryEntry.from_dict(e) for e in config.get("history", [])]

    #     instance = object.__new__(cls)

    #     backend_class_name = config.get("backend_class", "EagerMemoryBackend")
    #     if backend_class_name == "EagerMemoryBackend":
    #         backend = EagerMemoryBackend.load(
    #             root,
    #             deserializer=instance._deserialize_object,
    #             mmap_features=mmap_features,
    #         )
    #     else:
    #         raise ValueError(f"Unknown backend class '{backend_class_name}'.")

    #     BaseTable.__init__(
    #         instance,
    #         _backend=backend,
    #         history=history,
    #     )
    #     return instance

    # @classmethod
    # def _load_legacy(
    #     cls,
    #     root: Path,
    #     *,
    #     mmap_features: bool = False,
    # ) -> "BaseTable[OT]":
    #     """
    #     Fallback loader for tables saved with the old per-object-pickle format
    #     (schema_version 1).
    #     """
    #     import pickle as _pickle

    #     meta_info = json.loads((root / "_meta.json").read_text())
    #     n = meta_info["n"]

    #     instance = object.__new__(cls)

    #     obj_dir = root / "objects"
    #     objects = []
    #     for i in range(n):
    #         raw = (obj_dir / f"{i:07d}.pkl").read_bytes()
    #         objects.append(instance._deserialize_object(raw))

    #     parquet_path = root / "metadata.parquet"
    #     csv_path = root / "metadata.csv"
    #     if parquet_path.exists():
    #         metadata = pd.read_parquet(parquet_path)
    #     elif csv_path.exists():
    #         metadata = pd.read_csv(csv_path)
    #     else:
    #         metadata = pd.DataFrame(index=range(n))

    #     feat_dir = root / "features"
    #     features: Dict[str, np.ndarray] = {}
    #     if feat_dir.exists():
    #         for npy_path in sorted(feat_dir.glob("*.npy")):
    #             feat_name = npy_path.stem
    #             if mmap_features:
    #                 features[feat_name] = np.load(str(npy_path), mmap_mode="r")
    #             else:
    #                 features[feat_name] = np.load(str(npy_path), allow_pickle=False)

    #     history_path = root / "history.json"
    #     history = []
    #     if history_path.exists():
    #         raw_history = json.loads(history_path.read_text())
    #         history = [HistoryEntry.from_dict(e) for e in raw_history]

    #     backend = EagerMemoryBackend(
    #         objects=objects,
    #         metadata=metadata,
    #         features=features,
    #     )
        
    #     BaseTable.__init__(
    #         instance,
    #         _backend=backend,
    #         history=history,
    #     )
    #     return instance

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
        """
        instance = object.__new__(self.__class__)
        BaseTable.__init__(
            instance,
            _backend=backend,
            history=history,
        )
        return instance