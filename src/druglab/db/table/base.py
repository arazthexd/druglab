"""
druglab.db.table.base
~~~~~~~~~~~~~~~
BaseTable: abstract orchestrator enforcing the four-property contract.
HistoryEntry: immutable record written by pipe blocks.

Pickle has been purged.  Object serialization is now driven exclusively by
explicit callbacks supplied either at call time or via the
``_get_default_object_writer`` / ``_get_default_object_reader`` hooks that
concrete subclasses are expected to override.

Saving a non-empty table whose subclass does not override those hooks (and
where no ``object_writer`` is passed directly) raises ``NotImplementedError``
at save time rather than silently producing a fragile pickle bundle.
"""

from __future__ import annotations

import re
import copy
import json
import inspect
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

    Serialization contract
    ----------------------
    Pickle is **not** used as a default.  Object I/O is driven by:

    1. ``_get_default_object_writer()`` — returns a writer callable or
       ``None``.  Subclasses override this to provide a domain-safe default
       (e.g., writing SMILES or SDF blocks).
    2. ``_get_default_object_reader()`` — symmetric reader hook.
    3. At save time, if neither the caller nor the subclass supplies a writer
       and the table is non-empty, ``save()`` raises ``NotImplementedError``
       rather than falling back to pickle.
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
        **kwargs: Any,
    ) -> None:
        if _backend is not None:
            self._backend = _backend
        else:
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
        return self._backend.get_objects()

    @objects.setter
    def objects(self, objs: List[OT]):
        self._mutate_with_validation(
            domain="object",
            method="objects.setter",
            mutate=lambda: self._backend.update_objects(objs),
        )

    @property
    def metadata(self) -> pd.DataFrame:
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
        return self._backend.get_metadata().columns.tolist()

    @property
    def features(self) -> Dict[str, np.ndarray]:
        return self._backend.get_features()

    @features.setter
    def features(self, feats: Dict[str, np.ndarray]):
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
        return self._backend.get_feature_names()

    @property
    def history(self) -> List[HistoryEntry]:
        return self._history

    @property
    def n(self) -> int:
        return len(self._backend)

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        feat_names = self._backend.get_feature_names()
        feat_summary = ", ".join(
            f"{k}:{self._backend.get_feature_shape(k)}" for k in feat_names
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
        if not isinstance(self._backend, OverlayBackend):
            self._backend.validate()

    def _run_post_mutation_validation(self, *, domain: str, method: str) -> None:
        try:
            self._validate()
        except Exception as exc:
            raise ValueError(
                f"Post-mutation validation failed in domain='{domain}', method='{method}': {exc}"
            ) from exc

    def _mutate_with_validation(self, *, domain: str, method: str, mutate: Callable[[], None]) -> None:
        try:
            mutate()
        except Exception as exc:
            raise type(exc)(
                f"Mutation failed in domain='{domain}', method='{method}': {exc}"
            ) from exc
        self._run_post_mutation_validation(domain=domain, method=method)

    # ------------------------------------------------------------------
    # Serialization hooks  (Task 1 & 2)
    # ------------------------------------------------------------------

    def _get_default_object_writer(self) -> Optional[Callable[[List[Any], Path], None]]:
        """
        Return the default object writer for this table subclass, or ``None``.

        Subclasses **must** override this to provide a domain-safe serialization
        strategy (e.g. writing SMILES or SDF blocks).  Returning ``None``
        causes ``save()`` to raise ``NotImplementedError`` for non-empty tables,
        which is the correct fail-fast behaviour for tables that have not yet
        defined a safe codec.
        """
        return None

    @classmethod
    def _get_default_object_reader(cls) -> Optional[Callable[[Path], List[Any]]]:
        """
        Return the default object reader for this table subclass, or ``None``.

        Subclasses **must** override this to match ``_get_default_object_writer``.
        """
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        path: Union[str, Path],
        overwrite: bool = False,
        object_writer: Optional[Callable[[List[Any], Path], None]] = None,
        metadata_format: str = "parquet",
        **kwargs: Any,
    ) -> Path:
        """
        Save the table to a ``.dlb`` directory bundle.

        Parameters
        ----------
        path : str | Path
            Destination path for the bundle directory.
        overwrite : bool
            Replace an existing bundle when ``True``.
        object_writer : Callable, optional
            Explicit writer that receives the full object list and the
            ``objects/`` directory.  When omitted, ``_get_default_object_writer``
            is called.  If that also returns ``None`` and the table is non-empty
            a ``NotImplementedError`` is raised — **no pickle fallback**.
        metadata_format : {"parquet", "csv"}
            Serialization format for the metadata DataFrame.  Recorded in the
            manifest so ``load()`` reads the matching file without guessing.
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

        # Resolve writer: explicit arg > subclass default > fail-fast.
        writer = object_writer or self._get_default_object_writer()

        if writer is None and len(self) > 0:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not define a safe object serialization "
                "strategy.  Override `_get_default_object_writer()` on your Table subclass "
                "or pass `object_writer=...` directly to `save()`."
            )

        temp_dir = Path(tempfile.mkdtemp(dir=root.parent, prefix=".tmp_dlb_"))

        try:
            self._backend.save(
                temp_dir,
                object_writer=writer,
                metadata_format=metadata_format,
            )

            n = len(self._backend)
            history_data = [e.to_dict() for e in self._history]
            config = {
                "table_class": self.__class__.__name__,
                "table_module": self.__class__.__module__,
                "backend_class": self._backend.get_name(),
                "backend_module": self._backend.get_module(),
                "schema_version": 2,
                "n": n,
                "history": history_data,
                # Record format so load() can reconstruct without guessing.
                "metadata_format": metadata_format,
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
        object_reader: Optional[Callable[[Path], List[Any]]] = None,
        **kwargs: Any,
    ) -> "BaseTable[OT]":
        """
        Load a table saved with ``save()``.

        The ``metadata_format`` stored in ``config.json`` is forwarded to the
        backend so the metadata store reads the correct file extension.
        The object reader is resolved from either the arguments or the 
        ``_get_default_object_reader`` method.
        """
        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"No table found at '{root}'.")

        config_path = root / "config.json"
        config: dict = json.loads(config_path.read_text(encoding="utf-8"))
        history = [HistoryEntry.from_dict(e) for e in config.pop("history", [])]

        backend_name   = config.pop("backend_class", "EagerMemoryBackend")
        backend_module = config.pop("backend_module", "druglab.db.backend")
        table_name     = config.pop("table_class", "BaseTable")
        table_module   = config.pop("table_module", "druglab.db.table")
        metadata_format = config.pop("metadata_format", "parquet")
        config.pop("schema_version", None)
        config.pop("n", None)

        import importlib
        backend_class: Type[BaseStorageBackend] = getattr(
            importlib.import_module(backend_module), backend_name
        )
        table_class: Type[BaseTable] = getattr(
            importlib.import_module(table_module), table_name
        )

        object_reader = object_reader or cls._get_default_object_reader()
        if object_reader is None:
            raise NotImplementedError(
                f"{cls.__name__} does not define a safe object deserialization "
                "strategy.  Override `_get_default_object_reader()` on your Table subclass "
                "or pass `object_reader=...` directly to `load()`."
            )
        
        backend = backend_class.load(
            root,
            object_reader=object_reader,
            metadata_format=metadata_format,
            **kwargs,
        )
        table = table_class(_backend=backend, history=history, **config)
        return table

    # ------------------------------------------------------------------
    # Backend Delegation API
    # ------------------------------------------------------------------

    def get_metadata(self, idx=None, cols=None) -> pd.DataFrame:
        return self._backend.get_metadata(idx=idx, cols=cols)

    def add_metadata_column(self, name, value, idx=None, na=None, **kwargs) -> None:
        self._mutate_with_validation(
            domain="metadata", method="add_metadata_column",
            mutate=lambda: self._backend.add_metadata_column(name=name, value=value, idx=idx, na=na, **kwargs),
        )

    def add_metadata_columns(self, columns, idx=None, na=None, **kwargs) -> None:
        self._mutate_with_validation(
            domain="metadata", method="add_metadata_columns",
            mutate=lambda: self._backend.add_metadata_columns(columns=columns, idx=idx, na=na, **kwargs),
        )

    def update_metadata(self, values, idx=None, **kwargs) -> None:
        self._mutate_with_validation(
            domain="metadata", method="update_metadata",
            mutate=lambda: self._backend.update_metadata(values=values, idx=idx, **kwargs),
        )

    def drop_metadata_columns(self, cols=None) -> None:
        self._mutate_with_validation(
            domain="metadata", method="drop_metadata_columns",
            mutate=lambda: self._backend.drop_metadata_columns(cols=cols),
        )

    def get_feature(self, name: str, idx=None) -> np.ndarray:
        return self._backend.get_feature(name=name, idx=idx)

    def get_features(self, names=None, idx=None) -> Dict[str, np.ndarray]:
        return self._backend.get_features(names=names, idx=idx)

    def update_feature(self, name: str, array: np.ndarray, idx=None, na=None, **kwargs) -> None:
        if not re.match(r"^[a-zA-Z0-9_=]+$", name):
            raise ValueError(
                f"Invalid feature name '{name}'. Feature names must be alphanumeric "
                "and contain no spaces or special characters."
            )
        self._mutate_with_validation(
            domain="feature", method="update_feature",
            mutate=lambda: self._backend.update_feature(name=name, array=array, idx=idx, na=na, **kwargs),
        )

    def update_features(self, arrays, idx=None, na=None, **kwargs) -> None:
        self._mutate_with_validation(
            domain="feature", method="update_features",
            mutate=lambda: self._backend.update_features(arrays=arrays, idx=idx, na=na, **kwargs),
        )

    def drop_feature(self, name: str) -> None:
        self._mutate_with_validation(
            domain="feature", method="drop_feature",
            mutate=lambda: self._backend.drop_feature(name=name),
        )

    def get_feature_shape(self, name: str) -> Tuple:
        return self._backend.get_feature_shape(name=name)

    def get_feature_names(self) -> List[str]:
        return self._backend.get_feature_names()

    def get_objects(self, idx=None):
        return self._backend.get_objects(idx=idx)

    def update_objects(self, objs, idx=None, **kwargs) -> None:
        self._mutate_with_validation(
            domain="object", method="update_objects",
            mutate=lambda: self._backend.update_objects(objs=objs, idx=idx, **kwargs),
        )

    def has_feature(self, name: str) -> bool:
        return name in self.get_feature_names()

    def merge_metadata(self, df: pd.DataFrame, on: str) -> None:
        current_cols = set(self.metadata_columns)
        if on not in current_cols:
            raise ValueError(f"Join column '{on}' not found in table metadata.")
        if on not in df.columns:
            raise ValueError(f"Join column '{on}' not found in provided DataFrame.")

        target_keys = self.get_metadata(cols=[on])
        target_keys["_orig_idx"] = np.arange(self.n)
        aligned_df: pd.DataFrame = target_keys.merge(df, on=on, how="left")
        aligned_df = aligned_df.sort_values("_orig_idx").reset_index(drop=True)
        aligned_df.drop(columns=[on, "_orig_idx"], inplace=True)

        if len(aligned_df) != self.n:
            raise ValueError(
                f"Merge resulted in {len(aligned_df)} rows, but table has {self.n}."
            )

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

    def append_history(self, entry: HistoryEntry) -> None:
        self._history.append(entry)

    def materialize(self, target_path: Optional[Path] = None) -> "BaseTable[OT]":
        concrete_backend = self._backend.materialize(target_path=target_path)
        return self._new_instance_from_backend(concrete_backend, list(self._history))

    def commit(self) -> None:
        if not isinstance(self._backend, OverlayBackend):
            raise TypeError("commit() is only available on tables backed by OverlayBackend.")
        self._backend.commit()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            return self.subset(key)

        attr = key[0]

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
            raise ValueError("Invalid slicing format.")

        if attr in (META, M):
            return self._backend.get_metadata(idx=idx, cols=names)
        elif attr in (OBJ, O):
            if names is not None:
                raise ValueError("Objects do not have named columns.")
            return self._backend.get_objects(idx=idx)
        elif attr in (FEAT, F):
            if isinstance(names, str):
                return self._backend.get_feature(name=names, idx=idx)
            elif isinstance(names, list) or names is None:
                return self._backend.get_features(names=names, idx=idx)
            else:
                raise ValueError(f"Expected str, List[str], or None for feature names.")
        else:
            raise ValueError(f"Unknown attribute identifier: {attr!r}.")

    def subset(self, indices, *, copy_objects: bool = True) -> "BaseTable[OT]":
        n = self.n
        if isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, slice):
            indices = list(range(*indices.indices(n)))

        idx = np.asarray(indices)
        if idx.dtype == bool:
            if len(idx) != n:
                raise ValueError(f"Boolean mask length {len(idx)} must match table length {n}.")
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

    def copy(self) -> "BaseTable[OT]":
        return self._new_instance_from_backend(
            backend=self._backend.clone(), history=list(self._history)
        )

    @classmethod
    def concat(cls, tables, *, handle_missing_features: str = "zeros") -> "BaseTable[OT]":
        if not tables:
            raise ValueError("concat() requires at least one table.")

        types = {type(t) for t in tables}
        if len(types) > 1:
            raise TypeError(f"All tables must be the same type; got {types}.")

        all_objects: List[Any] = []
        for t in tables:
            all_objects.extend(t._backend.get_objects())

        meta_frames = [t._backend.get_metadata() for t in tables]
        combined_meta = pd.concat(meta_frames, ignore_index=True, sort=False)

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
                        raise ValueError(f"Feature '{key}' missing in at least one table.")
                    shape = (len(t),) + present_arr.shape[1:]
                    if handle_missing_features == "nan":
                        if not np.issubdtype(present_arr.dtype, np.floating):
                            raise ValueError(
                                f"Feature '{key}' has dtype {present_arr.dtype}, "
                                "which cannot represent NaN."
                            )
                        fill = np.full(shape, np.nan, dtype=present_arr.dtype)
                    else:
                        fill = np.zeros(shape, dtype=present_arr.dtype)
                    parts.append(fill)
            combined_features[key] = np.concatenate(parts, axis=0)

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
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_instance_from_backend(
        self,
        backend: BaseStorageBackend,
        history: List[HistoryEntry],
    ) -> "BaseTable[OT]":
        instance = object.__new__(self.__class__)
        BaseTable.__init__(instance, _backend=backend, history=history)
        return instance