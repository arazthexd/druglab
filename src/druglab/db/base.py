"""
druglab.db.base
~~~~~~~~~~~~~~~
BaseTable: abstract base class enforcing the four-property contract.
HistoryEntry: immutable record written by pipe blocks.
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
from typing import Any, Dict, Generic, List, Optional, Sequence, TypeVar, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Type variable for the object list (Mol, Reaction, …)
# ---------------------------------------------------------------------------
T = TypeVar("T")


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

class BaseTable(ABC, Generic[T]):
    """
    Abstract base class for all DrugLab table types.

    Invariant (always enforced):
        len(objects) == len(metadata)
        features[k].shape[0] == len(objects)  for every key k

    Subclasses must implement:
        _serialize_object(obj)   -> bytes-like or JSON-able
        _deserialize_object(raw) -> T
        _object_type_name()      -> str   (used in repr and error messages)

    Subclasses may override:
        copy()  if T objects need deep-copy semantics beyond pickle
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        objects: Optional[List[T]] = None,
        metadata: Optional[pd.DataFrame] = None,
        features: Optional[Dict[str, np.ndarray]] = None,
        history: Optional[List[HistoryEntry]] = None,
        *,
        auto_convert_numeric: bool = False,
    ) -> None:
        self._objects: List[T] = objects if objects is not None else []
        
        if metadata is not None:
            self._metadata: pd.DataFrame = metadata.reset_index(drop=True)
            if auto_convert_numeric and not self._metadata.empty:
                self.try_numerize_metadata()
        else:
            self._metadata = pd.DataFrame()

        self._features: Dict[str, np.ndarray] = features if features is not None else {}
        self._history: List[HistoryEntry] = history if history is not None else []
        self._validate()

    def try_numerize_metadata(
        self,
        columns: Optional[List[str]] = None,
    ) -> None:
        if columns is None:
            columns = self._metadata.columns
        columns_set = set(columns)

        # Convert object/string columns to numeric where possible
        # (Crucial for CSV/SDF loads where floats are imported as strings)
        for col in self._metadata.select_dtypes(include=["object", "string"]).columns:
            if col not in columns_set:
                continue
            try:
                self._metadata[col] = pd.to_numeric(self._metadata[col])
            except (ValueError, TypeError):
                pass # If it can't be safely converted, leave as strings

    # ------------------------------------------------------------------
    # Properties (read access; mutation goes through explicit methods)
    # ------------------------------------------------------------------

    @property
    def objects(self) -> List[T]:
        return self._objects

    @property
    def metadata(self) -> pd.DataFrame:
        return self._metadata

    @property
    def metadata_columns(self) -> List[str]:
        """List of all metadata column names."""
        return self._metadata.columns.tolist()

    @property
    def features(self) -> Dict[str, np.ndarray]:
        return self._features

    @property
    def history(self) -> List[HistoryEntry]:
        return self._history

    # Convenience
    @property
    def n(self) -> int:
        """Number of rows / objects in the table."""
        return len(self._objects)

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        feat_summary = ", ".join(
            f"{k}:{v.shape}" for k, v in self._features.items()
        )
        return (
            f"{self.__class__.__name__}("
            f"n={self.n}, "
            f"metadata_cols={list(self._metadata.columns)}, "
            f"features=[{feat_summary}])"
        )

    # ------------------------------------------------------------------
    # Invariant enforcement
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        n = len(self._objects)

        if len(self._metadata) != n and len(self._metadata) != 0:
            raise ValueError(
                f"metadata has {len(self._metadata)} rows but objects has {n}. "
                "They must be the same length."
            )
        # If metadata is empty DataFrame, pad it to the right shape
        if len(self._metadata) == 0 and n > 0:
            self._metadata = pd.DataFrame(index=range(n))

        for key, arr in self._features.items():
            if arr.shape[0] != n:
                raise ValueError(
                    f"Feature '{key}' has {arr.shape[0]} rows but objects has {n}."
                )

    # ------------------------------------------------------------------
    # Abstract interface (subclasses implement object serialisation)
    # ------------------------------------------------------------------

    @abstractmethod
    def _serialize_object(self, obj: T) -> bytes:
        """Serialise a single object to bytes for disk storage."""

    @abstractmethod
    def _deserialize_object(self, raw: bytes) -> T:
        """Deserialise bytes back to an object."""

    @abstractmethod
    def _object_type_name(self) -> str:
        """Short human-readable name for the object type (e.g. 'Mol')."""

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def add_metadata_column(self, name: str, values: Sequence) -> None:
        """
        Add (or overwrite) a metadata column. ``values`` must have
        the same length as the table.
        """
        if len(values) != self.n:
            raise ValueError(
                f"values length {len(values)} != table length {self.n}"
            )
        self._metadata[name] = list(values)

    def drop_metadata_columns(self, names: Union[str, List[str]]) -> None:
        """
        Drop one or more metadata columns in-place.
        """
        if isinstance(names, str):
            names = [names]
        self._metadata.drop(columns=names, inplace=True)

    def rename_metadata_columns(self, columns: Dict[str, str]) -> None:
        """
        Rename metadata columns in-place using a dictionary mapping.
        """
        self._metadata.rename(columns=columns, inplace=True)

    def update_metadata(self, df: pd.DataFrame, on: Optional[str] = None) -> None:
        """
        Merge external metadata into the table in-place.
        
        Parameters
        ----------
        df : pd.DataFrame
            The external metadata dataframe to merge.
        on : str, optional
            Column name to join on. If None, performs a strict row-index alignment,
            which requires `len(df) == self.n`.
        """
        if on is None:
            if len(df) != self.n:
                raise ValueError(
                    f"Length of update DataFrame ({len(df)}) must match "
                    f"table length ({self.n}) when joining by index."
                )
            # Assign values directly to avoid index mismatch issues
            for col in df.columns:
                self._metadata[col] = df[col].values
        else:
            if on not in self._metadata.columns:
                raise ValueError(f"Join column '{on}' not found in table metadata.")
            if on not in df.columns:
                raise ValueError(f"Join column '{on}' not found in provided DataFrame.")
            
            # To maintain exact row order (aligning with self._objects), we track
            # original indices, merge, and then sort back to original order.
            temp_col = "__orig_idx__"
            while temp_col in self._metadata.columns or temp_col in df.columns:
                temp_col = f"_{temp_col}"

            overlap = [c for c in df.columns if c != on and c in self._metadata.columns]
            left: pd.DataFrame = (
                self._metadata.drop(columns=overlap).assign(**{temp_col: np.arange(self.n)})
            )
            merged: pd.DataFrame = left.merge(df, on=on, how="left")
            merged: pd.DataFrame = merged.sort_values(temp_col)
            merged: pd.DataFrame = merged.reset_index(drop=True)
            merged.drop(columns=[temp_col], inplace=True)
            
            if len(merged) != self.n:
                raise ValueError(
                    f"Merge resulted in {len(merged)} rows, but table has {self.n}. "
                    "Ensure the join key does not create duplicates."
                )
            self._metadata = merged

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    def add_feature(self, name: str, array: np.ndarray) -> None:
        """
        Add (or overwrite) a feature array.  ``array.shape[0]`` must
        equal ``self.n``.
        """
        if array.shape[0] != self.n:
            raise ValueError(
                f"Feature array '{name}' has {array.shape[0]} rows; "
                f"table has {self.n}."
            )
        self._features[name] = array

    def drop_feature(self, name: str) -> None:
        del self._features[name]

    def has_feature(self, name: str) -> bool:
        return name in self._features

    # ------------------------------------------------------------------
    # History helpers (pipe blocks use these)
    # ------------------------------------------------------------------

    def append_history(self, entry: HistoryEntry) -> None:
        """Append one entry to the audit log. Called by pipe blocks."""
        self._history.append(entry)

    # ------------------------------------------------------------------
    # Slicing / subsetting
    # ------------------------------------------------------------------

    def subset(
        self,
        indices: Union[List[int], np.ndarray, pd.Series],
        *,
        copy_objects: bool = True,
    ) -> "BaseTable[T]":
        """
        Return a new table containing only the rows at ``indices``.
        The returned table is a fresh instance of the same subclass.

        Parameters
        ----------
        indices
            Integer indices or boolean mask.
        copy_objects
            Whether to deep-copy the object list (default True).
        """
        idx = np.asarray(indices)
        if idx.dtype == bool:
            if len(idx) != self.n:
                raise ValueError(
                    f"Boolean mask length {len(idx)} must match table length {self.n}."
                )
            idx = np.where(idx)[0]

        objs = [
            (copy.deepcopy(self._objects[i]) if copy_objects else self._objects[i])
            for i in idx
        ]
        meta = self._metadata.iloc[idx].reset_index(drop=True)
        feats = {k: v[idx] for k, v in self._features.items()}
        hist = list(self._history) + [
            HistoryEntry.now(
                block_name="BaseTable.subset",
                config={"n_indices": len(idx)},
                rows_in=self.n,
                rows_out=len(idx),
            )
        ]
        return self._new_instance(objs, meta, feats, hist)

    def __getitem__(self, key: Union[int, slice, List[int], np.ndarray, pd.Series]) -> "BaseTable[T]":
        """
        Support table[0], table[0:10], table[[1,3,5]], table[bool_array].
        This is the preferred Pythonic way to filter by metadata, e.g.:
            table[table.metadata["MolWt"] < 500]
        """
        if isinstance(key, int):
            key = [key]
        elif isinstance(key, slice):
            key = list(range(*key.indices(self.n)))
        return self.subset(key)

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy(self) -> "BaseTable[T]":
        """Return a fully independent deep copy of this table."""
        return self._new_instance(
            objects=copy.deepcopy(self._objects),
            metadata=self._metadata.copy(deep=True),
            features={k: v.copy() for k, v in self._features.items()},
            history=list(self._history),
        )

    # ------------------------------------------------------------------
    # Concatenation (class method — concat two tables of the same type)
    # ------------------------------------------------------------------

    @classmethod
    def concat(
        cls,
        tables: List["BaseTable[T]"],
        *,
        handle_missing_features: str = "zeros",
    ) -> "BaseTable[T]":
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
        all_objects: List[T] = []
        for t in tables:
            all_objects.extend(t._objects)

        # metadata
        meta_frames = [t._metadata for t in tables]
        combined_meta = pd.concat(meta_frames, ignore_index=True, sort=False)

        # features — union of all keys
        all_keys = set()
        for t in tables:
            all_keys.update(t._features.keys())

        combined_features: Dict[str, np.ndarray] = {}
        for key in all_keys:
            parts = []
            for t in tables:
                if key in t._features:
                    parts.append(t._features[key])
                else:
                    if handle_missing_features == "raise":
                        raise ValueError(
                            f"Feature '{key}' missing in at least one table."
                        )
                    present = next(
                        t._features[key]
                        for t in tables
                        if key in t._features
                    )
                    shape = (len(t),) + present.shape[1:]
                    fill = (
                        np.full(shape, np.nan, dtype=present.dtype)
                        if handle_missing_features == "nan"
                        else np.zeros(shape, dtype=present.dtype)
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

        return tables[0]._new_instance(
            all_objects, combined_meta, combined_features, combined_history
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> Path:
        """
        Save the table to a directory on disk.

        Layout
        ------
        <path>/
            objects/
                0000000.pkl   (one file per object)
            metadata.parquet
            features/
                <name>.npy    (or <name>.dat + <name>.npy for memmaps)
            history.json
            _meta.json        (type name, schema version)
        """
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)

        # --- objects ---
        obj_dir = root / "objects"
        obj_dir.mkdir(exist_ok=True)
        for i, obj in enumerate(self._objects):
            (obj_dir / f"{i:07d}.pkl").write_bytes(self._serialize_object(obj))

        # --- metadata ---
        if not self._metadata.empty:
            try:
                self._metadata.to_parquet(root / "metadata.parquet", index=False)
            except ImportError:
                self._metadata.to_csv(root / "metadata.csv", index=False)

        # --- features ---
        feat_dir = root / "features"
        feat_dir.mkdir(exist_ok=True)
        for name, arr in self._features.items():
            safe_name = name.replace("/", "_")
            if isinstance(arr, np.memmap):
                # Save shape/dtype descriptor; the memmap file stays in place
                np.save(feat_dir / f"{safe_name}.npy", np.array(arr))
            else:
                np.save(feat_dir / f"{safe_name}.npy", arr)

        # --- history ---
        history_data = [e.to_dict() for e in self._history]
        (root / "history.json").write_text(
            json.dumps(history_data, indent=2), encoding="utf-8"
        )

        # --- meta ---
        (root / "_meta.json").write_text(
            json.dumps(
                {
                    "type": self.__class__.__name__,
                    "object_type": self._object_type_name(),
                    "schema_version": 1,
                    "n": self.n,
                }
            ),
            encoding="utf-8",
        )

        return root

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        mmap_features: bool = False,
    ) -> "BaseTable[T]":
        """
        Load a table saved with ``save()``.

        Parameters
        ----------
        path
            Directory written by ``save()``.
        mmap_features
            If True, load feature arrays as ``numpy.memmap`` (memory-mapped)
            instead of loading them fully into RAM. Useful for large datasets.
        """
        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"No table found at '{root}'.")

        meta_info = json.loads((root / "_meta.json").read_text())
        n = meta_info["n"]

        # --- objects ---
        obj_dir = root / "objects"
        objects = []
        for i in range(n):
            raw = (obj_dir / f"{i:07d}.pkl").read_bytes()
            objects.append(cls._deserialize_object_static(raw))

        # --- metadata ---
        parquet_path = root / "metadata.parquet"
        csv_path = root / "metadata.csv"
        if parquet_path.exists():
            metadata = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            metadata = pd.read_csv(csv_path)
        else:
            metadata = pd.DataFrame(index=range(n))

        # --- features ---
        feat_dir = root / "features"
        features: Dict[str, np.ndarray] = {}
        if feat_dir.exists():
            for npy_path in sorted(feat_dir.glob("*.npy")):
                name = npy_path.stem
                if mmap_features:
                    features[name] = np.load(str(npy_path), mmap_mode="r")
                else:
                    features[name] = np.load(str(npy_path), allow_pickle=False)

        # --- history ---
        history_path = root / "history.json"
        history = []
        if history_path.exists():
            raw_history = json.loads(history_path.read_text())
            history = [HistoryEntry.from_dict(e) for e in raw_history]

        return cls(
            objects=objects,
            metadata=metadata,
            features=features,
            history=history,
            # Data loaded from disk has already been converted on the initial save if necessary
            auto_convert_numeric=False, 
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_instance(
        self,
        objects: List[T],
        metadata: pd.DataFrame,
        features: Dict[str, np.ndarray],
        history: List[HistoryEntry],
    ) -> "BaseTable[T]":
        """
        Create a new instance of *this* subclass with the given data.
        Subclasses with custom __init__ signatures should override this.
        """
        instance = object.__new__(self.__class__)
        # Ensure we don't redundantly process string->numeric checks when subsetting/copying internally
        BaseTable.__init__(
            instance, 
            objects=objects, 
            metadata=metadata, 
            features=features, 
            history=history, 
            auto_convert_numeric=False
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