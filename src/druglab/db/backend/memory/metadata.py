"""In-memory metadata store with Just-In-Time format selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from ...indexing import RowSelection
from ..base.stores import BaseMetadataStore

if TYPE_CHECKING:
    from druglab.db.backend.overlay.deltas import MetadataDelta
    from druglab.db.indexing import INDEX_LIKE


__all__ = ["MemoryMetadataStore"]

_SUPPORTED_FORMATS = ("parquet", "csv")


class MemoryMetadataStore(BaseMetadataStore):
    def __init__(
        self,
        metadata: Optional[pd.DataFrame] = None,
        *,
        n_rows_hint: Optional[int] = None,
    ) -> None:
        if metadata is None:
            self._metadata = pd.DataFrame(index=range(n_rows_hint or 0))
        else:
            self._metadata = metadata
        self._journal: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_metadata(
        self,
        idx: Optional["INDEX_LIKE"] = None,
        cols: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        sel = RowSelection.from_raw(idx, len(self._metadata))
        df = self._metadata if sel.is_full else self._metadata.iloc[sel.positions]
        if cols is not None:
            if isinstance(cols, str):
                cols = [cols]
            df = df[cols]
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_metadata_column(
        self,
        name: str,
        value: Union[pd.Series, np.ndarray, List[Any]],
        idx: Optional["INDEX_LIKE"] = None,
        na: Any = None,
    ) -> None:
        sel = RowSelection.from_raw(idx, len(self._metadata))
        value = np.asarray(value)

        if sel.is_full:
            self._metadata[name] = value
            return

        resolved = sel.positions
        if na is None and np.issubdtype(value.dtype, np.integer):
            raise ValueError("na must be provided when populating integer columns")
        if na is None and np.issubdtype(value.dtype, np.floating):
            na = np.nan
        if value.shape[0] != len(resolved):
            raise ValueError(f"Expected {len(resolved)} values, got {value.shape[0]}")

        arr = np.full(len(self._metadata), na, dtype=np.asarray(value).dtype)
        arr[resolved] = value
        self._metadata[name] = arr

    def update_metadata(
        self,
        values: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
        idx: Optional["INDEX_LIKE"] = None,
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
                    f"Column '{col}' does not exist in metadata. Use add_metadata_column."
                )
            if sel.is_full:
                self._metadata[col] = val
            else:
                self._metadata.iloc[
                    sel.positions, self._metadata.columns.get_loc(col)
                ] = val

    def drop_metadata_columns(self, cols: Optional[Union[str, List[str]]] = None) -> None:
        if cols is None:
            self._metadata = pd.DataFrame(index=self._metadata.index)
        else:
            if isinstance(cols, str):
                cols = [cols]
            self._metadata = self._metadata.drop(columns=cols)

    def get_metadata_columns(self) -> List[str]:
        return list(self._metadata.columns)

    def n_rows(self) -> int:
        n_rows, n_columns = self._metadata.shape
        if n_rows == 0 and n_columns == 0:
            return len(self._metadata.index)
        return n_rows

    def append(self, data: Any) -> None:
        self._metadata = pd.concat([self._metadata, data], ignore_index=True)

    def gather_materialized_state(self, index_map: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if index_map is not None:
            return {"metadata": self._metadata.iloc[index_map].reset_index(drop=True).copy()}
        return {"metadata": self._metadata.copy()}

    # ------------------------------------------------------------------
    # Persistence — Just-In-Time format selection
    # ------------------------------------------------------------------

    def save(self, path: Path, format: str = "parquet", **kwargs) -> None:
        """
        Persist the metadata DataFrame.

        Parameters
        ----------
        path : Path
            Bundle root directory.
        format : {"parquet", "csv"}
            Serialization format chosen **at save time**.  Defaults to
            ``"parquet"`` for compact binary storage.  Pass ``"csv"`` to
            produce a human-readable text file.

        Raises
        ------
        ValueError
            If *format* is not one of the supported values.
        """
        if format not in _SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported metadata format '{format}'. "
                f"Choose one of: {_SUPPORTED_FORMATS}."
            )

        if format == "parquet":
            self._metadata.to_parquet(path / "metadata.parquet", index=True)
        else:
            self._metadata.to_csv(path / "metadata.csv", index=True)

    @classmethod
    def load(cls, path: Path, format: str = "parquet", **kwargs) -> "MemoryMetadataStore":
        """
        Restore the metadata DataFrame.

        Parameters
        ----------
        path : Path
            Bundle root directory.
        format : {"parquet", "csv"}
            Must match the format used when the bundle was saved.  The
            ``CompositeStorageBackend`` reads this from the manifest and
            passes it here automatically.
        """
        if format == "parquet":
            parquet_path = path / "metadata.parquet"
            if parquet_path.exists():
                return cls(metadata=pd.read_parquet(parquet_path))
        else:  # csv
            csv_path = path / "metadata.csv"
            if csv_path.exists():
                return cls(metadata=pd.read_csv(csv_path))

        print("WARNING: No metadata found when loading bundle.")
        return cls(metadata=None)

    # ------------------------------------------------------------------
    # Transaction protocol
    # ------------------------------------------------------------------

    def begin_transaction(self, delta: "MetadataDelta", index_map: np.ndarray) -> None:
        """
        Snapshot the rows/columns that will be mutated or dropped by *delta*.

        Journal structure::

            {
                "modified_cols": {
                    "<col>": {
                        "existed": bool,
                        "indices": np.ndarray | None,  # None → whole column snapshot
                        "old_values": np.ndarray | None,
                    }
                },
                "dropped_cols": {
                    "<col>": np.ndarray   # full column backup
                },
            }
        """
        self._journal = {"modified_cols": {}, "dropped_cols": {}}

        if delta.local is not None:
            for col in delta.local.columns:
                if col in delta.deleted_cols:
                    continue
                if col in self._metadata.columns:
                    self._journal["modified_cols"][col] = {
                        "existed": True,
                        "indices": index_map.copy(),
                        "old_values": self._metadata.iloc[index_map][col].values.copy(),
                    }
                else:
                    # New column; rollback must drop it
                    self._journal["modified_cols"][col] = {
                        "existed": False,
                        "indices": None,
                        "old_values": None,
                    }

        for col in delta.deleted_cols:
            if col in self._metadata.columns:
                self._journal["dropped_cols"][col] = self._metadata[col].copy()

    def commit_transaction(self, delta: "MetadataDelta", index_map: np.ndarray) -> None:
        """Apply *delta* to the store; journal must have been set by begin_transaction."""
        if self._journal is None:
            raise RuntimeError("Cannot commit without beginning a transaction first.")

        # Drop deleted columns
        if delta.deleted_cols:
            existing = set(self._metadata.columns)
            to_drop = list(delta.deleted_cols & existing)
            if to_drop:
                self.drop_metadata_columns(to_drop)

        # Apply local delta
        if delta.local is not None and not delta.local.empty:
            existing_cols = set(self._metadata.columns)
            for col in delta.local.columns:
                if col in delta.deleted_cols:
                    continue
                val = delta.local[col].values
                if col in existing_cols:
                    self.update_metadata({col: val}, idx=index_map)
                else:
                    self.add_metadata_column(col, val, idx=index_map)

        # NOTE: Do NOT clear self._journal here.
        # apply_deltas() clears it only after ALL three stores commit successfully.

    def rollback_transaction(self) -> None:
        """Restore rows/columns backed up in begin_transaction."""
        if self._journal is None:
            return

        # Restore modified columns
        for col, entry in self._journal["modified_cols"].items():
            if not entry["existed"]:
                # Column was brand new — remove any partial write
                if col in self._metadata.columns:
                    self._metadata.drop(columns=[col], inplace=True)
            else:
                # Restore the specific rows
                if col in self._metadata.columns:
                    col_pos = self._metadata.columns.get_loc(col)
                    self._metadata.iloc[entry["indices"], col_pos] = entry["old_values"]
                else:
                    # Column was dropped mid-commit and not yet restored; re-add it
                    self._metadata[col] = np.nan
                    col_pos = self._metadata.columns.get_loc(col)
                    self._metadata.iloc[entry["indices"], col_pos] = entry["old_values"]

        # Restore dropped columns
        for col, series in self._journal["dropped_cols"].items():
            self._metadata[col] = series.values

        self._journal = None


MemoryMetadataMixin = MemoryMetadataStore