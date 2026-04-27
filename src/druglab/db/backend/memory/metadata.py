"""In-memory metadata store."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from ...indexing import RowSelection
from ..base.stores import BaseMetadataStore

if TYPE_CHECKING:
    from druglab.db.indexing import INDEX_LIKE


__all__ = ["MemoryMetadataStore"]


class MemoryMetadataStore(BaseMetadataStore):
    def __init__(self, metadata: Optional[pd.DataFrame] = None, *, n_rows_hint: Optional[int] = None) -> None:
        if metadata is None:
            self._metadata = pd.DataFrame(index=range(n_rows_hint or 0))
        else:
            self._metadata = metadata

    def get_metadata(self, idx: Optional["INDEX_LIKE"] = None, cols: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        sel = RowSelection.from_raw(idx, len(self._metadata))
        df = self._metadata if sel.is_full else self._metadata.iloc[sel.positions]
        if cols is not None:
            if isinstance(cols, str):
                cols = [cols]
            df = df[cols]
        return df.reset_index(drop=True)

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

    def update_metadata(self, values: Union[pd.DataFrame, pd.Series, Dict[str, Any]], idx: Optional["INDEX_LIKE"] = None) -> None:
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
                raise KeyError(f"Column '{col}' does not exist in metadata. Use add_metadata_column.")
            if sel.is_full:
                self._metadata[col] = val
            else:
                self._metadata.iloc[sel.positions, self._metadata.columns.get_loc(col)] = val

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

    def gather_materialized_state(self, index_map: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if index_map is not None:
            return {"metadata": self._metadata.iloc[index_map].reset_index(drop=True).copy()}
        return {"metadata": self._metadata.copy()}

    def save(self, path: Path) -> None:
        if not self._metadata.empty:
            try:
                self._metadata.to_parquet(path / "metadata.parquet", index=False)
            except Exception:
                self._metadata.to_csv(path / "metadata.csv", index=False)
        else:
            pd.DataFrame(index=range(len(self._metadata.index))).to_csv(path / "metadata.csv", index=True)

    @classmethod
    def load(cls, path: Path) -> "MemoryMetadataStore":
        parquet_path = path / "metadata.parquet"
        csv_path = path / "metadata.csv"
        if parquet_path.exists():
            return cls(metadata=pd.read_parquet(parquet_path))
        if csv_path.exists():
            return cls(metadata=pd.read_csv(csv_path))
        print("WARNING: No metadata found when loading bundle.")
        return cls(metadata=pd.DataFrame())


MemoryMetadataMixin = MemoryMetadataStore