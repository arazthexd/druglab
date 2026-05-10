
from typing import Any, Dict

from ..base import InMemoryEngine
from ..utils import DatasetInfo, ReadOptions, WriteOptions, EngineCapabilities

try:
    import pandas as pd
    PD_LOADED = True
except ImportError:
    PD_LOADED = False
    
class PandasEngine(InMemoryEngine):
    """
    In-memory engine backed by a dict of pandas DataFrames.
 
    Has the minimal implementations for in-memory engines:
    - _read_arrow / _write_arrow  (mandatory)
    - info / list_datasets / exists  (mandatory)
    - backend / clear / memory_usage  (InMemoryEngine contract)
    """
 
    def __init__(self) -> None:
        if not PD_LOADED:
            raise ImportError("PandasEngine requires pandas to be installed.")
        self._store: Dict[str, pd.DataFrame] = {}  # dataset name → pd.DataFrame
 
    # --- Identity ---
 
    @property
    def name(self) -> str:
        return "pandas"
 
    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_arrow=True,
            supports_pandas=True,
            supports_numpy=True,
            supports_sql=False,        # no SQL, but see execute() note below
            supports_filter=True,
            supports_projection=True,
            supports_append=True,
            supports_delete=True,
            supports_streaming=True,
            is_persistent=False,
        )
 
    # --- Backend access ---
 
    @property
    def backend(self) -> Dict[str, pd.DataFrame]:
        return self._store
 
    # --- Core I/O ---
 
    def _read_arrow(self, dataset: str, options: ReadOptions):
        import pyarrow as pa
        if dataset not in self._store:
            raise KeyError(f"Dataset '{dataset}' not found.")
        df = self._store[dataset]
 
        # Apply projection
        if options.columns:
            df = df[options.columns]
 
        # Apply filters: list of (col, op, val)
        if options.filters:
            mask = None
            op_map = {
                "==": lambda a, b: a == b,
                "!=": lambda a, b: a != b,
                ">":  lambda a, b: a > b,
                ">=": lambda a, b: a >= b,
                "<":  lambda a, b: a < b,
                "<=": lambda a, b: a <= b,
                "in": lambda a, b: a.isin(b),
            }
            for col, op, val in options.filters:
                condition = op_map[op](df[col], val)
                mask = condition if mask is None else (mask & condition)
            df = df[mask]
 
        # Apply offset / limit
        if options.offset:
            df = df.iloc[options.offset:]
        if options.limit is not None:
            df = df.iloc[: options.limit]
 
        return pa.Table.from_pandas(df, preserve_index=False)
 
    def _write_arrow(self, dataset: str, table, options: WriteOptions) -> None:
        df = table.to_pandas()
        if options.mode == "error_if_exists" and dataset in self._store:
            raise FileExistsError(f"Dataset '{dataset}' already exists.")
        if options.mode == "append" and dataset in self._store:
            import pandas as pd
            self._store[dataset] = pd.concat(
                [self._store[dataset], df], ignore_index=True
            )
        else:
            self._store[dataset] = df
 
    # --- Schema & metadata ---
 
    def info(self, dataset: str) -> DatasetInfo:
        if dataset not in self._store:
            raise KeyError(f"Dataset '{dataset}' not found.")
        df = self._store[dataset]
        return DatasetInfo(
            name=dataset,
            num_rows=len(df),
            num_columns=len(df.columns),
            column_names=list(df.columns),
            column_types={c: str(t) for c, t in df.dtypes.items()},
            size_bytes=df.memory_usage(deep=True).sum(),
        )
 
    def list_datasets(self) -> list[str]:
        return list(self._store.keys())
 
    def exists(self, dataset: str) -> bool:
        return dataset in self._store
 
    # --- InMemoryEngine contract ---
 
    def clear(self, dataset: str | None = None) -> None:
        if dataset is None:
            self._store.clear()
        else:
            self._store.pop(dataset, None)
 
    def memory_usage(self, dataset: str | None = None) -> int:
        if dataset is not None:
            return int(self._store[dataset].memory_usage(deep=True).sum())
        return sum(
            int(df.memory_usage(deep=True).sum()) for df in self._store.values()
        )
 
    # --- Mutations ---
 
    def delete(self, dataset: str, filters: list[tuple] | None = None) -> int:
        if dataset not in self._store:
            raise KeyError(f"Dataset '{dataset}' not found.")
        if filters is None:
            n = len(self._store.pop(dataset))
            return n
        df = self._store[dataset]
        original_len = len(df)
        # Keep rows that do NOT match any filter
        read_opts = ReadOptions(filters=filters)
        import pyarrow as pa
        matching = self._read_arrow(dataset, read_opts).to_pandas()
        self._store[dataset] = df.loc[~df.index.isin(matching.index)]
        return original_len - len(self._store[dataset])
 
    # --- Native escape hatch (pandas query string) ---
 
    def execute(self, query: str, parameters: dict | None = None):
        """
        Accepts a pandas-style query string:  "age > 30 and city == 'NYC'"
        The `dataset` must be the first token before a pipe if you want
        multi-dataset ops — for single-table, pass "tablename | query".
        """
        import pyarrow as pa
        if "|" in query:
            dataset, expr = [s.strip() for s in query.split("|", 1)]
        else:
            raise ValueError("PandasEngine.execute() expects 'dataset | query_expr'.")
        df = self._store[dataset].query(expr)
        return pa.Table.from_pandas(df, preserve_index=False)