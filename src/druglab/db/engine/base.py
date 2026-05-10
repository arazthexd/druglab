from __future__ import annotations

from typing import Any, Self, Iterator
from abc import ABC, abstractmethod
from pathlib import Path

import pyarrow as pa

from .utils import ReadOptions, WriteOptions, EngineCapabilities, DatasetInfo

# ---------------------------------------------------------------------------
# BASE ENGINE
# ---------------------------------------------------------------------------
 
class BaseEngine(ABC):
    """
    The universal contract every engine must honour.
 
    Concrete engines implement the `_*` private methods; the public API adds
    validation, connection-state checks, and logging around them.
    """

    # ------------------------------------------------------------------
    # Identity & capabilities
    # ------------------------------------------------------------------
 
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name, e.g. 'pandas', 'duckdb'."""
 
    @property
    @abstractmethod
    def capabilities(self) -> EngineCapabilities:
        """Declare what this engine supports."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
 
    @abstractmethod
    def connect(self) -> None:
        """
        Open connections, allocate resources, validate credentials.
        Called automatically by __enter__.
        """
 
    @abstractmethod
    def disconnect(self) -> None:
        """
        Release connections and flush any in-flight writes.
        Called automatically by __exit__.
        """
 
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """True if connect() has been called and disconnect() has not."""
 
    def __enter__(self) -> Self:
        self.connect()
        return self
 
    def __exit__(self, *_: Any) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Core I/O — Arrow as the main language
    # ------------------------------------------------------------------
 
    @abstractmethod
    def _read_arrow(self, dataset: str, options: ReadOptions) -> pa.Table:
        """
        Return the dataset as a pyarrow.Table.
        Engines that cannot produce Arrow natively must convert internally.
        """
 
    @abstractmethod
    def _write_arrow(self, dataset: str, table: pa.Table, options: WriteOptions) -> None:
        """
        Persist a pyarrow.Table.
        `dataset` is the logical name (table name, file path, key, etc.).
        """
 
    def read(self, dataset: str, options: ReadOptions | None = None) -> pa.Table:
        """Public read — returns a pyarrow.Table."""
        self._assert_connected()
        return self._read_arrow(dataset, options or ReadOptions())
 
    def write(self, dataset: str, table: pa.Table, options: WriteOptions | None = None) -> None:
        """Public write — accepts a pyarrow.Table."""
        self._assert_connected()
        self._write_arrow(dataset, table, options or WriteOptions())

    # ------------------------------------------------------------------
    # Schema & metadata
    # ------------------------------------------------------------------
 
    @abstractmethod
    def info(self, dataset: str) -> DatasetInfo:
        """
        Return schema and lightweight stats without reading all data.
        Engines should implement this cheaply (e.g. read Parquet footer,
        query INFORMATION_SCHEMA, inspect Zarr .zmetadata).
        """
 
    @abstractmethod
    def list_datasets(self) -> list[str]:
        """
        Return all dataset names visible to this engine
        (tables, files, arrays, keys …).
        """
 
    @abstractmethod
    def exists(self, dataset: str) -> bool:
        """Return True if the dataset exists."""

    # ------------------------------------------------------------------
    # Streaming / batched reads  (optional — check capabilities first)
    # ------------------------------------------------------------------
 
    def stream(
        self,
        dataset: str,
        options: ReadOptions | None = None,
    ) -> Iterator:
        """
        Yield pyarrow.RecordBatches one at a time.
        Default implementation reads everything and then yields batches;
        engines with native streaming should override this.
        """
        opts = options or ReadOptions()
        table = self.read(dataset, opts)
        batch_size = opts.batch_size or len(table)
        for batch in table.to_batches(max_chunksize=batch_size):
            yield batch
 
    # ------------------------------------------------------------------
    # Mutations  (optional — check capabilities first)
    # ------------------------------------------------------------------
 
    def append(self, dataset: str, table, options: WriteOptions | None = None) -> None:
        """Append rows to an existing dataset."""
        opts = WriteOptions(**(vars(options) if options else {}))
        opts.mode = "append"
        self.write(dataset, table, opts)
 
    def delete(self, dataset: str, filters: list[tuple] | None = None) -> int:
        """
        Delete rows matching `filters`, or the entire dataset if None.
        Returns the number of rows deleted (-1 if unknown).
        Engines that support deletes must override this.
        """
        raise NotImplementedError(
            f"{self.name} does not support row-level deletes. "
            "Check engine.capabilities.supports_delete before calling."
        )
 
    def upsert(self, dataset: str, table, key_columns: list[str]) -> None:
        """Insert or update rows identified by `key_columns`."""
        raise NotImplementedError(
            f"{self.name} does not support upserts. "
            "Check engine.capabilities.supports_upsert before calling."
        )

    # ------------------------------------------------------------------
    # Native execution escape hatch
    # ------------------------------------------------------------------
 
    def execute(self, query: str, parameters: dict[str, Any] | None = None):
        """
        Run a backend-native query (SQL, pandas query string, etc.)
        and return results as a pyarrow.Table.
 
        This is intentionally *not* abstract — not all engines support it —
        but engines that do (DuckDB, SQLite, …) should override it.
        """
        raise NotImplementedError(
            f"{self.name} does not expose a native query interface. "
            "Check engine.capabilities.supports_sql before calling."
        )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
 
    def __repr__(self) -> str:
        status = "connected" if self.is_connected else "disconnected"
        return f"<{self.__class__.__name__} [{status}]>"
 
    def _assert_connected(self) -> None:
        if not self.is_connected:
            raise RuntimeError(
                f"{self.name} engine is not connected. "
                "Call connect() or use the engine as a context manager."
            )
 
    def _require_capability(self, flag: str) -> None:
        if not getattr(self.capabilities, flag):
            raise NotImplementedError(
                f"{self.name} does not support '{flag}'. "
                f"Check engine.capabilities.{flag} before calling."
            )
        
# ---------------------------------------------------------------------------
# IN-MEMORY ENGINE (abstract mid-layer)
# ---------------------------------------------------------------------------
 
class InMemoryEngine(BaseEngine, ABC):
    """
    Base for engines whose data lives in process memory.
 
    **Extra contract**
    - Must expose the raw backend object via `.backend` for power users.
    - Must support `.clear()` to free memory.
    - Persistence is always False; remote is always False.
    """
 
    @property
    @abstractmethod
    def backend(self) -> Any:
        """
        The raw backend object, e.g. a dict of DataFrames for PandasEngine.
        Useful for power users who want to reach past the abstraction layer.
        """
 
    @abstractmethod
    def clear(self, dataset: str | None = None) -> None:
        """
        Drop all data for `dataset`, or everything if dataset is None.
        Useful for resetting state between tests.
        """
 
    @abstractmethod
    def memory_usage(self, dataset: str | None = None) -> int:
        """Return approximate memory usage in bytes."""
 
    # In-memory engines are always "connected" — no I/O to open.
    def connect(self) -> None:
        pass
 
    def disconnect(self) -> None:
        pass
 
    @property
    def is_connected(self) -> bool:
        return True
 
 
# ---------------------------------------------------------------------------
# ON-DISK ENGINE (abstract mid-layer)
# ---------------------------------------------------------------------------
 
class OnDiskEngine(BaseEngine, ABC):
    """
    Base for engines that persist data on the local filesystem.
 
    **Extra contract**
    - Construction takes a `root` path; all dataset paths are relative to it.
    - Exposes `.flush()` and `.compact()` for explicit durability control.
    - Exposes `.dataset_path()` so callers can hand off the raw path to other
      tools without going through the engine API.
    """
 
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._connected = False
 
    @property
    def root(self) -> Path:
        return self._root
 
    def dataset_path(self, dataset: str) -> Path:
        """Return the absolute filesystem path for a dataset."""
        return self._root / dataset
 
    @abstractmethod
    def flush(self) -> None:
        """
        Force any buffered writes to durable storage.
        For engines with write-behind caches (e.g. Zarr chunk buffers).
        """
 
    def compact(self, dataset: str) -> None:
        """
        Rewrite fragmented files into a single, optimally-sized file.
        Optional; engines that support it (DuckDB VACUUM, Parquet rewrite)
        should override this.
        """
        # Default no-op — override in engines that benefit from compaction.
 
    @property
    def is_connected(self) -> bool:
        return self._connected
 
 
# ---------------------------------------------------------------------------
# CLOUD ENGINE (abstract mid-layer — placeholder for future use)
# ---------------------------------------------------------------------------
 
class CloudEngine(BaseEngine, ABC):
    """
    Base for engines backed by remote object stores (S3, GCS, Azure Blob).
 
    **Extra contract**
    - Construction takes a URI (e.g. s3://bucket/prefix).
    - Credential resolution is backend-specific (boto3 chain, GCP ADC, …);
      the base class does not touch credentials.
    - Async I/O is recommended — engines should expose `aread` / `awrite`
      if the backend supports it.
    """
 
    def __init__(self, uri: str) -> None:
        self._uri = uri
        self._connected = False
 
    @property
    def uri(self) -> str:
        return self._uri
 
    @property
    def is_connected(self) -> bool:
        return self._connected
 
    @abstractmethod
    async def aread(self, dataset: str, options: ReadOptions | None = None):
        """Async read — returns a pyarrow.Table."""
 
    @abstractmethod
    async def awrite(self, dataset: str, table, options: WriteOptions | None = None) -> None:
        """Async write — accepts a pyarrow.Table."""