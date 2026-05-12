"""
Abstract engine interface.

This module defines the abstract contracts
for moving data in and out of various storage backends.

**Hierarchy**
 
`BaseEngine`
    Universal contract. Every engine, regardless of storage medium, must
    implement these methods. Includes delete_rows and drop as base contracts.
 
`InMemoryEngine(BaseEngine)`
    Adds: no-op lifecycle, clear(), memory_usage(), backend property,
    shared helpers: _collect_reader, _apply_schema_evolution, _check_if_exists.
 
`PersistentEngine(BaseEngine)`
    Abstract mid-layer shared by OnDiskEngine and CloudEngine.
    Adds: connect/disconnect as abstract (persistence requires real I/O),
    is_connected tracking, flush().
 
`OnDiskEngine(PersistentEngine)`
    Adds: root path, dataset_path(), compact().
 
`CloudEngine(PersistentEngine)`
    Adds: URI, region/credential hints, async read/write contract.
"""


from __future__ import annotations

from typing import Any, Self, Iterator
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pad

from .utils import ReadOptions, WriteOptions, DatasetInfo
from .utils import normalize_to_reader
from .utils import WriteMode, IfExists, SchemaEvolution, SchemaError
from .utils import apply_schema_evolution, should_proceed_with_write
from .utils import AsyncEngineMixin

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
    # Identity
    # ------------------------------------------------------------------
 
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name, e.g. 'pandas', 'duckdb'."""

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
    # Core read
    # ------------------------------------------------------------------

    @abstractmethod
    def _scan(self, dataset: str, options: "ReadOptions") -> pad.Scanner:
        """
        Return a Scanner with filters and projections attached but NO data read.
 
        For on-disk engines (DuckDB, Parquet, Zarr): produce a native scanner.
        For in-memory engines (pandas, numpy): wrap in InMemoryDataset.
 
        The Scanner is engine-internal.  It exists so:

            1.  filters are pushed into the source before data crosses boundaries
            2.  other engines can consume it without going through Python objects 
                DuckDB can FROM a Scanner directly via conn.execute("SELECT ... FROM scanner"))
        """

    def read(self, dataset: str, options: "ReadOptions | None" = None) -> pa.RecordBatchReader:
        """
        Public streaming read.  Returns a RecordBatchReader.
 
        Nothing is loaded into memory until the caller iterates batches.
        The caller controls memory pressure by choosing batch_size in options.
 
        **Usage**
        ```python
        reader = engine.read("molecules")
        for batch in reader:                     # pa.RecordBatch per iteration
            process(batch)
 
        # Or read everything at once (explicit opt-in):
        table = engine.to_table("molecules")

        # Which is just the short hand for this:
        table = engine.read("molecules").to_table()
        ```
        """
        opts = options or ReadOptions()
        scanner = self._scan(dataset, opts)
        return scanner.to_reader()
    
    def to_table(self, dataset: str, options: "ReadOptions | None" = None) -> pa.Table:
        """
        Materialise the entire output dataset into a pa.Table.
 
        This is an explicit, named operation — the caller is saying
        "I know this fits in memory and I want a Table."
        It is NOT the default return type.
        """
        opts = options or ReadOptions()
        scanner = self._scan(dataset, opts)
        return scanner.to_table()
    
    def to_pandas(self, dataset: str, options: "ReadOptions | None" = None):
        """Materialise as a pandas DataFrame.  Explicit opt-in."""
        return self.to_table(dataset, options).to_pandas()
    
    # ------------------------------------------------------------------
    # Core write
    # ------------------------------------------------------------------

    @abstractmethod
    def _write_reader(self, dataset: str, reader: pa.RecordBatchReader, options: "WriteOptions") -> None:
        """
        Persist data arriving as a RecordBatchReader.
 
        Engines consume the reader in batches — they never need to load the
        full dataset into memory.  This is the only write path engines implement.
 
        The reader may be consumed exactly once.
        """
 
    def write(self, dataset: str, data: Any, options: "WriteOptions | None" = None) -> None:
        """
        Write data to a dataset.
 
        Accepts pa.Table, pa.RecordBatchReader, pa.dataset.Scanner,
        pa.dataset.Dataset, pd.DataFrame, np.ndarray (structured), or dict.
        Normalises to RecordBatchReader internally — engines only see one type.
        """
        opts = options or WriteOptions()
        reader = normalize_to_reader(data)
        self._write_reader(dataset, reader, opts)

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
    # Scanner first-class escape hatch
    # ------------------------------------------------------------------
 
    def scanner(
        self,
        dataset: str,
        options: "ReadOptions | None" = None,
    ) -> pad.Scanner:
        """
        Return the Scanner directly for advanced use cases:
 
        - Pass to another engine:    duckdb_engine.from_scanner(scanner)
        - Inspect the schema:        scanner.projected_schema
        - Count without reading:     scanner.count_rows()
        - Write to Parquet directly: pad.write_dataset(scanner, "/out", format="parquet")
 
        Most callers should use read() or to_table() instead.
        """
        return self._scan(dataset, options or ReadOptions())
 
    def from_scanner(self, dataset: str, scanner: pad.Scanner, options: "WriteOptions | None" = None) -> None:
        """
        Write directly from a Scanner produced by another engine.
        Avoids materialising a Table as an intermediary.
 
        Example: copy a filtered DuckDB result into a Parquet file engine:
        ```python
            parquet_engine.from_scanner("output", duckdb_engine.scanner("molecules",
                ReadOptions(filters=[("mw", "<=", 500)])))
        ```
        """
        self._write_reader(dataset, scanner.to_reader(), options or WriteOptions())

    # ------------------------------------------------------------------
    # Delete — base contract, present on every engine
    # ------------------------------------------------------------------

    @abstractmethod
    def delete_rows(
        self,
        dataset: str,
        row_filter: list[tuple] | None = None,
    ) -> int:
        """
        Delete rows from a dataset.
 
        Parameters
        ----------
        dataset     Target dataset.
        row_filter  (col, op, val) filter list. Matching rows are deleted.
                    If None, ALL rows are deleted (dataset is emptied but
                    its schema is preserved — analogous to SQL TRUNCATE).
                    This is intentionally distinct from drop().
 
        Returns the number of rows deleted.
        """

    @abstractmethod
    def drop(self, dataset: str) -> None:
        """
        Remove an entire dataset including its schema and all associated
        metadata (analogous to SQL DROP TABLE).

        Raises KeyError if the dataset does not exist.
        Use exists() to check first if unsure.

        For column removal use drop_columns().
        """

    @abstractmethod
    def drop_columns(self, dataset: str, columns: list[str]) -> None:
        """
        Remove specific columns from a dataset, leaving all rows and the
        remaining schema intact (analogous to SQL ALTER TABLE DROP COLUMN).

        Raises KeyError   if the dataset does not exist.
        Raises ValueError if any name in *columns* is not present in the schema.
        """

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
        
# ---------------------------------------------------------------------------
# IN-MEMORY ENGINE (abstract mid-layer)
# ---------------------------------------------------------------------------
 
class InMemoryEngine(BaseEngine, ABC):
    """
    Abstract base for engines whose data lives entirely in process memory.
 
    **Overrides**
    - connect / disconnect / is_connected — no-ops; always ready.
 
    **Adds**
    - memory_usage()  Approximate bytes consumed.
    - backend         Property exposing raw internal storage for power users.
    - _collect_reader / _apply_schema_evolution / _check_if_exists
                      Shared write-time helpers used by all in-memory engines.
                      
    NOTE: _apply_schema_evolution and _check_if_exists delegate to module-level 
    functions in utils so on-disk engines can reuse the same logic without 
    inheriting from this class.
    """

    # ------------------------------------------------------------------
    # Lifecycle — no-ops for in-memory
    # ------------------------------------------------------------------
 
    def connect(self) -> None:
        """No-op. In-memory engines require no connection setup."""
 
    def disconnect(self) -> None:
        """No-op. In-memory engines have no connection to close."""
 
    @property
    def is_connected(self) -> bool:
        return True
 
    def __enter__(self) -> "InMemoryEngine":
        return self
 
    def __exit__(self, *_: Any) -> None:
        pass

    # ------------------------------------------------------------------
    # Abstract contract
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def backend(self) -> Any:
        """Raw internal storage (e.g. dict[str, pa.Table])."""
 
    @abstractmethod
    def memory_usage(self, dataset: str | None = None) -> int:
        """Return approximate memory usage in bytes for one or all datasets."""

    # ------------------------------------------------------------------
    # Shared helpers available to all in-memory subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_reader(reader: pa.RecordBatchReader) -> pa.Table:
        """
        Drain a RecordBatchReader into a pa.Table.
 
        In-memory engines always need the full table before they can do
        anything with it (merge, upsert, etc.), so this helper avoids
        repeating the same pattern in every subclass.
        """
        return pa.Table.from_batches(list(reader), schema=reader.schema)

    # Delegate to the module-level helper
    _apply_schema_evolution = staticmethod(apply_schema_evolution)

    def _should_proceed_with_write(self, dataset: str, if_exists: IfExists) -> bool:
        """
        Enforce the IfExists policy before a write.

        Resolves existence via self.exists() and delegates the policy logic
        to utils.should_proceed_with_write().

        Returns True  → proceed with the write.
        Returns False → skip the write (IGNORE case).
        Raises        → ERROR / CREATE_ONLY violations.
        """
        return should_proceed_with_write(
            self.exists(dataset), dataset, if_exists
        )
    
# ---------------------------------------------------------------------------
# PERSISTENT ENGINE (shared mid-layer for OnDisk and Cloud)
# ---------------------------------------------------------------------------
 
class PersistentEngine(BaseEngine, ABC):
    """
    Abstract mid-layer for engines that persist data outside the process.
 
    **Shared behaviour**
    - connect / disconnect are abstract — persistent engines always have
      real I/O to set up (file handles, network connections, auth tokens).
    - is_connected is tracked via a _connected flag set by subclasses.
    - flush() is abstract — persistent engines must expose explicit durability
      control since data lives outside the process.
 
    **Not included here**
    - Filesystem paths (OnDiskEngine)
    - URIs and async I/O (CloudEngine)
    """

    def __init__(self) -> None:
        self._connected: bool = False
 
    @property
    def is_connected(self) -> bool:
        return self._connected
 
    @abstractmethod
    def connect(self) -> None:
        """
        Open connections, acquire file handles, resolve credentials.
        Must set self._connected = True on success.
        """
 
    @abstractmethod
    def disconnect(self) -> None:
        """
        Flush in-flight writes and release all resources.
        Must set self._connected = False.
        """
 
    @abstractmethod
    def flush(self) -> None:
        """
        Force any buffered or in-flight writes to durable storage.
        For engines with write-behind caches (Zarr chunk buffers,
        DuckDB WAL, object-store multipart uploads).
        """
    
# ---------------------------------------------------------------------------
# ON-DISK ENGINE
# ---------------------------------------------------------------------------

class OnDiskEngine(PersistentEngine, ABC):
    """
    Abstract base for engines that persist data on the local filesystem.
 
    **Adds**
    
    - root          The directory under which all datasets live.
    - dataset_path  Resolve a dataset name to an absolute Path.
    - compact()     Rewrite fragmented files into optimally-sized ones.
                    Default is a documented no-op; engines that benefit
                    (DuckDB VACUUM, Parquet rewrite) override it.
 
    **Lifecycle**

    connect() should validate that self.root exists and is writable,
    open any persistent connections (e.g. DuckDB file handle), and
    set self._connected = True.
 
    disconnect() should call flush() then release all handles.
    """
 
    def __init__(self, root: str | Path) -> None:
        super().__init__()
        self._root = Path(root)
 
    @property
    def root(self) -> Path:
        """Root directory. All dataset paths are relative to this."""
        return self._root
 
    def dataset_path(self, dataset: str) -> Path:
        """
        Resolve a logical dataset name to an absolute filesystem path.
        Exposes the raw path so other tools (rsync, S3 sync, nbformat)
        can work with the data directly without going through the engine.
        """
        return self._root / dataset
 
    def compact(self, dataset: str) -> None:
        """
        Rewrite fragmented storage into optimally-sized files.
 
        Default: documented no-op — not all on-disk engines fragment.
        Override in engines where compaction is meaningful:
          DuckDB  → VACUUM
          Parquet → rewrite row groups
          Zarr    → rechunk
        """
 
    def connect(self) -> None:
        """
        Default implementation: validate root exists and is writable.
        Engines with additional setup (DuckDB connection, file locks)
        should call super().connect() then do their own setup.
        """
        self._root.mkdir(parents=True, exist_ok=True)
        if not self._root.is_dir():
            raise NotADirectoryError(f"root is not a directory: {self._root}")
        self._connected = True
 
    def disconnect(self) -> None:
        """Default: flush then mark disconnected."""
        self.flush()
        self._connected = False

    def flush(self) -> None:
        """
        Default no-op flush for engines that write atomically
        (e.g. write-then-rename).  Override for engines with write-behind
        caches (DuckDB WAL, Zarr chunk buffers).

        This default is provided so that OnDiskEngine.disconnect() — which
        calls flush() — does not leave the abstract method unresolved in
        subclasses that do not need explicit flushing.  Without it, any
        concrete subclass that writes atomically and omits flush() would
        raise TypeError at instantiation rather than at the point where
        flushing is actually relevant.
        """

# ---------------------------------------------------------------------------
# CLOUD ENGINE
# ---------------------------------------------------------------------------
 
class CloudEngine(PersistentEngine, ABC):
    """
    Abstract base for engines backed by remote object stores
    (S3, GCS, Azure Blob, etc.).
 
    Adds
    ----
    - uri           The root URI (e.g. 's3://bucket/prefix').
    - region        Optional region hint passed to the SDK.
    - dataset_uri   Resolve a dataset name to a full URI.
    - aread / awrite  Async variants of read/write (preferred for cloud I/O).

    Sync / Async contract
    ----------------------
    CloudEngine inherits AsyncEngineMixin, which provides _run_async() as a
    bridge between the synchronous BaseEngine interface and async
    implementations.  Concrete cloud engines must:

    1.  Implement aread() and awrite().
    2.  Implement the synchronous _scan() and _write_reader() required by
        BaseEngine by delegating to the async counterparts via _run_async():

            def _scan(self, dataset, options):
                tbl = self._run_async(self._afetch_table(dataset, options))
                return pad.dataset(tbl).scanner()

            def _write_reader(self, dataset, reader, options):
                self._run_async(self.awrite(dataset, reader, options))

    3.  Never call _run_async() from inside an already-running event loop.
        Use aread() / awrite() directly in async contexts.

    This keeps LSP intact: CloudEngine IS-A BaseEngine and all synchronous
    methods work, while the async path remains the preferred and efficient
    route for cloud I/O.

    Credentials
    -----------
    Credential resolution is intentionally left to the backend SDK:
      AWS  → boto3 credential chain (env, ~/.aws, instance profile)
      GCP  → Application Default Credentials
      Azure→ DefaultAzureCredential
    The engine should not accept raw secrets as constructor arguments.
    Use environment variables or SDK-native credential providers instead.
 
    Lifecycle
    ---------
    connect() should initialise the SDK client and validate that the
    root URI is reachable (e.g. a lightweight HEAD / list operation).
    disconnect() should flush multipart uploads and release SDK clients.
    """
 
    def __init__(self, uri: str, region: str | None = None) -> None:
        super().__init__()
        self._uri = uri.rstrip("/")
        self._region = region
 
    @property
    def uri(self) -> str:
        """Root URI. All dataset URIs are relative to this."""
        return self._uri
 
    @property
    def region(self) -> str | None:
        return self._region
 
    def dataset_uri(self, dataset: str) -> str:
        """Resolve a logical dataset name to a full URI."""
        return f"{self._uri}/{dataset}"
 
    @abstractmethod
    async def aread(
        self,
        dataset: str,
        options: ReadOptions | None = None,
    ) -> pa.RecordBatchReader:
        """
        Async streaming read. Returns a RecordBatchReader.
 
        Use when the caller is in an async context (web server, async
        pipeline) and cannot afford to block the event loop on network I/O.
        """
 
    @abstractmethod
    async def awrite(
        self,
        dataset: str,
        data: Any,
        options: WriteOptions | None = None,
    ) -> None:
        """
        Async write. Accepts the same input types as write().
 
        Implementations should use multipart / resumable uploads where
        available so large datasets don't require holding everything in
        memory during the upload.
        """
 
    async def adelete_rows(
        self,
        dataset: str,
        row_filter: list[tuple] | None = None,
    ) -> int:
        """
        Async row deletion. Default: delegates to synchronous delete_rows().
 
        Cloud engines where deletion is expensive (requires rewriting
        Parquet files on S3) should override this with a proper async
        implementation or raise NotImplementedError with a clear message
        explaining the cost.
        """
        return self.delete_rows(dataset, row_filter)
 
    async def adrop(self, dataset: str) -> None:
        """Async dataset drop. Default: delegates to synchronous drop()."""
        self.drop(dataset)
 
    def flush(self) -> None:
        """
        Default: no-op for cloud engines that use atomic uploads.
        Engines with multipart upload buffers should override this to
        complete or abort any in-flight uploads.
        """
