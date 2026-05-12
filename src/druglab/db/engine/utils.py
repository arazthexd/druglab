from __future__ import annotations
 
import abc
import asyncio
import os
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import pyarrow as pa
import pyarrow.dataset as pad
import numpy as np

# ---------------------------------------------------------------------------
# Capability descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EngineCapabilities:
    """
    Declares what a concrete engine actually supports.
    Callers should check this before calling optional methods.
    """
    # Data movement
    supports_arrow: bool = True          # pyarrow.Table I/O (always True for BaseEngine)
    supports_pandas: bool = False        # native pandas I/O
    supports_numpy: bool = False         # native numpy I/O
 
    # Query model
    supports_sql: bool = False           # SQL via execute()
    supports_filter: bool = True         # predicate push-down / row filtering
    supports_projection: bool = True     # column selection
 
    # Mutations
    supports_append: bool = False        # incremental writes
    supports_upsert: bool = False        # update-or-insert semantics
    supports_delete: bool = False        # row-level deletes
 
    # Transactions / ACID
    supports_transactions: bool = False
 
    # Streaming / chunked reads
    supports_streaming: bool = False
 
    # Persistence
    is_persistent: bool = False          # data survives process restart
    is_remote: bool = False              # data lives outside the local machine
 
    # Schema evolution
    supports_schema_evolution: bool = False

# ---------------------------------------------------------------------------
# Read options
# ---------------------------------------------------------------------------

@dataclass
class ReadOptions:
    """Options that apply across all engines for a read operation."""
    columns: list[str] | None = None        # column projection (None = all)
    filters: list[tuple] | None = None      # list of (col, op, val) triples
    limit: int | None = None                # max rows to return
    offset: int = 0                         # skip first N rows
    batch_size: int = 128_000               # rows per batch in streaming reads

# ---------------------------------------------------------------------------
# Write options
# ---------------------------------------------------------------------------

class WriteMode(str, Enum):
    """
    Describes how incoming rows are merged with any existing data.
 
    OVERWRITE
        Replace the entire dataset with the incoming data.
        Existing rows are discarded regardless of content.
 
    APPEND
        Add incoming rows after existing rows with no deduplication check.
        Use when you know the incoming rows are new (e.g. time-series).
        Will create duplicates if the same rows are written twice.
 
    UPSERT
        For each incoming row: if a row with the same value of `key_columns`
        already exists, update it; otherwise insert it.
        Requires WriteOptions.key_columns to be set.
        Error if key_columns is empty.
 
    REPLACE_WHERE
        Replace only the rows that match WriteOptions.replace_filter,
        leaving all other rows untouched.
        Incoming data replaces the matched rows entirely (not column-by-column).
        Requires WriteOptions.replace_filter to be set.
        Analogous to Delta Lake's replaceWhere option.
    """
    OVERWRITE      = "overwrite"
    APPEND         = "append"
    UPSERT         = "upsert"
    REPLACE_WHERE  = "replace_where"

class IfExists(str, Enum):
    """
    What to do if the target dataset does or does not exist yet.
    This is a creation guard, orthogonal to WriteMode.
 
    REPLACE 
        Create the dataset if it doesn't exist;
        silently proceed if it does.
    ERROR               
        Raise if the dataset already exists.
        Useful for preventing accidental overwrites.
    IGNORE              
        Do nothing if the dataset already exists.
        The write is a no-op when the dataset is present.
    CREATE_ONLY         
        Raise if the dataset does NOT exist.
        Useful for enforcing that a schema was pre-created.
    """
    REPLACE     = "replace"
    ERROR       = "error"
    IGNORE      = "ignore"
    CREATE_ONLY = "create_only"

class SchemaEvolution(str, Enum):
    """
    How to handle schema mismatches between incoming data and existing data.
 
    STRICT      
        Incoming schema must exactly match existing schema.
        Extra or missing columns raise SchemaError.
    MERGE       
        Add new columns from the incoming schema to the existing one,
        filling missing values with null for existing rows.
        Columns present in existing but absent in incoming are kept
        and filled with null for the new rows.
    OVERWRITE   
        Adopt the incoming schema wholesale.
        Existing columns absent in incoming are dropped.
    """
    STRICT    = "strict"
    MERGE     = "merge"
    OVERWRITE = "overwrite"

class SchemaError(Exception):
    """Raised when incoming schema is incompatible with existing schema."""

@dataclass
class WriteOptions:
    """
    Controls how a write operation behaves.

    Attributes:
        mode (WriteMode): 
            How rows are merged.
        if_exists (IfExists): 
            Creation guard.
        schema_evolution (SchemaEvolution): 
            How schema mismatches are handled.
        key_columns (Optional[List[str]]): 
            Required when mode=UPSERT. The columns that identify a row uniquely. 
            Rows with matching key values are updated; others are inserted.
        replace_filter (Optional[List[Tuple[str, str, Any]]]): 
            Required when mode=REPLACE_WHERE. A (col, op, val) filter list — rows 
            matching this are replaced; rows not matching are kept unchanged.
        partition_by (Optional[List[str]]): 
            Hive-partition the output by these columns. Ignored by in-memory 
            engines (no filesystem).
        compression (Optional[str]): 
            Codec hint (e.g. "snappy", "zstd"). Ignored by in-memory engines 
            (no serialisation).
        max_rows_per_file (Optional[int]): 
            Split output across multiple files. Ignored by in-memory engines.

    These options below are hints to on-disk / cloud engines. In-memory engines 
    document why they ignore them.
    """
    mode: WriteMode = WriteMode.OVERWRITE
    if_exists: IfExists = IfExists.REPLACE
    schema_evolution: SchemaEvolution = SchemaEvolution.STRICT
 
    # Mode-specific parameters
    key_columns: list[str] = field(default_factory=list)
    replace_filter: list[tuple] | None = None
 
    # On-disk / cloud hints (no-ops for in-memory engines)
    partition_by: list[str] = field(default_factory=list)
    compression: str | None = None
    max_rows_per_file: int | None = None
 
    def __post_init__(self) -> None:
        if self.mode == WriteMode.UPSERT and not self.key_columns:
            raise ValueError("WriteMode.UPSERT requires key_columns to be set.")
        if self.mode == WriteMode.REPLACE_WHERE and self.replace_filter is None:
            raise ValueError("WriteMode.REPLACE_WHERE requires replace_filter to be set.")

# ---------------------------------------------------------------------------
# Schema wrapper
# ---------------------------------------------------------------------------
 
@dataclass(frozen=True)
class DatasetInfo:
    """Lightweight schema + stats — no data movement required."""
    name: str
    num_rows: int | None          # None if unknown without a full scan
    num_columns: int
    column_names: list[str]
    column_types: dict[str, str]  # column name → human-readable type string
    size_bytes: int | None        # None if not available
    extra: dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Schema helpers  (module-level so any engine layer can import them)
# ---------------------------------------------------------------------------

def apply_schema_evolution(
    existing: pa.Table,
    incoming: pa.Table,
    evolution: SchemaEvolution,
) -> tuple[pa.Table, pa.Table]:
    """
    Reconcile the schemas of *existing* and *incoming* according to *evolution*.

    Returns (existing_reconciled, incoming_reconciled) — both aligned to the
    agreed schema so the caller can safely concatenate or merge them.

    STRICT     → raise SchemaError if schemas differ at all.
    MERGE      → union of both schemas; fill missing columns with null.
    OVERWRITE  → adopt incoming schema; drop columns absent from incoming.

    Note for on-disk / cloud engines
    ---------------------------------
    This helper materialises full pa.Table objects and is appropriate for
    in-memory engines and small-scale on-disk engines.  Large on-disk engines
    (Parquet, DuckDB) should compare schemas via their footer /
    INFORMATION_SCHEMA without materialising tables, then implement column-
    level reconciliation directly rather than calling this function.  It
    remains the canonical reference for the intended semantics.
    """
    e_names = set(existing.schema.names)
    i_names = set(incoming.schema.names)

    if evolution == SchemaEvolution.STRICT:
        if existing.schema != incoming.schema:
            raise SchemaError(
                f"Schema mismatch (SchemaEvolution.STRICT).\n"
                f"  Existing : {existing.schema}\n"
                f"  Incoming : {incoming.schema}\n"
                f"  Only in existing : {e_names - i_names}\n"
                f"  Only in incoming : {i_names - e_names}"
            )
        return existing, incoming

    if evolution == SchemaEvolution.OVERWRITE:
        keep = [c for c in existing.schema.names if c in i_names]
        return existing.select(keep), incoming

    # MERGE: full outer union of columns.
    all_columns = list(existing.schema.names) + [
        c for c in incoming.schema.names if c not in e_names
    ]
    for col in i_names - e_names:
        dtype = incoming.schema.field(col).type
        existing = existing.append_column(
            col, pa.array([None] * existing.num_rows, type=dtype)
        )
    for col in e_names - i_names:
        dtype = existing.schema.field(col).type
        incoming = incoming.append_column(
            col, pa.array([None] * incoming.num_rows, type=dtype)
        )
    return existing.select(all_columns), incoming.select(all_columns)


def should_proceed_with_write(exists: bool, dataset: str, if_exists: IfExists) -> bool:
    """
    Enforce the IfExists policy before a write.

    Parameters
    ----------
    exists      Whether the dataset already exists.  The caller must resolve
                this using whatever method is cheapest for its storage backend
                (Parquet footer, information_schema, S3 HEAD request, …).
    dataset     Dataset name used in error messages only.
    if_exists   The policy to enforce.

    Returns True  → proceed with the write.
    Returns False → skip the write (IGNORE branch).
    Raises        → ERROR / CREATE_ONLY violations.

    Accepting a plain bool decouples the policy check from how existence is
    determined, letting each engine do the cheapest possible check without
    being forced through a generic self.exists() call.
    """
    if if_exists == IfExists.ERROR and exists:
        raise FileExistsError(
            f"Dataset '{dataset}' already exists (if_exists={IfExists.ERROR})."
        )
    if if_exists == IfExists.IGNORE and exists:
        return False
    if if_exists == IfExists.CREATE_ONLY and not exists:
        raise FileNotFoundError(
            f"Dataset '{dataset}' does not exist "
            f"(if_exists={IfExists.CREATE_ONLY}). "
            "Create the dataset first before writing to it."
        )
    return True

# ---------------------------------------------------------------------------
# AsyncEngineMixin
# ---------------------------------------------------------------------------

class AsyncEngineMixin:
    """
    Mixin that grants an engine class async-capable I/O methods and a
    bridge between the synchronous BaseEngine interface and async
    implementations.

    Usage
    -----
    Concrete cloud engines inherit from both their PersistentEngine subclass
    *and* this mixin, then implement aread / awrite.  The mixin provides
    _run_async() so that the mandatory synchronous BaseEngine methods
    (_scan, _write_reader) can delegate to the async path without
    requiring the caller to manage an event loop:

        class S3Engine(CloudEngine, AsyncEngineMixin):
            async def aread(self, dataset, options=None): ...
            async def awrite(self, dataset, data, options=None): ...

            # Fulfil the synchronous BaseEngine contract by bridging:
            def _scan(self, dataset, options):
                tbl = self._run_async(self._afetch_table(dataset, options))
                return pad.dataset(tbl).scanner()

            def _write_reader(self, dataset, reader, options):
                self._run_async(self.awrite(dataset, reader, options))

    Event-loop strategy
    -------------------
    _run_async() always runs the coroutine in a *fresh* event loop that is
    created and destroyed for each call.  This avoids the "cannot run a
    nested event loop" RuntimeError when synchronous code is called from
    inside an already-running loop (Jupyter notebooks, FastAPI handlers,
    Django async views).

    If you are already inside an async context, call aread / awrite directly
    rather than going through the synchronous bridge.

    For high-throughput engines a dedicated background thread with a
    persistent event loop is more efficient than creating a new loop per
    call.  Override _run_async() to adopt that pattern.
    """

    @staticmethod
    def _run_async(coro: Coroutine) -> Any:
        """
        Run *coro* to completion in a fresh, isolated event loop.

        Never call this from inside an already-running event loop.
        Use ``await engine.aread(...)`` from async contexts instead.
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def normalize_to_reader(data: Any) -> pa.RecordBatchReader:
    """
    Convert any supported input type to a pa.RecordBatchReader.
 
    Accepted types (checked with isinstance, not duck-typing):
        pa.Table               >> wrap in InMemoryDataset, get scanner, get reader
        pa.RecordBatchReader   >> pass through
        pa.dataset.Dataset     >> scanner with no filters, then reader
        pa.dataset.Scanner     >> directly to reader
        pd.DataFrame           >> Table >> reader (pandas is optional)
        np.ndarray (structured)>> Table >> reader
        dict of arrays/lists   >> Table >> reader
 
    Third-party types can be registered with register_reader_converter().
    """
    # 1. Already a reader.
    if isinstance(data, pa.RecordBatchReader):
        return data
 
    # 2. PyArrow Table >> wrap in dataset so we go through the same path.
    if isinstance(data, pa.Table):
        ds: pad.Dataset = pad.dataset(data)
        return ds.scanner().to_reader()
 
    # 3. Scanner → reader directly (preserves the attached query plan).
    if isinstance(data, pad.Scanner):
        return data.to_reader()
 
    # 4. Dataset → default scanner → reader.
    if isinstance(data, pad.Dataset):
        return data.scanner().to_reader()
 
    # 5. Pandas DataFrame (optional dependency).
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            ds = pad.dataset(pa.Table.from_pandas(data, preserve_index=False))
            return ds.scanner().to_reader()
    except ImportError:
        pass
 
    # 6. Numpy structured array (named dtype = column-oriented).
    if isinstance(data, np.ndarray) and data.dtype.names:
        tbl = pa.Table.from_pydict({name: data[name] for name in data.dtype.names})
        ds = pad.dataset(tbl)
        return ds.scanner().to_reader()
 
    # 7. Dict of arrays / lists.
    if isinstance(data, dict):
        ds = pad.dataset(pa.Table.from_pydict(data))
        return ds.scanner().to_reader()
 
    # 8. Registered third-party converters.
    for predicate, converter in _READER_CONVERTERS:
        if predicate(data):
            result = converter(data)
            if not isinstance(result, pa.RecordBatchReader):
                raise TypeError(
                    f"Converter for {type(data).__name__} must return "
                    f"pa.RecordBatchReader, got {type(result).__name__}."
                )
            return result
 
    raise TypeError(
        f"Cannot convert {type(data).__name__} to pa.RecordBatchReader. "
        "Register a converter with register_reader_converter()."
    )
 
 
_READER_CONVERTERS: list[tuple] = []
def register_reader_converter(predicate, converter) -> None:
    """
    Register a converter for a third-party type.
 
    predicate : callable(obj) -> bool
    converter : callable(obj) -> pa.RecordBatchReader
    """
    _READER_CONVERTERS.append((predicate, converter))
