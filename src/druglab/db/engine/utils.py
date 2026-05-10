from __future__ import annotations
 
import abc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

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
# Read / Write option bundles
# ---------------------------------------------------------------------------

@dataclass
class ReadOptions:
    """Options that apply across all engines for a read operation."""
    columns: list[str] | None = None        # column projection (None = all)
    filters: list[tuple] | None = None      # list of (col, op, val) triples
    limit: int | None = None                # max rows to return
    offset: int = 0                         # skip first N rows
    batch_size: int | None = None           # rows per batch in streaming reads
 
 
@dataclass
class WriteOptions:
    """Options that apply across all engines for a write operation."""
    mode: str = "overwrite"                 # "overwrite" | "append" | "error_if_exists"
    schema_evolution: str = "strict"        # "strict" | "merge" | "overwrite"
    partition_by: list[str] = field(default_factory=list)
    compression: str | None = None          # hint; backend may ignore

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
