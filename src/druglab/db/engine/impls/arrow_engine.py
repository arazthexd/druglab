from __future__ import annotations
 
import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
 
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pad
import pyarrow.compute as pc

from ..base import InMemoryEngine
from ..utils import ReadOptions, WriteOptions, WriteMode, SchemaEvolution, DatasetInfo
from ..utils import normalize_to_reader
from ..utils import SchemaError

def _build_arrow_expression(filters: list[tuple]) -> pc.Expression:
    op_map = {
        "==": pc.equal,
        "!=": pc.not_equal,
        "<":  pc.less,
        "<=": pc.less_equal,
        ">":  pc.greater,
        ">=": pc.greater_equal,
    }
    expr = None
    for col, op, val in filters:
        if op in ("in", "not in"):
            field_expr = pc.field(col).isin(val)
            if op == "not in":
                field_expr = ~field_expr
        else:
            field_expr = op_map[op](pc.field(col), val)
        expr = field_expr if expr is None else (expr & field_expr)
    return expr

class ArrowInMemoryEngine(InMemoryEngine):
    """
    In-memory engine backed by pa.Table.
    """
    
    def __init__(self) -> None:
        self._store: dict[str, pa.Table] = {}

    @property
    def name(self) -> str:
        return "arrow_inmemory"
    
    @property
    def capabilities(self):
        raise NotImplementedError(
            "EngineCapabilities has not yet been inserted into the base workflow."
        )
 
    @property
    def backend(self) -> dict[str, pa.Table]:
        """Direct access to stored Arrow Tables."""
        return self._store
    
    def memory_usage(self, dataset: str | None = None) -> int:
        if dataset is not None:
            return self._store[dataset].nbytes
        return sum(tbl.nbytes for tbl in self._store.values())
    
    def _scan(self, dataset: str, options: ReadOptions) -> pad.Scanner:
        if dataset not in self._store:
            raise KeyError(f"Dataset '{dataset}' not found.")
 
        tbl = self._store[dataset]
 
        # Slice before converting to Scanner — avoids scanning unneeded rows.
        if options.offset or options.limit is not None:
            tbl = tbl.slice(options.offset, options.limit)
 
        ds: pad.Dataset = pad.dataset(tbl)
        scan_kwargs: dict[str, Any] = {"batch_size": options.batch_size}
 
        if options.columns:
            missing = set(options.columns) - set(tbl.schema.names)
            if missing:
                raise KeyError(f"Columns not found in '{dataset}': {missing}")
            scan_kwargs["columns"] = options.columns
 
        if options.filters:
            scan_kwargs["filter"] = _build_arrow_expression(options.filters)
 
        return ds.scanner(**scan_kwargs)

    def write(
        self,
        dataset: str,
        data: Any,
        options: WriteOptions | None = None,
    ) -> None:
        opts = options or WriteOptions()
 
        # 1. Check existence guard first — may skip or raise before any I/O.
        if not self._should_proceed_with_write(dataset, opts.if_exists):
            return
 
        # 2. Normalise input to pa.Table (in-memory engines always need the full table).
        incoming = self._collect_reader(normalize_to_reader(data))
 
        # 3. Dispatch to the appropriate merge strategy.
        if opts.mode == WriteMode.OVERWRITE:
            self._write_overwrite(dataset, incoming, opts)
 
        elif opts.mode == WriteMode.APPEND:
            self._write_append(dataset, incoming, opts)
 
        elif opts.mode == WriteMode.UPSERT:
            self._write_upsert(dataset, incoming, opts)
 
        elif opts.mode == WriteMode.REPLACE_WHERE:
            self._write_replace_where(dataset, incoming, opts)
 
        else:
            raise ValueError(f"Unknown WriteMode: {opts.mode!r}")
        
    def _write_overwrite(self, dataset: str, incoming: pa.Table, opts: WriteOptions) -> None:
        # Schema evolution only matters if we're keeping the existing schema
        # as a reference. For overwrite we still enforce STRICT if asked —
        # useful to catch accidental schema changes.
        if self.exists(dataset) and opts.schema_evolution == SchemaEvolution.STRICT:
            existing = self._store[dataset]
            if existing.schema != incoming.schema:
                raise SchemaError(
                    f"Schema mismatch on overwrite of '{dataset}'. "
                    f"Use SchemaEvolution.OVERWRITE to suppress this check."
                )
        self._store[dataset] = incoming
 
    def _write_append(self, dataset: str, incoming: pa.Table, opts: WriteOptions) -> None:
        if not self.exists(dataset):
            self._store[dataset] = incoming
            return
 
        existing = self._store[dataset]
        existing, incoming = self._apply_schema_evolution(
            existing, incoming, opts.schema_evolution
        )
        self._store[dataset] = pa.concat_tables([existing, incoming])
 
    def _write_upsert(self, dataset: str, incoming: pa.Table, opts: WriteOptions) -> None:
        """
        Insert new rows; update existing rows that share the same key_columns values.
 
        Algorithm (in-memory, O(N)):
          1. Build a set of key tuples from incoming.
          2. Drop rows from existing whose key is in that set.
          3. Concatenate the trimmed existing with the full incoming.
 
        For large datasets, consider DuckDB's native ON CONFLICT instead.
        """
        if not self.exists(dataset):
            self._store[dataset] = incoming
            return
 
        existing = self._store[dataset]
        existing, incoming = self._apply_schema_evolution(
            existing, incoming, opts.schema_evolution
        )
 
        keys = opts.key_columns
        for col in keys:
            if col not in existing.schema.names:
                raise KeyError(f"key_column '{col}' not found in existing dataset '{dataset}'.")
            if col not in incoming.schema.names:
                raise KeyError(f"key_column '{col}' not found in incoming data.")
 
        # Build a boolean mask: True for existing rows whose key is NOT in incoming.
        incoming_keys = set(
            zip(*[incoming.column(k).to_pylist() for k in keys])
        )
        existing_key_cols = zip(*[existing.column(k).to_pylist() for k in keys])
        keep_mask = pa.array(
            [tuple(row) not in incoming_keys for row in existing_key_cols]
        )
        survivors = existing.filter(keep_mask)
        self._store[dataset] = pa.concat_tables([survivors, incoming])
 
    def _write_replace_where(
        self, dataset: str, incoming: pa.Table, opts: WriteOptions
    ) -> None:
        """
        Replace only the rows matching replace_filter; leave all others intact.
 
        Algorithm:
          1. Build a mask of rows in existing that match the filter.
          2. Keep only rows that do NOT match (invert the mask).
          3. Concatenate survivors with the full incoming data.
        """
        if not self.exists(dataset):
            self._store[dataset] = incoming
            return
 
        existing = self._store[dataset]
        existing, incoming = self._apply_schema_evolution(
            existing, incoming, opts.schema_evolution
        )
 
        filter_expr = _build_arrow_expression(opts.replace_filter)
        # Rows that match the filter are the ones being replaced — drop them.
        survivors = existing.filter(pc.invert(filter_expr))
        self._store[dataset] = pa.concat_tables([survivors, incoming])

    def info(self, dataset: str) -> DatasetInfo:
        if dataset not in self._store:
            raise KeyError(f"Dataset '{dataset}' not found.")
        tbl = self._store[dataset]
        return DatasetInfo(
            name=dataset,
            num_rows=tbl.num_rows,
            num_columns=tbl.num_columns,
            column_names=tbl.schema.names,
            column_types={f.name: str(f.type) for f in tbl.schema},
            size_bytes=tbl.nbytes,
        )
 
    def list_datasets(self) -> list[str]:
        return list(self._store.keys())
    
    def exists(self, dataset):
        return dataset in self._store
    
    def drop(self, dataset):
        if not self.exists(dataset):
            raise KeyError(f"Dataset '{dataset}' not found.")
        d = self._store.pop(dataset)
        del d

    def delete_rows(self, dataset, row_filter = None):
        raise NotImplementedError()
    
    def drop_columns(self, dataset, columns):
        raise NotImplementedError()
        
