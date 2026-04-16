"""
Readers for druglab.io.

Hierarchy
---------
::

    BaseFormatReader          – abstract: yields records from a single file
        SDFFormatReader
        SMILESFormatReader
        CSVFormatReader
        RXNFormatReader
        MOLFormatReader

    BaseReader                – abstract: drives one or more files
        BatchReader           – streaming / memory-safe (yields batches)
        EagerReader           – loads everything into memory at once

Convenience
-----------
    read_file(path, ...)      – one-shot helper; returns list[MoleculeRecord]
"""

from __future__ import annotations

import csv
import logging
import inspect
import os
from abc import ABC, abstractmethod
from typing import (
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    Any
)

from druglab.io._record import MoleculeRecord, ReactionRecord
from druglab.io.exceptions import DrugLabIOError, RecordParseError, UnsupportedFormatError

logger = logging.getLogger(__name__)

# Type alias for the union record type
AnyRecord = Union[MoleculeRecord, ReactionRecord]
OnErrorPolicy = Literal["raise", "skip", "warn"]


# ---------------------------------------------------------------------------
# Abstract base for per-format readers
# ---------------------------------------------------------------------------


class BaseFormatReader(ABC):
    """Abstract base class for single-file format readers.

    Subclasses must implement :meth:`iter_records`, which yields individual
    :class:`~druglab.io._record.MoleculeRecord` or
    :class:`~druglab.io._record.ReactionRecord` objects lazily.

    Parameters
    ----------
    on_error:
        What to do when a single record cannot be parsed.
        ``"raise"`` – propagate the exception (default).
        ``"skip"``  – silently skip the bad record.
        ``"warn"``  – log a warning and skip.
    """

    def __init__(self, on_error: OnErrorPolicy = "raise") -> None:
        self.on_error = on_error

    @abstractmethod
    def iter_records(self, path: str) -> Iterator[AnyRecord]:
        """Yield records one-by-one from *path*."""
        ...

    def read_all(self, path: str) -> List[AnyRecord]:
        """Convenience: collect all records from *path* into a list."""
        return list(self.iter_records(path))

    # ------------------------------------------------------------------
    # Internal error handling
    # ------------------------------------------------------------------

    def _handle_error(
        self, source: str, index: int, exc: Exception
    ) -> Optional[AnyRecord]:
        """Apply ``on_error`` policy; return *None* to signal skip."""
        err = RecordParseError(source, index, str(exc))
        if self.on_error == "raise":
            raise err from exc
        if self.on_error == "warn":
            logger.warning(str(err))
        return None  # skip


# ---------------------------------------------------------------------------
# SDF reader
# ---------------------------------------------------------------------------


class SDFFormatReader(BaseFormatReader):
    """Read SD files (.sdf / .sd), yielding one :class:`MoleculeRecord` per
    molecule.

    Uses RDKit's ``ForwardSDMolSupplier`` so files can be arbitrarily large
    without loading everything into memory.

    Parameters
    ----------
    sanitize:
        Whether to sanitize each molecule (default *True*).
    remove_Hs:
        Whether to remove explicit hydrogen atoms (default *True*).
    on_error:
        Error policy (see :class:`BaseFormatReader`).
    """

    def __init__(
        self,
        sanitize: bool = True,
        remove_Hs: bool = True,
        on_error: OnErrorPolicy = "raise",
    ) -> None:
        super().__init__(on_error=on_error)
        self.sanitize = sanitize
        self.remove_Hs = remove_Hs

    def iter_records(self, path: str) -> Iterator[MoleculeRecord]:
        try:
            from rdkit.Chem import ForwardSDMolSupplier  # type: ignore
            from rdkit import Chem
        except ImportError as e:
            raise DrugLabIOError(
                "RDKit is required for SDF reading. Install it with: "
                "conda install -c conda-forge rdkit"
            ) from e

        with open(path, "rb") as fh:
            supplier = ForwardSDMolSupplier(
                fh, sanitize=self.sanitize, removeHs=self.remove_Hs
            )
            for idx, mol in enumerate(supplier):
                if mol is None:
                    rec = self._handle_error(
                        path, idx, ValueError("RDKit returned None molecule")
                    )
                    if rec is not None:
                        yield rec
                    continue
                
                mol: Chem.Mol
                props: Dict[str, object] = mol.GetPropsAsDict()
                yield MoleculeRecord(
                    mol=mol,
                    name=mol.GetProp("_Name") if mol.HasProp("_Name") else "",
                    properties=props,
                    source=path,
                    index=idx,
                )


# ---------------------------------------------------------------------------
# SMILES reader
# ---------------------------------------------------------------------------


class SMILESFormatReader(BaseFormatReader):
    """Read SMILES files (.smi / .smiles), one SMILES per line.

    Expected line formats (tab or space separated)::

        <SMILES>
        <SMILES> <name>
        <SMILES> <name> [key=value ...]

    Parameters
    ----------
    delimiter:
        Column separator; defaults to whitespace splitting.
    smiles_col:
        Column index (0-based) of the SMILES string (default 0).
    name_col:
        Column index of the name field; *None* → no name (default 1).
    sanitize:
        Whether to sanitize (default *True*).
    on_error:
        Error policy.
    """

    def __init__(
        self,
        delimiter: Optional[str] = None,
        smiles_col: Union[int, str] = 0,
        name_col: Optional[Union[int, str]] = 1,
        sanitize: bool = True,
        on_error: OnErrorPolicy = "raise",
    ) -> None:
        super().__init__(on_error=on_error)
        self.delimiter = delimiter
        self.smiles_col = self._normalize_column_selector(smiles_col, role="smiles")
        self.name_col = self._normalize_column_selector(name_col, role="name")
        self.sanitize = sanitize

    @staticmethod
    def _normalize_column_selector(
        selector: Optional[Union[int, str]],
        role: str,
    ) -> Optional[int]:
        if selector is None:
            return None
        if isinstance(selector, int):
            return selector
        normalized = selector.strip().lower()
        if normalized.isdigit():
            return int(normalized)
        if role == "smiles" and normalized in {"smiles", "smi"}:
            return 0
        if role == "name" and normalized in {"name", "id", "title"}:
            return 1
        raise ValueError(
            f"Unsupported {role}_col selector for SMILESFormatReader: {selector!r}. "
            "Use an integer index (or supported aliases)."
        )

    def iter_records(self, path: str) -> Iterator[MoleculeRecord]:
        try:
            from rdkit.Chem import MolFromSmiles  # type: ignore
        except ImportError as e:
            raise DrugLabIOError("RDKit is required for SMILES reading.") from e

        with open(path, "r", encoding="utf-8") as fh:
            for idx, raw_line in enumerate(fh):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = (
                    line.split(self.delimiter)
                    if self.delimiter
                    else line.split()
                )
                if self.smiles_col >= len(parts):
                    self._handle_error(
                        path, idx, IndexError("SMILES column out of range")
                    )
                    continue
                smiles = parts[self.smiles_col]
                name = (
                    parts[self.name_col]
                    if (self.name_col is not None and self.name_col < len(parts))
                    else ""
                )
                try:
                    mol = MolFromSmiles(smiles, sanitize=self.sanitize)
                    if mol is None:
                        raise ValueError(f"Could not parse SMILES: {smiles!r}")
                except Exception as exc:
                    result = self._handle_error(path, idx, exc)
                    if result is not None:
                        yield result
                    continue

                yield MoleculeRecord(
                    mol=mol,
                    name=name,
                    properties={"SMILES": smiles},
                    source=path,
                    index=idx,
                )


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------


class CSVFormatReader(BaseFormatReader):
    """Read CSV / TSV files containing SMILES columns.

    The reader expects at least one column that holds SMILES strings.  All
    other columns become molecule properties.

    Parameters
    ----------
    smiles_col:
        Name of the SMILES column (default ``"SMILES"``).
    name_col:
        Name of the molecule-name column, or *None* (default ``None``).
    delimiter:
        CSV delimiter (default ``","``; use ``"\\t"`` for TSV).
    sanitize:
        Whether to sanitize parsed molecules (default *True*).
    on_error:
        Error policy.
    """

    def __init__(
        self,
        smiles_col: Union[int, str] = "SMILES",
        name_col: Optional[Union[int, str]] = None,
        delimiter: str = ",",
        sanitize: bool = True,
        on_error: OnErrorPolicy = "raise",
    ) -> None:
        super().__init__(on_error=on_error)
        self.smiles_col = smiles_col
        self.name_col = name_col
        self.delimiter = delimiter
        self.sanitize = sanitize

    @staticmethod
    def _resolve_column_name(
        fieldnames: List[str],
        selector: Optional[Union[int, str]],
        role: str,
        path: str,
    ) -> Optional[str]:
        if selector is None:
            return None
        if isinstance(selector, int):
            if selector < 0 or selector >= len(fieldnames):
                raise DrugLabIOError(
                    f"{role} column index {selector} is out of range for {path}. "
                    f"Available columns: {fieldnames}"
                )
            return fieldnames[selector]
        if selector not in fieldnames:
            raise DrugLabIOError(
                f"{role} column '{selector}' not found in {path}. "
                f"Available columns: {fieldnames}"
            )
        return selector

    def iter_records(self, path: str) -> Iterator[MoleculeRecord]:
        try:
            from rdkit.Chem import MolFromSmiles  # type: ignore
        except ImportError as e:
            raise DrugLabIOError("RDKit is required for CSV/SMILES reading.") from e

        # Auto-detect TSV delimiter from extension
        delimiter = self.delimiter
        if path.lower().endswith(".tsv") and delimiter == ",":
            delimiter = "\t"

        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            if reader.fieldnames is None:
                return

            fieldnames = list(reader.fieldnames)
            smiles_col_name = self._resolve_column_name(
                fieldnames, self.smiles_col, role="SMILES", path=path
            )
            assert smiles_col_name is not None
            name_col_name = self._resolve_column_name(
                fieldnames, self.name_col, role="name", path=path
            )

            for idx, row in enumerate(reader):
                smiles = row.get(smiles_col_name, "").strip()
                if not smiles:
                    self._handle_error(
                        path, idx, ValueError("Empty SMILES field")
                    )
                    continue

                name = (
                    row.get(name_col_name, "")
                    if name_col_name
                    else ""
                )
                props = {k: v for k, v in row.items() if k != smiles_col_name}

                try:
                    mol = MolFromSmiles(smiles, sanitize=self.sanitize)
                    if mol is None:
                        raise ValueError(f"Could not parse SMILES: {smiles!r}")
                except Exception as exc:
                    result = self._handle_error(path, idx, exc)
                    if result is not None:
                        yield result
                    continue

                yield MoleculeRecord(
                    mol=mol,
                    name=name,
                    properties=props,
                    source=path,
                    index=idx,
                )


# ---------------------------------------------------------------------------
# RXN reader
# ---------------------------------------------------------------------------


class RXNFormatReader(BaseFormatReader):
    """Read MDL RXN / RD files, yielding one :class:`ReactionRecord` per
    reaction.

    Parameters
    ----------
    sanitize:
        Whether to sanitize component molecules (default *True*).
    on_error:
        Error policy.
    """

    def __init__(
        self,
        sanitize: bool = True,
        on_error: OnErrorPolicy = "raise",
    ) -> None:
        super().__init__(on_error=on_error)
        self.sanitize = sanitize

    def iter_records(self, path: str) -> Iterator[ReactionRecord]:
        try:
            from rdkit.Chem import AllChem  # type: ignore
        except ImportError as e:
            raise DrugLabIOError("RDKit is required for RXN reading.") from e

        with open(path, "r", encoding="utf-8") as fh:
            rxn_block = fh.read()

        try:
            rxn = AllChem.ReactionFromRxnBlock(
                rxn_block, sanitize=self.sanitize
            )
            if rxn is None:
                raise ValueError("ReactionFromRxnBlock returned None.")
        except Exception as exc:
            result = self._handle_error(path, 0, exc)
            if result is not None:
                yield result  # type: ignore[misc]
            return

        yield ReactionRecord(
            rxn=rxn,
            name=os.path.splitext(os.path.basename(path))[0],
            properties={},
            source=path,
            index=0,
        )


# ---------------------------------------------------------------------------
# MOL reader
# ---------------------------------------------------------------------------


class MOLFormatReader(BaseFormatReader):
    """Read a single MDL MOL file, yielding one :class:`MoleculeRecord`.

    Parameters
    ----------
    sanitize:
        Whether to sanitize (default *True*).
    remove_Hs:
        Whether to remove explicit Hs (default *True*).
    on_error:
        Error policy.
    """

    def __init__(
        self,
        sanitize: bool = True,
        remove_Hs: bool = True,
        on_error: OnErrorPolicy = "raise",
    ) -> None:
        super().__init__(on_error=on_error)
        self.sanitize = sanitize
        self.remove_Hs = remove_Hs

    def iter_records(self, path: str) -> Iterator[MoleculeRecord]:
        try:
            from rdkit.Chem import MolFromMolFile  # type: ignore
        except ImportError as e:
            raise DrugLabIOError("RDKit is required for MOL reading.") from e

        try:
            mol = MolFromMolFile(
                path, sanitize=self.sanitize, removeHs=self.remove_Hs
            )
            if mol is None:
                raise ValueError("MolFromMolFile returned None.")
        except Exception as exc:
            result = self._handle_error(path, 0, exc)
            if result is not None:
                yield result
            return

        yield MoleculeRecord(
            mol=mol,
            name=mol.GetProp("_Name") if mol.HasProp("_Name") else "",
            properties=mol.GetPropsAsDict(),
            source=path,
            index=0,
        )


# ---------------------------------------------------------------------------
# High-level reader classes: BatchReader & EagerReader
# ---------------------------------------------------------------------------


class BaseReader(ABC):
    """Abstract base for multi-file readers.

    Subclasses control *when* data is materialised (lazily vs. eagerly).

    Parameters
    ----------
    paths:
        One or more file paths to read.  Mixed formats are supported as long
        as each file has a recognised extension.
    on_error:
        Error policy forwarded to every format reader.
    format_kwargs:
        Optional per-format overrides, e.g.
        ``{"csv": {"smiles_col": "SMILES"}, "smi": {"smiles_col": 0}}``.
    reader_kwargs:
        Shared keyword arguments forwarded to compatible format readers.
    """

    def __init__(
        self,
        paths: Union[str, Sequence[str]],
        on_error: OnErrorPolicy = "raise",
        format_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        **reader_kwargs,
    ) -> None:
        if isinstance(paths, str):
            paths = [paths]
        self.paths: List[str] = list(paths)
        self.on_error = on_error
        self.reader_kwargs = reader_kwargs
        self.format_kwargs = {k.lower(): dict(v) for k, v in (format_kwargs or {}).items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_format_reader(self, path: str) -> BaseFormatReader:
        from druglab.io.formats import detect_format, get_reader_cls

        fmt = detect_format(path)
        cls = get_reader_cls(fmt)
        kwargs = dict(self.reader_kwargs)
        kwargs.update(self.format_kwargs.get(fmt, {}))
        allowed = set(inspect.signature(cls.__init__).parameters) - {"self", "on_error"}
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return cls(on_error=self.on_error, **filtered)

    def _iter_all_records(self) -> Generator[AnyRecord, None, None]:
        """Yield every record from every registered path, in order."""
        for path in self.paths:
            reader = self._build_format_reader(path)
            yield from reader.iter_records(path)

    @abstractmethod
    def __iter__(self):
        ...


class BatchReader(BaseReader):
    """Memory-safe streaming reader over one or more files.

    Yields *batches* of records rather than individual objects, allowing
    downstream code to process arbitrarily large datasets without exhausting
    RAM.

    Parameters
    ----------
    paths:
        One or more file paths to read.
    batch_size:
        Number of records per batch (default 1 000).
    on_error:
        Error policy (``"raise"``, ``"skip"``, or ``"warn"``).
    format_kwargs:
        Optional per-format reader kwargs for mixed-format workflows.
    **reader_kwargs:
        Shared reader kwargs applied when accepted by a format reader.

    Examples
    --------
    >>> reader = BatchReader(["big.sdf", "extra.sdf"], batch_size=500)
    >>> for batch in reader:
    ...     process(batch)           # batch is List[MoleculeRecord]

    You can also iterate record-by-record:

    >>> for record in reader.iter_records():
    ...     process(record)

    Or collect into a flat list (use with small files only!):

    >>> records = reader.collect()
    """

    def __init__(
        self,
        paths: Union[str, Sequence[str]],
        batch_size: int = 1_000,
        on_error: OnErrorPolicy = "raise",
        format_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        **reader_kwargs,
    ) -> None:
        super().__init__(
            paths,
            on_error=on_error,
            format_kwargs=format_kwargs,
            **reader_kwargs,
        )
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[List[AnyRecord]]:
        """Iterate over batches of records."""
        batch: List[AnyRecord] = []
        for record in self._iter_all_records():
            batch.append(record)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def iter_records(self) -> Iterator[AnyRecord]:
        """Iterate over individual records (unbatched)."""
        yield from self._iter_all_records()

    def collect(self) -> List[AnyRecord]:
        """Materialise **all** records into a list.

        .. warning::
            Only safe for datasets that fit comfortably in memory.  For large
            datasets, iterate over batches with :meth:`__iter__` instead.
        """
        return list(self._iter_all_records())

    def n_batches_estimate(self, total: int) -> int:
        """Estimate the number of batches given a *total* record count."""
        return max(1, -(-total // self.batch_size))  # ceiling division

    def __repr__(self) -> str:
        return (
            f"BatchReader(paths={self.paths!r}, batch_size={self.batch_size}, "
            f"on_error={self.on_error!r})"
        )


class EagerReader(BaseReader):
    """Eager (total-load) reader: loads all records into memory immediately.

    Unlike :class:`BatchReader`, this class materialises everything at
    construction time (or when :meth:`read` is called explicitly).

    Parameters
    ----------
    paths:
        One or more file paths to read.
    on_error:
        Error policy.
    format_kwargs:
        Optional per-format reader kwargs for mixed-format workflows.
    **reader_kwargs:
        Shared reader kwargs applied when accepted by a format reader.

    Examples
    --------
    >>> er = EagerReader(["compounds.sdf"])
    >>> records = er.read()          # List[MoleculeRecord]
    >>> for rec in er:               # also iterable
    ...     print(rec.name)
    """

    def __init__(
        self,
        paths: Union[str, Sequence[str]],
        on_error: OnErrorPolicy = "raise",
        format_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        **reader_kwargs,
    ) -> None:
        super().__init__(
            paths,
            on_error=on_error,
            format_kwargs=format_kwargs,
            **reader_kwargs,
        )
        self._records: Optional[List[AnyRecord]] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def read(self) -> List[AnyRecord]:
        """Load and return all records.

        Results are cached so subsequent calls are instantaneous.
        """
        if self._records is None:
            self._records = list(self._iter_all_records())
        return self._records

    def __iter__(self) -> Iterator[AnyRecord]:
        yield from self.read()

    def __len__(self) -> int:
        return len(self.read())

    def __getitem__(self, idx: int) -> AnyRecord:
        return self.read()[idx]

    def invalidate_cache(self) -> None:
        """Clear the in-memory cache so :meth:`read` re-reads from disk."""
        self._records = None

    def __repr__(self) -> str:
        n = len(self._records) if self._records is not None else "not loaded"
        return (
            f"EagerReader(paths={self.paths!r}, records={n}, "
            f"on_error={self.on_error!r})"
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def read_file(
    path: str,
    on_error: OnErrorPolicy = "raise",
    **reader_kwargs,
) -> List[AnyRecord]:
    """Read a single file and return all records as a list.

    This is a thin wrapper around :class:`EagerReader` for one-off use.

    Parameters
    ----------
    path:
        Path to the file to read.
    on_error:
        Error policy (``"raise"``, ``"skip"``, ``"warn"``).
    **reader_kwargs:
        Extra keyword arguments forwarded to the format-specific reader (e.g.
        ``sanitize=False``, ``smiles_col="canonical_smiles"``).

    Returns
    -------
    list
        List of :class:`~druglab.io._record.MoleculeRecord` or
        :class:`~druglab.io._record.ReactionRecord` objects.

    Examples
    --------
    >>> from druglab.io import read_file
    >>> records = read_file("compounds.sdf")
    >>> records = read_file("data.csv", smiles_col="Smiles", on_error="warn")
    """
    reader = EagerReader([path], on_error=on_error, **reader_kwargs)
    return reader.read()