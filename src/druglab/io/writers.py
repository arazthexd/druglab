"""
Writers for druglab.io.

Hierarchy
---------
::

    BaseFormatWriter          – abstract context-manager writer
        SDFWriter
        SMILESWriter
        CSVWriter
        RXNWriter
        MOLWriter

Convenience
-----------
    write_file(records, path, ...)  – one-shot helper

All writers implement the context-manager protocol and expose a
:meth:`write` / :meth:`write_many` interface so they can be used both
standalone and from within pipelines::

    with SDFWriter("output.sdf") as w:
        for record in records:
            w.write(record)

    # or one-shot
    write_file(records, "output.sdf")
"""

from __future__ import annotations

import csv
import logging
import os
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Sequence, Union

from druglab.io._record import AnyRecord, MoleculeRecord, ReactionRecord
from druglab.io.exceptions import DrugLabIOError, UnsupportedFormatError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseFormatWriter(ABC):
    """Abstract base class for all druglab.io writers.

    Subclasses must implement :meth:`_write_record`.  The base class handles
    opening/closing the output file and the context-manager protocol.

    Parameters
    ----------
    path:
        Destination file path.  Parent directories are created automatically.
    overwrite:
        If *False* (default) and *path* already exists, a
        :exc:`~druglab.io.exceptions.DrugLabIOError` is raised.
    """

    def __init__(self, path: str, overwrite: bool = False) -> None:
        self.path = path
        self.overwrite = overwrite
        self._handle = None  # opened in __enter__

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "BaseFormatWriter":
        if not self.overwrite and os.path.exists(self.path):
            raise DrugLabIOError(
                f"Output file already exists: {self.path!r}. "
                "Pass overwrite=True to replace it."
            )
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._handle = self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._close()
        return False  # do not suppress exceptions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, record: AnyRecord) -> None:
        """Write a single record to the output file."""
        self._write_record(record)

    def write_many(self, records: Iterable[AnyRecord]) -> int:
        """Write an iterable of records; return the number written."""
        n = 0
        for rec in records:
            self.write(rec)
            n += 1
        return n

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _open(self):
        """Open and return the underlying file handle."""
        return open(self.path, "w", encoding="utf-8")

    def _close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    @abstractmethod
    def _write_record(self, record: AnyRecord) -> None:
        """Write a single *record* to *self._handle*."""
        ...


# ---------------------------------------------------------------------------
# SDF writer
# ---------------------------------------------------------------------------


class SDFWriter(BaseFormatWriter):
    """Write molecules to an SD file.

    Parameters
    ----------
    path:
        Destination ``.sdf`` path.
    include_properties:
        If *True* (default), write all molecule properties as SDF data fields.
    overwrite:
        Replace existing file if *True*.
    """

    def __init__(
        self,
        path: str,
        include_properties: bool = True,
        overwrite: bool = False,
    ) -> None:
        super().__init__(path, overwrite=overwrite)
        self.include_properties = include_properties
        self._writer = None

    def _open(self):
        try:
            from rdkit.Chem import SDWriter  # type: ignore
        except ImportError as e:
            raise DrugLabIOError("RDKit is required for SDF writing.") from e

        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._writer = SDWriter(self.path)
        return self._writer

    def _close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _write_record(self, record: AnyRecord) -> None:
        if not isinstance(record, MoleculeRecord):
            raise TypeError(
                f"SDFWriter expects MoleculeRecord, got {type(record).__name__}"
            )
        if record.mol is None:
            logger.warning("Skipping invalid MoleculeRecord (mol is None).")
            return

        mol = record.mol
        if record.name:
            mol.SetProp("_Name", record.name)
        if self.include_properties:
            for k, v in record.properties.items():
                if isinstance(v, float):
                    mol.SetDoubleProp(k, v)
                elif isinstance(v, int):
                    mol.SetIntProp(k, v)
                else:
                    mol.SetProp(k, str(v))
        self._writer.write(mol)


# ---------------------------------------------------------------------------
# SMILES writer
# ---------------------------------------------------------------------------


class SMILESWriter(BaseFormatWriter):
    """Write molecules to a whitespace-delimited SMILES file.

    Parameters
    ----------
    path:
        Destination ``.smi`` / ``.smiles`` path.
    delimiter:
        Column separator (default ``" "``).
    write_name:
        If *True* (default), write the molecule name as the second column.
    canonical:
        If *True* (default), output canonical SMILES via RDKit.
    overwrite:
        Replace existing file if *True*.
    """

    def __init__(
        self,
        path: str,
        delimiter: str = " ",
        write_name: bool = True,
        canonical: bool = True,
        overwrite: bool = False,
    ) -> None:
        super().__init__(path, overwrite=overwrite)
        self.delimiter = delimiter
        self.write_name = write_name
        self.canonical = canonical

    def _write_record(self, record: AnyRecord) -> None:
        if not isinstance(record, MoleculeRecord):
            raise TypeError(
                f"SMILESWriter expects MoleculeRecord, got {type(record).__name__}"
            )
        if record.mol is None:
            logger.warning("Skipping invalid MoleculeRecord (mol is None).")
            return

        try:
            from rdkit.Chem import MolToSmiles  # type: ignore

            smiles = MolToSmiles(record.mol) if self.canonical else (
                record.properties.get("SMILES", MolToSmiles(record.mol))
            )
        except ImportError as e:
            raise DrugLabIOError("RDKit is required for SMILES writing.") from e

        parts = [smiles]
        if self.write_name:
            parts.append(record.name or "")
        self._handle.write(self.delimiter.join(parts) + "\n")


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------


class CSVWriter(BaseFormatWriter):
    """Write molecules to a CSV file with a SMILES column.

    Parameters
    ----------
    path:
        Destination ``.csv`` path.
    smiles_col:
        Header for the SMILES column (default ``"SMILES"``).
    name_col:
        Header for the name column, or *None* to omit (default ``"Name"``).
    extra_cols:
        Ordered list of property keys to include as additional columns.
        If *None*, all properties are written (in insertion order).
    delimiter:
        CSV delimiter (default ``","``).
    overwrite:
        Replace existing file if *True*.
    """

    def __init__(
        self,
        path: str,
        smiles_col: str = "SMILES",
        name_col: Optional[str] = "Name",
        extra_cols: Optional[List[str]] = None,
        delimiter: str = ",",
        overwrite: bool = False,
    ) -> None:
        super().__init__(path, overwrite=overwrite)
        self.smiles_col = smiles_col
        self.name_col = name_col
        self.extra_cols = extra_cols
        self.delimiter = delimiter
        self._csv_writer = None
        self._header_written = False

    def _open(self):
        handle = open(self.path, "w", encoding="utf-8", newline="")
        return handle

    def _write_record(self, record: AnyRecord) -> None:
        if not isinstance(record, MoleculeRecord):
            raise TypeError(
                f"CSVWriter expects MoleculeRecord, got {type(record).__name__}"
            )
        if record.mol is None:
            logger.warning("Skipping invalid MoleculeRecord (mol is None).")
            return

        try:
            from rdkit.Chem import MolToSmiles  # type: ignore

            smiles = MolToSmiles(record.mol)
        except ImportError as e:
            raise DrugLabIOError("RDKit is required for CSV writing.") from e

        prop_keys = self.extra_cols if self.extra_cols is not None else list(record.properties.keys())

        if not self._header_written:
            header = [self.smiles_col]
            if self.name_col:
                header.append(self.name_col)
            header.extend(prop_keys)
            self._csv_writer = csv.DictWriter(
                self._handle,
                fieldnames=header,
                delimiter=self.delimiter,
                extrasaction="ignore",
            )
            self._csv_writer.writeheader()
            self._header_written = True

        row = {self.smiles_col: smiles}
        if self.name_col:
            row[self.name_col] = record.name
        for k in prop_keys:
            row[k] = record.properties.get(k, "")
        self._csv_writer.writerow(row)


# ---------------------------------------------------------------------------
# RXN writer
# ---------------------------------------------------------------------------


class RXNWriter(BaseFormatWriter):
    """Write reactions to MDL RXN format (one file per reaction).

    Since the RXN format is a single-reaction format, each call to
    :meth:`write` will create a separate file named
    ``<base>_<index>.rxn`` when writing multiple reactions.

    Parameters
    ----------
    path:
        Base path.  For a single reaction this is used verbatim; for multiple
        reactions the index is injected before the extension.
    overwrite:
        Replace existing files if *True*.
    """

    def __init__(self, path: str, overwrite: bool = False) -> None:
        super().__init__(path, overwrite=overwrite)
        self._index = 0

    def _open(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        return None  # We open per-record

    def _close(self) -> None:
        pass  # Nothing to close globally

    def _write_record(self, record: AnyRecord) -> None:
        if not isinstance(record, ReactionRecord):
            raise TypeError(
                f"RXNWriter expects ReactionRecord, got {type(record).__name__}"
            )
        if record.rxn is None:
            logger.warning("Skipping invalid ReactionRecord (rxn is None).")
            return

        try:
            from rdkit.Chem import AllChem  # type: ignore
        except ImportError as e:
            raise DrugLabIOError("RDKit is required for RXN writing.") from e

        # Determine output path
        if self._index == 0:
            out_path = self.path
        else:
            root, ext = os.path.splitext(self.path)
            out_path = f"{root}_{self._index}{ext}"

        if not self.overwrite and os.path.exists(out_path):
            raise DrugLabIOError(
                f"Output file already exists: {out_path!r}. "
                "Pass overwrite=True to replace it."
            )

        rxn_block = AllChem.ReactionToRxnBlock(record.rxn)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(rxn_block)

        self._index += 1


# ---------------------------------------------------------------------------
# MOL writer
# ---------------------------------------------------------------------------


class MOLWriter(BaseFormatWriter):
    """Write a single molecule to MDL MOL format.

    Writes only the first valid record; subsequent calls raise a warning.

    Parameters
    ----------
    path:
        Destination ``.mol`` path.
    overwrite:
        Replace existing file if *True*.
    """

    def __init__(self, path: str, overwrite: bool = False) -> None:
        super().__init__(path, overwrite=overwrite)
        self._written = False

    def _write_record(self, record: AnyRecord) -> None:
        if not isinstance(record, MoleculeRecord):
            raise TypeError(
                f"MOLWriter expects MoleculeRecord, got {type(record).__name__}"
            )
        if record.mol is None:
            logger.warning("Skipping invalid MoleculeRecord (mol is None).")
            return

        if self._written:
            logger.warning(
                "MOLWriter: attempted to write more than one molecule to a .mol "
                "file. Only the first molecule was written."
            )
            return

        try:
            from rdkit.Chem import MolToMolBlock  # type: ignore

            block = MolToMolBlock(record.mol)
        except ImportError as e:
            raise DrugLabIOError("RDKit is required for MOL writing.") from e

        self._handle.write(block)
        self._written = True


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def write_file(
    records: Union[AnyRecord, Iterable[AnyRecord]],
    path: str,
    overwrite: bool = False,
    **writer_kwargs,
) -> int:
    """Write *records* to *path*, auto-detecting the format from the extension.

    Parameters
    ----------
    records:
        A single record or an iterable of records.
    path:
        Output file path.  The extension determines the writer class used.
    overwrite:
        Replace the file if it already exists.
    **writer_kwargs:
        Extra keyword arguments forwarded to the writer constructor.

    Returns
    -------
    int
        Number of records written.

    Examples
    --------
    >>> from druglab.io import write_file
    >>> write_file(mol_records, "output.sdf")
    >>> write_file(mol_records, "output.csv", smiles_col="canonical_smiles")
    """
    from druglab.io.formats import detect_format, get_writer_cls

    fmt = detect_format(path)
    writer_cls = get_writer_cls(fmt)

    if isinstance(records, (MoleculeRecord, ReactionRecord)):
        records = [records]

    with writer_cls(path, overwrite=overwrite, **writer_kwargs) as w:  # type: ignore[call-arg]
        return w.write_many(records)