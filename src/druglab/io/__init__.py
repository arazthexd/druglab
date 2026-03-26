"""
druglab.io
==========
File reading and writing utilities for molecules, reactions, and related
cheminformatics data.

Public API
----------
Readers
~~~~~~~
- :class:`BatchReader`  - memory-safe streaming reader over multiple files
- :class:`EagerReader`  - load all records from one or more files into memory

Writers
~~~~~~~
- :class:`SDFWriter`
- :class:`SMILESWriter`
- :class:`CSVWriter`
- :class:`RXNWriter`
- :class:`MOLWriter`

Format handlers
~~~~~~~~~~~~~~~
- :func:`read_file`   - single-file convenience wrapper (returns a list)
- :func:`write_file`  - single-file convenience wrapper
- :data:`SUPPORTED_FORMATS` - frozenset of recognised extensions

Exceptions
~~~~~~~~~~
- :exc:`UnsupportedFormatError`
- :exc:`DrugLabIOError`
"""

from druglab.io.exceptions import UnsupportedFormatError, DrugLabIOError
from druglab.io.formats import (
    SUPPORTED_FORMATS,
    detect_format,
    get_reader_cls,
    get_writer_cls,
)
from druglab.io.readers import BatchReader, EagerReader, read_file
from druglab.io.writers import (
    SDFWriter,
    SMILESWriter,
    CSVWriter,
    RXNWriter,
    MOLWriter,
    write_file,
)

__all__ = [
    # exceptions
    "UnsupportedFormatError",
    "DrugLabIOError",
    # format helpers
    "SUPPORTED_FORMATS",
    "detect_format",
    "get_reader_cls",
    "get_writer_cls",
    # readers
    "BatchReader",
    "EagerReader",
    "read_file",
    # writers
    "SDFWriter",
    "SMILESWriter",
    "CSVWriter",
    "RXNWriter",
    "MOLWriter",
    "write_file",
]