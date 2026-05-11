r"""
Provide a unified data storage engine interface.

This module defines the abstract contracts and concrete implementations
for moving data in and out of various storage backends.

**Design Principles**

1. **Apache Arrow is the interchange format.** Every engine must produce
   and consume a `pyarrow.Table`. This is the only thing the base
   class enforces for data movement — backends handle the rest themselves.
2. **Capability flags.** Callers can check `engine.capabilities` before
   calling an optional method.
3. **Explicit Lifecycle.** Engines are context managers; connect/disconnect
   are separated from construction so the same engine object can be reused.
4. **Native Execution.** `execute()` lets callers drop into whatever query
   language the backend natively speaks (SQL, pandas query strings, xarray
   selectors, etc.) without the base class trying to abstract it away.
"""

from .base import *
from .utils import *
from .impls import *
