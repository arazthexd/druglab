r"""
Provide a unified data storage engine interface.

This module defines the abstract contracts and concrete implementations
for moving data in and out of various storage backends.

**Design Principles**

1. **Apache Arrow is the interchange format.** Every engine must produce
   and consume a `pyarrow.Table`. This is the only thing the base
   class enforces for data movement — backends handle the rest themselves.
2. **Explicit Lifecycle.** Engines are context managers; connect/disconnect
   are separated from construction so the same engine object can be reused.

TODO: Use `EngineCapabilities` from `utils` to manage optional methods.
"""

from .base import *
from .utils import *
from .impls import *
