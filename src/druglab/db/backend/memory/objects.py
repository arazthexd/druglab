"""
druglab.db.backend.memory.objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In-memory storage mixin for objects.

Currently, the only supported mixin is ``MemoryObjectMixin`` which manages objects
in RAM as a Python list. However, other mixins may be supported in the future.
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from ...indexing import RowSelection
from ..base.mixins import BaseObjectMixin

if TYPE_CHECKING:
    from druglab.db.indexing import INDEX_LIKE

__all__ = ['MemoryObjectMixin']

class MemoryObjectMixin(BaseObjectMixin):
    """
    In-memory object storage mixin utilizing a standard Python list.

    Accepts ``objects`` keyword and initializes ``self._objects`` 
    via ``initialize_storage_context``.
    """

    def initialize_storage_context(
        self, 
        objects: Optional[List[Any]] = None, 
        **kwargs: Any
    ) -> None:
        """Initialize storage context via ``objects`` kwarg.
        
        Parameters
        ----------
        objects : List[Any], optional
            List of objects to initialize storage context with.
        **kwargs
            Additional keyword arguments. These are passed in the MRO chain.
        """
        self._objects = objects or []
        super().initialize_storage_context(**kwargs)

    def get_objects(self, idx: Optional[INDEX_LIKE] = None) -> Union[Any, List[Any]]:
        """
        Retrieve one or more objects from the table.
        
        Parameters
        ----------
        idx : Optional[INDEX_LIKE], default None
            Row selector. 

        Returns
        -------
        Union[Any, List[Any]]
            Returns a single object if `idx` is an integer. Otherwise, returns 
            a list of objects.
        """
        if idx is None:
            return self._objects.copy()

        if isinstance(idx, int):
            n = len(self._objects)
            i = int(idx)
            if i >= n or i < -n:
                raise IndexError(
                    f"index {idx} is out of bounds for axis 0 with size {n}"
                )
            index = n + i if i < 0 else i
            return self._objects[index]

        sel = RowSelection.from_raw(idx, len(self._objects))
        return sel.apply_to_list(self._objects)

    def update_objects(
        self,
        objs: Union[Any, List[Any]],
        idx: Optional[INDEX_LIKE] = None,
        **kwargs
    ) -> None:
        
        if idx is None:
            self._objects = list(objs)
            return

        if isinstance(idx, int):
            n = len(self._objects)
            index = n + idx if idx < 0 else idx
            self._objects[index] = objs
            return

        sel = RowSelection.from_raw(idx, len(self._objects))
        if len(sel.positions) != len(objs):
            raise ValueError(
                "Length of objs sequence must match length of resolved index."
            )
        for i, obj in zip(sel.positions, objs):
            self._objects[i] = obj

    def _n_objects(self) -> int:
        return len(self._objects)

    def _validate_objects(self) -> None:
        pass