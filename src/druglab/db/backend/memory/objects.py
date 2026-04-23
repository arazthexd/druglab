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

    # ------------------------------------------------------------------
    # Initialization Hooks
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Object Mixin API
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Persistence Hooks
    # ------------------------------------------------------------------

    def save_storage_context(
        self,
        path: Path,
        object_writer: Optional[Callable[[List[Any], Path], None]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Persist objects to ``<path>/objects/``.
 
        Parameters
        ----------
        object_writer : Callable[[List[Any], Path], None], optional
            Bulk writer with signature ``(objects, dir_path) -> None``.
            When provided, the mixin delegates all writing to it - enabling
            domain-specific formats (e.g. RDKit ``SDWriter``).  When ``None``,
            falls back to streaming pickle (``stream_v2`` format, one
            ``pickle.dump`` per object to prevent list-level OOM).
        """
        obj_dir = path / "objects"
        obj_dir.mkdir(exist_ok=True)
 
        if object_writer is not None:
            object_writer(self._objects, obj_dir)
        else:
            # Default: stream each object via pickle without per-object
            # serialisation (serialized=False).  Objects that are not
            # picklable must supply a custom object_writer.
            with open(obj_dir / "objects.pkl", "wb") as f:
                pickle.dump(
                    {
                        "format": "stream_v2",
                        "count": len(self._objects),
                        "serialized": False,
                    },
                    f,
                )
                for obj in self._objects:
                    pickle.dump(obj, f)
 
        super().save_storage_context(path, object_writer=object_writer, **kwargs)
 
    @classmethod
    def load_storage_context(
        cls,
        path: Path,
        object_reader: Optional[Callable[[Path], List[Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Read objects from ``<path>/objects/`` and add them to the accumulated
        kwargs under the key ``"objects"``.
 
        Parameters
        ----------
        object_reader : Callable[[Path], List[Any]], optional
            Bulk reader with signature ``(dir_path) -> List[Any]``.
            When provided, the mixin delegates entirely to it (enabling
            domain-specific deserialisers, e.g. an RDKit SDF reader).
            When ``None``, reads the default ``stream_v2`` pickle bundle.
 
            **Backward-compat note:** The old ``deserializer`` parameter (a
            per-object ``bytes -> obj`` callable) is no longer supported at
            the backend level.  ``BaseTable._make_object_reader()`` wraps
            ``_deserialize_object`` into a bulk reader for you automatically.
        """
        objects: List[Any] = []
 
        if object_reader is not None:
            objects = object_reader(path / "objects")
        else:
            obj_path = path / "objects" / "objects.pkl"
            if obj_path.exists():
                with open(obj_path, "rb") as f:
                    raw_payload = pickle.load(f)
 
                    if isinstance(raw_payload, dict) and raw_payload.get(
                        "format"
                    ) in {"stream_v1", "stream_v2"}:
                        count = int(raw_payload["count"])
                        is_serialized = raw_payload.get("serialized", False)
                        raw_list = [pickle.load(f) for _ in range(count)]
                    else:
                        # Legacy: entire list was pickled in one call.
                        raw_list = raw_payload
                        is_serialized = False
 
                    # Without a reader, we cannot deserialise serialized bytes;
                    # return as raw bytes.  The caller (BaseTable.load) always
                    # supplies an object_reader so this branch is only hit
                    # when EagerMemoryBackend.load is called directly with no
                    # reader (e.g. in unit tests storing plain Python objects).
                    objects = raw_list
 
        result = super().load_storage_context(
            path, object_reader=object_reader, **kwargs
        )
        result["objects"] = objects
        return result