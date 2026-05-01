"""In-memory object store — pickle-free."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

import numpy as np

from ...indexing import RowSelection
from ..base.stores import BaseObjectStore

if TYPE_CHECKING:
    from druglab.db.indexing import INDEX_LIKE


__all__ = ["MemoryObjectStore"]


class MemoryObjectStore(BaseObjectStore):
    def __init__(self, objects: Optional[List[Any]] = None) -> None:
        self._objects = list(objects) if objects is not None else []

    def get_objects(self, idx: Optional["INDEX_LIKE"] = None) -> Union[Any, List[Any]]:
        if idx is None:
            return self._objects.copy()

        if isinstance(idx, (int, np.integer)):
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

    def update_objects(self, objs: Union[Any, List[Any]], idx: Optional["INDEX_LIKE"] = None) -> None:
        if idx is None:
            self._objects = list(objs)
            return

        if isinstance(idx, (int, np.integer)):
            n = len(self._objects)
            index = n + idx if idx < 0 else idx
            self._objects[index] = objs
            return

        sel = RowSelection.from_raw(idx, len(self._objects))
        if len(sel.positions) != len(objs):
            raise ValueError("Length of objs sequence must match length of resolved index.")
        for i, obj in zip(sel.positions, objs):
            self._objects[int(i)] = obj

    def n_rows(self) -> int:
        return len(self._objects)

    def gather_materialized_state(self, index_map: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if index_map is not None:
            return {"objects": [copy.deepcopy(self._objects[i]) for i in index_map]}
        return {"objects": list(self._objects)}

    def save(
        self,
        path: Path,
        object_writer: Optional[Callable[[List[Any], Path], None]] = None,
    ) -> None:
        """
        Persist objects using an explicit *object_writer* callback.

        Parameters
        ----------
        path : Path
            Bundle root directory.  The writer receives ``path / "objects"``.
        object_writer : Callable[[List[Any], Path], None]
            **Required when there are objects to persist.**  Receives the full
            object list and the ``objects/`` sub-directory path.  The caller
            is responsible for choosing a safe, domain-appropriate format.

        Raises
        ------
        RuntimeError
            If *object_writer* is ``None`` and the store is non-empty.
        """
        obj_dir = path / "objects"
        obj_dir.mkdir(exist_ok=True)

        if not self._objects:
            # Nothing to write — create an empty sentinel so load() can detect
            # an intentionally empty store without raising.
            (obj_dir / ".empty").touch()
            return

        if object_writer is None:
            raise RuntimeError(
                "MemoryObjectStore.save() requires an explicit `object_writer` callback. "
                "No default pickle serialization is available. "
                "Supply a writer via BaseTable.save(object_writer=...) or by overriding "
                "`_get_default_object_writer()` on your Table subclass."
            )

        object_writer(self._objects, obj_dir)

    @classmethod
    def load(
        cls,
        path: Path,
        object_reader: Optional[Callable[[Path], List[Any]]] = None,
    ) -> "MemoryObjectStore":
        """
        Restore objects using an explicit *object_reader* callback.

        Parameters
        ----------
        path : Path
            Bundle root directory.  The reader receives ``path / "objects"``.
        object_reader : Callable[[Path], List[Any]]
            **Required when persisted objects exist.**  Returns the reconstructed
            object list from the ``objects/`` sub-directory.

        Raises
        ------
        RuntimeError
            If *object_reader* is ``None`` and a non-empty objects directory is
            found.
        """
        obj_dir = path / "objects"

        if not obj_dir.exists():
            return cls(objects=[])

        # Intentionally-empty store (written by save() when list was empty).
        if (obj_dir / ".empty").exists() and not any(
            p for p in obj_dir.iterdir() if p.name != ".empty"
        ):
            return cls(objects=[])

        if object_reader is None:
            raise RuntimeError(
                "MemoryObjectStore.load() requires an explicit `object_reader` callback. "
                "No default pickle deserialization is available. "
                "Supply a reader via EagerMemoryBackend.load(object_reader=...) or by "
                "overriding `_make_object_reader()` on your Table subclass."
            )

        return cls(objects=object_reader(obj_dir))