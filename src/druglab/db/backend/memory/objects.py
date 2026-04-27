"""In-memory object store."""

from __future__ import annotations

import copy
import pickle
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

        if isinstance(idx, int):
            n = len(self._objects)
            i = int(idx)
            if i >= n or i < -n:
                raise IndexError(f"index {idx} is out of bounds for axis 0 with size {n}")
            index = n + i if i < 0 else i
            return self._objects[index]

        sel = RowSelection.from_raw(idx, len(self._objects))
        return sel.apply_to_list(self._objects)

    def update_objects(self, objs: Union[Any, List[Any]], idx: Optional["INDEX_LIKE"] = None) -> None:
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
            raise ValueError("Length of objs sequence must match length of resolved index.")
        for i, obj in zip(sel.positions, objs):
            self._objects[int(i)] = obj

    def n_rows(self) -> int:
        return len(self._objects)

    def gather_materialized_state(self, index_map: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if index_map is not None:
            return {"objects": [copy.deepcopy(self._objects[i]) for i in index_map]}
        return {"objects": list(self._objects)}

    def save(self, path: Path, object_writer: Optional[Callable[[List[Any], Path], None]] = None) -> None:
        obj_dir = path / "objects"
        obj_dir.mkdir(exist_ok=True)

        if object_writer is not None:
            object_writer(self._objects, obj_dir)
            return

        with open(obj_dir / "objects.pkl", "wb") as f:
            pickle.dump({"format": "stream_v2", "count": len(self._objects), "serialized": False}, f)
            for obj in self._objects:
                pickle.dump(obj, f)

    @classmethod
    def load(
        cls,
        path: Path,
        object_reader: Optional[Callable[[Path], List[Any]]] = None,
    ) -> "MemoryObjectStore":
        if object_reader is not None:
            return cls(objects=object_reader(path / "objects"))

        obj_path = path / "objects" / "objects.pkl"
        if not obj_path.exists():
            print("WARNING: No objects found when loading bundle.")
            return cls(objects=[])

        with open(obj_path, "rb") as f:
            raw_payload = pickle.load(f)
            if isinstance(raw_payload, dict) and raw_payload.get("format") in {"stream_v1", "stream_v2"}:
                count = int(raw_payload["count"])
                return cls(objects=[pickle.load(f) for _ in range(count)])
            return cls(objects=raw_payload)


MemoryObjectMixin = MemoryObjectStore