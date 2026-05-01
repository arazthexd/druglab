from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Callable, List

import numpy as np

from .base import BaseStorageBackend
from .base.stores import BaseFeatureStore, BaseMetadataStore, BaseObjectStore

__all__ = ["CompositeStorageBackend"]

_DEFAULT_METADATA_FORMAT = "parquet"


class CompositeStorageBackend(BaseStorageBackend):
    BACKEND_NAME = "CompositeStorageBackend"

    def __init__(
        self,
        object_store: BaseObjectStore,
        metadata_store: BaseMetadataStore,
        feature_store: BaseFeatureStore,
    ) -> None:
        super().__init__()
        self._object_store = object_store
        self._metadata_store = metadata_store
        self._feature_store = feature_store
        self.validate()

    def __len__(self) -> int:
        return self._n_objects()

    # ------------------------------------------------------------------
    # Domain delegation
    # ------------------------------------------------------------------

    def get_objects(self, idx=None):
        return self._object_store.get_objects(idx)

    def update_objects(self, objs, idx=None, **kwargs) -> None:
        self._object_store.update_objects(objs, idx)

    def _n_objects(self) -> int:
        return self._object_store.n_rows()

    def get_metadata(self, idx=None, cols=None):
        return self._metadata_store.get_metadata(idx, cols)

    def add_metadata_column(self, name, value, idx=None, na=None, **kwargs) -> None:
        self._metadata_store.add_metadata_column(name, value, idx=idx, na=na)

    def update_metadata(self, values, idx=None, **kwargs) -> None:
        self._metadata_store.update_metadata(values, idx=idx)

    def drop_metadata_columns(self, cols=None) -> None:
        self._metadata_store.drop_metadata_columns(cols)

    def get_metadata_columns(self):
        return self._metadata_store.get_metadata_columns()

    def _n_metadata_rows(self) -> int:
        return self._metadata_store.n_rows()

    def get_feature(self, name: str, idx=None):
        return self._feature_store.get_feature(name, idx)

    def update_feature(self, name: str, array: np.ndarray, idx=None, na=None, **kwargs) -> None:
        self._feature_store.update_feature(name, array, idx=idx, na=na)

    def drop_feature(self, name: str) -> None:
        self._feature_store.drop_feature(name)

    def get_feature_names(self):
        return self._feature_store.get_feature_names()

    def get_feature_shape(self, name: str) -> tuple:
        return self._feature_store.get_feature_shape(name)

    def _n_feature_rows(self) -> int:
        return self._feature_store.n_rows()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        expected_len = len(self)
        meta_len = self._n_metadata_rows()
        feat_len = self._n_feature_rows()
        obj_len = self._n_objects()
        if not (expected_len == meta_len == feat_len == obj_len):
            raise ValueError(
                f"Backend Dimension Mismatch!\n"
                f"Global Length: {expected_len}\n"
                f"Metadata Rows: {meta_len}\n"
                f"Feature Rows:  {feat_len}\n"
                f"Object Count:  {obj_len}"
            )

    # ------------------------------------------------------------------
    # Materialized state
    # ------------------------------------------------------------------

    def _gather_materialized_state(
        self,
        target_path: Optional[Path] = None,
        index_map: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        result = {}
        result.update(self._object_store.gather_materialized_state(index_map=index_map))
        result.update(self._metadata_store.gather_materialized_state(index_map=index_map))
        result.update(self._feature_store.gather_materialized_state(index_map=index_map))
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_storage_context(
        self,
        path: Path,
        object_writer: Optional[Callable[[List[Any], Path], None]] = None,
        metadata_format: str = _DEFAULT_METADATA_FORMAT,
        **kwargs: Any,
    ) -> None:
        """
        Persist all three domain stores and write a ``composite_manifest.json``.

        Parameters
        ----------
        path : Path
            Bundle root directory.
        metadata_format : {"parquet", "csv"}
            Format used to serialise the metadata DataFrame.  The choice is
            recorded in the manifest so ``load_storage_context`` can instruct
            the metadata store to read the matching file extension.
        object_writer: Optional[Callable[[List[Any], Path], None]]
            Forwarded to the object store verbatim.
        **kwargs
            Any remaining kwargs are forwarded to ``_feature_store.save()``.
        """
        self._object_store.save(path, object_writer=object_writer)
        self._metadata_store.save(path, format=metadata_format)
        self._feature_store.save(path)

        manifest = {
            "object_store": {
                "module": self._object_store.__class__.__module__,
                "class": self._object_store.__class__.__name__,
            },
            "metadata_store": {
                "module": self._metadata_store.__class__.__module__,
                "class": self._metadata_store.__class__.__name__,
                # ← The format decision is persisted here so load can reconstruct
                #   the correct file without guessing.
                "format": metadata_format,
            },
            "feature_store": {
                "module": self._feature_store.__class__.__module__,
                "class": self._feature_store.__class__.__name__,
            },
        }
        (path / "composite_manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load_storage_context(cls, path: Path, **kwargs: Any) -> Dict[str, Any]:
        """
        Rebuild the three domain stores from a bundle directory.

        The ``"format"`` key stored in the manifest is forwarded to the
        metadata store's ``.load()`` so it reads the correct file type.
        """
        manifest_path = path / "composite_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("Missing composite_manifest.json.")

        manifest = json.loads(manifest_path.read_text())

        def _load_store(config: Dict[str, str], p: Path, extra_kwargs: Dict[str, Any]):
            mod = importlib.import_module(config["module"])
            store_cls = getattr(mod, config["class"])
            # Merge caller kwargs with manifest-level kwargs (manifest wins for
            # format-style keys; caller wins for things like object_reader).
            merged = {**extra_kwargs}
            try:
                return store_cls.load(p, **merged)
            except TypeError:
                return store_cls.load(p)

        # Extract format from manifest so the metadata store gets the right hint.
        meta_config = manifest["metadata_store"]
        metadata_format = meta_config.get("format", _DEFAULT_METADATA_FORMAT)

        # Caller-supplied object_reader takes precedence.
        object_reader = kwargs.get("object_reader")

        return {
            "object_store": _load_store(
                manifest["object_store"],
                path,
                {"object_reader": object_reader} if object_reader is not None else {},
            ),
            "metadata_store": _load_store(
                meta_config,
                path,
                {"format": metadata_format},
            ),
            "feature_store": _load_store(
                manifest["feature_store"],
                path,
                {},
            ),
        }