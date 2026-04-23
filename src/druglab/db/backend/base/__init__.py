"""
druglab.db.backend.base
~~~~~~~~~~~~~~~~~~~~~~~
Abstract base interface that all storage backends must implement.

The interface enforces strict Query Pushdown: index/slice arguments must be
passed directly to the backend so that out-of-core implementations (Zarr,
SQLite, HDF5) can read exactly the bytes they need without loading full
arrays into memory first.

Index normalisation is handled by ``druglab.db.indexing``, which is the
single source of truth for all row-addressing in DrugLab.
"""

from __future__ import annotations

from typing import Any, Callable, Optional
from typing_extensions import Self
from pathlib import Path

from .mixins import (
    BaseObjectMixin,
    BaseFeatureMixin,
    BaseMetadataMixin
)

class BaseStorageBackend(
    BaseObjectMixin,
    BaseMetadataMixin,
    BaseFeatureMixin
):
    """
    Minimal unified interface for managing DrugLab table state.
 
    Lifecycle orchestration
    ------------------------
    ``__init__`` fires three hooks in order after the cooperative MRO chain:
 
    1. ``initialize_storage_context(**kwargs)`` -- domain setup
    2. ``bind_capabilities()``                  -- inter-mixin wiring
    3. ``post_initialize_validate()``            -- consistency checks
 
    Concrete backends assembling multiple mixins do **not** need to override
    ``__init__`` for boilerplate: each mixin handles its own state, and the
    hooks handle the rest.
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self, **kwargs: Any) -> None:
        # This is the terminal node of the cooperative __init__ chain.
        # Domain mixins above us (MemoryObjectMixin, MemoryMetadataMixin, etc.)
        # each consume their own recognized kwarg (objects=, metadata=,
        # features=) and forward the rest via super().__init__(**remaining).
        # Any unrecognized kwargs that reach here are silently absorbed rather
        # than forwarded to object.__init__ (which accepts none).
        # This allows custom mixin __init__ methods to accept extra kwargs
        # (e.g. connection_string=) without breaking the chain.
        #
        # NOTE: do NOT call super().__init__(**kwargs) here -- object.__init__
        # rejects keyword arguments.
        # super().__init__() is intentionally NOT called with kwargs.
        # (object.__init__() takes no extra arguments.)
 
        # Fire lifecycle hooks in declared order.
        # Hooks receive the full original kwargs dict so specialized mixins
        # can consume what they need in initialize_storage_context.
        self.initialize_storage_context(**kwargs)
        self.bind_capabilities()
        self.post_initialize_validate()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        path: Path | str,
        serializer: Callable[[Any], bytes] | None = None
    ) -> None:
        """
        Persist backend state into a ``.dlb`` bundle directory.
        
        Delegates to the cooperative ``save_storage_context`` MRO chain so
        that each mixin (metadata, objects, features) handles its own domain.
        
        Parameters
        ----------
        path : Path
            The target ``.dlb`` directory (pre-created by the caller).
        serializer : Optional[Callable], default None
            An optional function `(obj) -> bytes` to serialize generic objects.
            Usually provided by the caller table.
        """
        path = Path(path).with_suffix(".dlb")
        if path.exists() and not path.is_dir():
            raise FileExistsError(f"File already exists: {path}")
        if path.exists():
            print("WARNING: A .dlb bundle already exists. Overwriting.")
        path.mkdir(parents=True, exist_ok=True)
        self.save_storage_context(
            path=path, 
            serializer=serializer
        )

    @classmethod
    def load(
        cls,
        path: Path,
        deserializer: Callable[[bytes], Any] | None = None
    ) -> Self:
        """
        Reconstruct the backend from a ``.dlb`` bundle directory.
 
        Delegates to the cooperative ``load_storage_context`` MRO chain so
        that each mixin reads its own domain.  The accumulated kwargs dict is
        then passed directly to ``cls()``.
 
        Parameters
        ----------
        path : Path
            The location of the ``.dlb`` bundle.
        deserializer : Optional[Callable], default None
            An optional function `(bytes) -> obj` to reconstruct stored objects.
            Usually provided by the caller table.
 
        Returns
        -------
        BaseStorageBackend
            A fully populated instance of the backend.
        """
        path = Path(path)
        kwargs = cls.load_storage_context(
            path=path,
            deserializer=deserializer
        )
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Validate backend-wide dimensional consistency.

        STRONGLY SUGGESTED: Validates the entire backend by checking
        individual domain integrity and ensuring dimension alignment.

        Raises
        ------
        ValueError
            If any dimension mismatch is detected.
        """
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