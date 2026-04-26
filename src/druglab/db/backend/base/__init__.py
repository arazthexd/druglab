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

from typing import Any, Optional
from typing_extensions import Self
from pathlib import Path
import uuid as _uuid_mod

import numpy as np

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

    Thread-safety notice
    --------------------
    Concrete storage backends mutate internal arrays/lists in place and do
    not provide write locks. Concurrent writes against the same backend
    instance are not thread-safe and are not process-safe.

    Pipeline orchestration is responsible for synchronization. Multiprocessing
    code must use ``OverlayBackend`` scatter-gather (prefetch -> detach ->
    worker mutation -> attach -> commit) instead of sharing mutable base
    backends across workers.
 
    Lifecycle orchestration
    ------------------------
    ``__init__`` fires three hooks in order after the cooperative MRO chain:
 
    1. ``initialize_storage_context(**kwargs)`` -- domain setup
    2. ``bind_capabilities()``                  -- inter-mixin wiring
    3. ``post_initialize_validate()``            -- consistency checks
 
    Concrete backends assembling multiple mixins do **not** need to override
    ``__init__`` for boilerplate: each mixin handles its own state, and the
    hooks handle the rest.

    ``backend.schema_uuid`` is a per-instance random UUID used by
    ``OverlayBackend.attach()`` to verify that a re-attached backend is the
    same instance (or an intentional clone) as the one detached from.
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

        # Assign a unique identity to every concrete backend instance.
        # This is done *before* the lifecycle hooks so that SchemaIdentity
        # can be captured inside OverlayBackend.__init__.
        if not hasattr(self, "schema_uuid"):
            self.schema_uuid: str = str(_uuid_mod.uuid4())
        
        # Fire lifecycle hooks in declared order.
        # Hooks receive the full original kwargs dict so specialized mixins
        # can consume what they need in initialize_storage_context.
        self.initialize_storage_context(**kwargs)
        self.bind_capabilities()
        self.post_initialize_validate()

    # ------------------------------------------------------------------
    # Clones (Partial/Full Deep Copies)
    # ------------------------------------------------------------------

    def clone(
        self,
        target_path: Optional[Path] = None,
        index_map: Optional[np.ndarray] = None,
    ) -> "BaseStorageBackend":
        """
        Build a new backend instance of *this* class with (optionally sliced)
        state gathered via the cooperative ``_gather_materialized_state`` hook.
 
        This is the single point where ``OverlayBackend.materialize()`` creates
        its Phase-A clone.  Because ``_gather_materialized_state`` is MRO-
        cooperative, any custom mixin that stores additional state only needs
        to implement that one hook; no changes to ``clone`` are needed.
 
        Parameters
        ----------
        target_path : Path, optional
            Reserved for future out-of-core backends.
        index_map : np.ndarray of dtype np.intp, optional
            Absolute row positions to include.  ``None`` → all rows.
 
        Returns
        -------
        BaseStorageBackend
            A new instance of ``type(self)`` with state matching *index_map*.
        """
        gathered = self._gather_materialized_state(
            target_path=target_path,
            index_map=index_map,
        )
        new_instance = self.__class__(**gathered)
        # Clones intentionally get a NEW uuid so attach() distinguishes them.
        new_instance.schema_uuid = str(_uuid_mod.uuid4())
        return new_instance
    
    def materialize(
        self,
        target_path: Optional[Path] = None,
    ) -> "BaseStorageBackend":
        """
        Return a disconnected backend instance representing this backend's
        current logical state.

        Concrete backends are already materialized, so their safe behavior is
        to return a deep copy via ``clone()``. Proxy backends (for example
        ``OverlayBackend``) may override this method to collapse deferred
        deltas into a concrete backend.
        """
        return self.clone(target_path=target_path)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        path: Path | str,
        **kwargs: Any
    ) -> None:
        """
        Persist backend state into a ``.dlb`` bundle directory.
        
        Delegates to the cooperative ``save_storage_context`` MRO chain so
        that each mixin (metadata, objects, features) handles its own domain.
        
        Parameters
        ----------
        path : Path
            The target ``.dlb`` directory (pre-created by the caller).
        **kwargs
            Forwarded down the chain.  Recognised kwargs (e.g. ``object_writer``)
            are consumed by the relevant mixin; all others reach the terminal
            node and are silently absorbed.
        """
        if path.exists():
            print("WARNING: A .dlb bundle already exists. Overwriting.")
        path.mkdir(parents=True, exist_ok=True)
        self.save_storage_context(
            path=path, 
            **kwargs
        )

    @classmethod
    def load(
        cls,
        path: Path,
        **kwargs: Any
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
        **kwargs
            Forwarded down the chain.  Recognised kwargs (e.g. ``object_reader``)
            are consumed by the relevant mixin; all others reach the terminal
            node and are silently absorbed.
 
        Returns
        -------
        BaseStorageBackend
            A fully populated instance of the backend.
        """
        path = Path(path)
        cls_kwargs = cls.load_storage_context(
            path=path,
            **kwargs
        )
        return cls(**cls_kwargs)
    
    def get_name(self) -> str:
        return self.__class__.__name__
    
    def get_module(self) -> str:
        return self.__class__.__module__

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