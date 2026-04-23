"""
druglab.db.backend.base.mixins._lifecycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Cooperative lifecycle hooks for domain mixins and capability mixins.

Intented for clean MRO chains during initialization of storage backend
subclasses.

Hooks (in rough call-order per operation)
------------------------------------------
initialize_storage_context(**kwargs)
    Called early in ``__init__``.  Each mixin wires up its own storage and
    forwards remaining kwargs via ``super()``.
 
bind_capabilities()
    Called after the full ``__init__`` MRO chain.  Used for inter-mixin wiring.
 
post_initialize_validate()
    Called last, for cross-domain consistency assertions.

save_storage_context(path, **kwargs)
    Cooperative save hook.  Each mixin persists its own data to ``path``
    and forwards via ``super()``.
 
load_storage_context(path, **kwargs) -> dict   [classmethod]
    Cooperative load hook.  Each mixin reads its own data from ``path``
    and merges its contribution into the returned kwargs dict.
    The dict is passed directly to ``cls(**result)`` by the caller.
"""

from typing import Any, Dict, Optional
from pathlib import Path

import numpy as np

__all__ = ['_LifecycleBase']

class _LifecycleBase:
    """
    Mixin base that defines the three cooperative lifecycle hooks.
 
    All domain mixins and capability mixins inherit from this class so that
    ``super()``-based cooperative calls propagate correctly through any MRO.

    Every hook listed below is a **terminal node**: it absorbs remaining
    kwargs without forwarding to ``object``, which accepts none.  Concrete
    subclasses must call their ``super()`` counterpart so the full chain fires.
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
 
    def initialize_storage_context(self, **kwargs: Any) -> None:
        """
        Cooperative lifecycle hook: finalize mixin-level storage setup.
 
        Implementors must call ``super().initialize_storage_context(**kwargs)``
        *after* their own logic so the full cooperative chain fires.
 
        Unknown kwargs are swallowed at the terminal node.
        """
        # Terminal node -- absorbs remaining kwargs.
 
    def bind_capabilities(self) -> None:
        """
        Cooperative lifecycle hook: wire up inter-mixin capability references.
 
        Called once after the full ``__init__`` chain completes, before
        ``post_initialize_validate``.  Must call ``super().bind_capabilities()``.
        """
        # Terminal node -- absorbs remaining kwargs.
 
    def post_initialize_validate(self) -> None:
        """
        Cooperative lifecycle hook: cross-domain consistency validation.
 
        Called last, after both prior hooks.  Raise ``ValueError`` to signal
        an invalid initial state.  Must call ``super().post_initialize_validate()``.
        """
        # Terminal node -- absorbs remaining kwargs.

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def _gather_materialized_state(
        self,
        target_path: Optional[Path] = None,
        index_map: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Cooperative hook: collect (optionally sliced) domain state as init kwargs.
 
        Used by ``BaseStorageBackend.clone_concrete()`` to produce a new
        concrete backend of the same class, optionally restricted to the rows
        in *index_map*.
 
        Each mixin slices its own structures when ``index_map`` is provided::
 
            result = super()._gather_materialized_state(target_path, index_map)
            if index_map is not None:
                result["features"] = {k: v[index_map].copy() ...}
            else:
                result["features"] = {k: v.copy() ...}
            return result
 
        Parameters
        ----------
        target_path : Path, optional
            Reserved for out-of-core backends that write slices to disk rather 
            than returning in-memory structures.
        index_map : np.ndarray of dtype np.intp, optional
            1-D array of absolute positions to include.  ``None`` means all rows.
 
        Returns
        -------
        dict
            Merged kwargs dict suitable for passing to ``self.__class__.__init__``.
        """
        # Terminal node – returns empty dict.
        return {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_storage_context(self, path: Path, **kwargs: Any) -> None:
        """
        Cooperative lifecycle hook: serialize mixin-level state to *path*.
 
        Each mixin writes its own data (features, metadata, objects) to the
        bundle directory and then calls::
 
            super().save_storage_context(path, **kwargs)
 
        so the full cooperative chain fires.
 
        Parameters
        ----------
        path : Path
            The target ``.dlb`` bundle directory (already created by the caller).
        **kwargs
            Forwarded down the chain.  Recognised kwargs (e.g. ``object_writer``)
            are consumed by the relevant mixin; all others reach the terminal
            node and are silently absorbed.
        """
        # Terminal node – absorbs remaining kwargs.
 
    @classmethod
    def load_storage_context(cls, path: Path, **kwargs: Any) -> Dict[str, Any]:
        """
        Cooperative lifecycle hook: deserialize mixin-level state from *path*.
 
        Each mixin reads its own data from the bundle directory, adds its
        contribution to the dict returned by ``super()``, and returns the
        merged result.  The caller passes the final dict directly to
        ``cls(**result)``::
 
            result = super().load_storage_context(path, **kwargs)
            result["features"] = {...}
            return result
 
        Parameters
        ----------
        path : Path
            The source ``.dlb`` bundle directory.
        **kwargs
            Forwarded down the chain.  Recognised kwargs (e.g. ``object_reader``,
            ``mmap_features``) are consumed by the relevant mixin.
 
        Returns
        -------
        dict
            Merged kwargs dict suitable for passing to ``cls.__init__``.
        """
        # Terminal node – returns empty dict.
        return {}