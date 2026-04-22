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
"""

from typing import Any

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