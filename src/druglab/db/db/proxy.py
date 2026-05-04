from __future__ import annotations

from typing import Any, Dict, Optional
from typing_extensions import Self

from .base import BaseDB

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..types import ColsLike, IdxLike


class RestrictedDBProxy(BaseDB):
    """
    A lightweight wrapper around a parent database.

    Intercepts ``request_engine`` calls and returns view-restricted engine
    instances.  Resulting view engines are **cached** on the proxy so that
    multiple calls for the same engine name within a single table operation
    return the same object (avoiding mask drift between operations).

    The proxy also carries an optional ``cols`` selection that is forwarded
    to the engine's ``materialize`` call by the table's read path.
    """

    def __init__(
        self,
        parent_db: BaseDB,
        namespace: str,
        rows: Optional["IdxLike"],
        cols: Optional["ColsLike"] = None,
        **kwargs,
    ):
        super().__init__()
        self._parent = parent_db
        self._namespace = namespace
        self._rows = rows
        self._cols = cols
        self._kwargs = kwargs

        # Share the parent's table registry so cross-table queries still work
        # inside a view.
        self._tables = parent_db._tables

        # Cache of already-spawned view engines for this proxy.
        # Key: engine name (e.g. "duckdb", "pandas").
        # This ensures that two calls to request_engine("duckdb") within one
        # table operation return the *same* view object, not two independent
        # slices that could diverge if the parent engine is mutated between
        # calls.
        self._view_engine_cache: Dict[str, Any] = {}

    def request_engine(self, name: str) -> Any:
        if name in self._view_engine_cache:
            return self._view_engine_cache[name]

        # Ask the parent for the real root engine
        root_engine = self._parent.request_engine(name)

        # Slice it down to our rows (cols are not stored on the engine; they
        # are forwarded at materialize time by the table).
        if self._rows is not None:
            view_engine = root_engine.spawn_view(
                namespace=self._namespace,
                rows=self._rows,
                **self._kwargs,
            )
        else:
            # rows=None means "no row restriction" — hand back the root engine
            # directly so we don't wrap needlessly.
            view_engine = root_engine

        self._view_engine_cache[name] = view_engine
        return view_engine

    def spawn_restricted_view(
        self,
        namespace: str,
        rows: Optional["IdxLike"],
        cols: Optional["ColsLike"] = None,
        **kwargs,
    ) -> Self:
        # Chaining: wrap the proxy in another proxy.  The inner engine's
        # _combine_masks handles mask intersection automatically.
        return RestrictedDBProxy(
            parent_db=self,
            namespace=namespace,
            rows=rows,
            cols=cols,
            **kwargs,
        )

    @property
    def cols(self) -> Optional["ColsLike"]:
        """The column selection active on this proxy, if any."""
        return self._cols