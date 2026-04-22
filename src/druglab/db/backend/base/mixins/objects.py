"""
druglab.db.backend.base.mixins.objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Abstract base interface for object handling in storage backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union, TYPE_CHECKING

from ._lifecycle import _LifecycleBase

if TYPE_CHECKING:
    from druglab.db.indexing import INDEX_LIKE

__all__ = ['BaseObjectMixin']

class BaseObjectMixin(_LifecycleBase, ABC):
    """
    Mixin for object handling in backends.
    """

    @abstractmethod
    def get_objects(
        self,
        idx: Optional[INDEX_LIKE] = None
    ) -> Union[Any, List[Any]]:
        """
        Fetch one or multiple objects with backend-level query pushdown.

        Parameters
        ----------
        idx
            ``int``        → return a single object.
            ``slice``      → return a list of objects.
            ``List[int]``  → return a list of objects in the specified order.
            ``np.ndarray`` → return a list of objects based on boolean/int mask.
            ``None``       → return all objects.
            
        Returns
        -------
        Union[Any, List[Any]]
            A single object if idx is an int, otherwise a list of objects.
        """

    @abstractmethod
    def update_objects(
        self,
        objs: Union[Any, List[Any]],
        idx: Optional[INDEX_LIKE] = None,
        **kwargs
    ) -> None:
        """
        Perform an in-place partial or full update of stored objects.

        Parameters
        ----------
        objs : Union[Any, List[Any]]
            The object or sequence of objects to insert.
        idx : Optional[INDEX_LIKE], default None
            The specific index/indices to overwrite. If None, the entire 
            internal list is replaced by `objs`.

        Raises
        ------
        ValueError
            If `idx` is a sequence but its length does not match `objs`.
        """
        

    def set_objects(self, objs: List[Any], **kwargs) -> None:
        """
        Replace the entire object store with a new list of objects.

        Parameters
        ----------
        objs : List[Any]
            The new list of objects that will completely overwrite the existing store.
        """
        if self._n_objects() != len(objs):
            raise ValueError(
                f"new objects has {len(objs)} items but backend has {self._n_objects()}"
            )
        self.update_objects(objs, **kwargs)

    @abstractmethod
    def _n_objects(self) -> int:
        """
        Get the total number of stored objects.

        Returns
        -------
        int
            Length of the internal object list.
        """

    def _validate_objects(self) -> None:
        """Validate the backend's object schema."""
        return
    
    def post_initialize_validate(self) -> None:
        """Validate object domain after full init; then propagate."""
        self._validate_objects()
        super().post_initialize_validate()
