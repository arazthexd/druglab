"""
druglab.db.backend.base.mixins.feature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Abstract base interface for feature handling in storage backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict, TYPE_CHECKING

import numpy as np

from ._lifecycle import _LifecycleBase

if TYPE_CHECKING:
    from druglab.db.indexing import INDEX_LIKE

__all__ = ['BaseFeatureMixin']

class BaseFeatureMixin(_LifecycleBase, ABC):
    """
    Mixin for feature handling in backends.
    """

    @abstractmethod
    def get_feature(
        self,
        name: str,
        idx: Optional[INDEX_LIKE] = None,
    ) -> np.ndarray:
        """
        Fetch a feature array with strict query pushdown.

        Parameters
        ----------
        name
            Feature key.
        idx
            Row selector.  If None, return the full array. Otherwise,
            selected rows will only be returned.

        Returns
        -------
        np.ndarray
            The (possibly subset) feature array.
        """

    def get_features(
        self,
        names: Optional[List[str]] = None,
        idx: Optional[INDEX_LIKE] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Fetch multiple feature arrays with strict query pushdown.
        
        Can be overridden to provide a more efficient implementation.
        
        Parameters
        ----------
        names
            List of feature keys.
        idx
            Row selector.  If None, return the full arrays. Otherwise,
            selected rows will only be returned for each feature.
        
        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary mapping feature keys to feature arrays.
        """

        if names is None:
            names = self.get_feature_names()
        return {name: self.get_feature(name, idx) for name in names}

    @abstractmethod
    def update_feature(
        self,
        name: str,
        array: np.ndarray,
        idx: Optional[INDEX_LIKE] = None,
        na: Any = None,
        **kwargs
    ) -> None:
        """Add or perform a partial, in-place update of a feature array."""

    def update_features(
        self,
        arrays: Dict[str, np.ndarray],
        idx: Optional[INDEX_LIKE] = None,
        na: Any = None,
        **kwargs
    ) -> None:
        """Add or perform a partial, in-place update of multiple feature arrays."""
        for name, array in arrays.items():
            self.update_feature(name, array, idx, na, **kwargs)

    @abstractmethod
    def drop_feature(self, name: str) -> None:
        """Remove a feature by name. Raises ``KeyError`` if absent."""

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return the list of stored feature keys."""

    @abstractmethod
    def get_feature_shape(self, name: str) -> tuple:
        """Return the shape of a feature array."""

    def _n_feature_rows(self) -> int:
        """Return the number of rows in each feature array."""
        if not self.get_feature_names():
            return len(self)
        return self.get_feature_shape(self.get_feature_names()[0])[0]

    def _validate_features(self) -> None:
        """Validate the backend's feature schema."""
        if not self.get_feature_names():
            return
        n = self._n_feature_rows()
        for name in self.get_feature_names():
            if n != self.get_feature_shape(name)[0]:
                raise ValueError(
                    f"Feature '{name}' has {self.get_feature_shape(name)[0]} rows, "
                    f"expected {n}"
                )
            
    def post_initialize_validate(self) -> None:
        """Validate feature domain after full init; then propagate."""
        self._validate_features()
        super().post_initialize_validate()