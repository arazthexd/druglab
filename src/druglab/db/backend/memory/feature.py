"""
druglab.db.backend.memory.feature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In-memory storage mixin for features.

Currently, the only supported mixin is ``MemoryFeatureMixin`` which manages features
in RAM as a dictionary of NumPy arrays. However, other mixins may be supported in the future.
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from ...indexing import RowSelection
from ..base.mixins import BaseFeatureMixin

if TYPE_CHECKING:
    from druglab.db.indexing import INDEX_LIKE



class MemoryFeatureMixin(BaseFeatureMixin):
    """
    In-memory feature storage mixin utilizing a dictionary of NumPy arrays.

    Lifecycle
    ---------
    ``initialize_storage_context`` sets up ``self._features`` from the stashed
    init arg.  ``bind_capabilities`` builds the initial ``FeatureRegistry``
    from whatever is in ``self._features`` at that point, so no manual
    registry construction is needed in ``EagerMemoryBackend.__init__``.
    """

    def initialize_storage_context(
        self, 
        features: Optional[Dict[str, np.ndarray]] = None, 
        **kwargs: Any
    ) -> None:
        """Initialize storage context via ``features`` kwarg.
        
        Parameters
        ----------
        features : Dict[str, np.ndarray], optional
            Dictionary of NumPy arrays to initialize storage context with.
        **kwargs
            Additional keyword arguments. These are passed in the MRO chain.
        """
        self._features = features or {}
        super().initialize_storage_context(**kwargs)

    def get_feature(self, name: str, idx: Optional[INDEX_LIKE] = None) -> np.ndarray:

        arr = self._features[name]
        sel = RowSelection.from_raw(idx, arr.shape[0])
        return sel.apply_to(arr)

    def update_feature(
        self,
        name: str,
        array: np.ndarray,
        idx: Optional[INDEX_LIKE] = None,
        na: Any = None,
        **kwargs
    ) -> None:
        
        if name not in self._features:
            if idx is None:
                if array.shape[0] != self._n_feature_rows():
                    raise ValueError(
                        f"Length of array's first dimension ({array.shape[0]}) must "
                        f"match number of feature rows ({self._n_feature_rows()})."
                    )
                self._features[name] = np.asarray(array).copy()
            else:
                sel = RowSelection.from_raw(idx, self._n_feature_rows() or len(array))
                n_rows = self._n_feature_rows() or (
                    sel.positions.max() + 1 if len(sel.positions) > 0 else 0
                )
                shape = (n_rows, *np.asarray(array).shape[1:])

                if na is None and np.issubdtype(np.asarray(array).dtype, np.floating):
                    na = np.nan
                elif na is None:
                    na = 0

                full_arr = np.full(shape, na, dtype=np.asarray(array).dtype)
                full_arr[sel.positions] = array
                self._features[name] = full_arr
        else:
            if idx is None:
                arr = np.asarray(array)
                if arr.shape[0] != self._features[name].shape[0]:
                    raise ValueError(
                        f"Cannot update feature '{name}': array has {arr.shape[0]} "
                        f"rows but existing feature has "
                        f"{self._features[name].shape[0]} rows."
                    )
                self._features[name] = arr.copy()
            else:
                sel = RowSelection.from_raw(idx, self._features[name].shape[0])
                self._features[name][sel.positions] = array

    def drop_feature(self, name: str) -> None:
        del self._features[name]

    def get_feature_names(self) -> List[str]:
        return list(self._features.keys())

    def get_feature_shape(self, name: str) -> tuple:
        return self._features[name].shape