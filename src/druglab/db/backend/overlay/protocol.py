"""
druglab.db.backend.overlay.protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Formal protocol / ABC that all overlay mixins depend on.

Instead of relying on duck-typed implicit attribute access to ``_base``,
``_index_map`` and ``_resolve_overlay_idx``, every mixin now declares its
dependency through ``OverlayContextProtocol``.  The concrete
``OverlayBackend`` class satisfies the protocol at runtime.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from druglab.db.backend.base import BaseStorageBackend
    from druglab.db.indexing import INDEX_LIKE

class OverlayContextProtocol:
    """
    Abstract context that all overlay domain mixins require.

    Concrete ``OverlayBackend`` satisfies this protocol.  Any mixin can
    type-annotate ``self`` as ``OverlayContextProtocol`` to get IDE support
    and static-analysis benefits.

    Properties
    ----------
    _base : BaseStorageBackend | None
        The underlying concrete backend.  May be ``None`` when detached.
    _index_map : np.ndarray
        1-D ``np.intp`` array mapping overlay row positions → base positions.

    Methods
    -------
    _resolve_overlay_idx(idx) -> np.ndarray
        Convert any ``INDEX_LIKE`` to a 1-D absolute overlay-position array.
    _translate(overlay_positions) -> np.ndarray
        Map overlay positions to base positions via ``_index_map``.
    _n_rows() -> int
        Number of rows this overlay exposes.
    """

    # -----------------------------------------------------------------
    # Attributes declared here so type checkers are happy; concrete class
    # provides the real values.
    # -----------------------------------------------------------------

    _base: Optional["BaseStorageBackend"]
    _index_map: np.ndarray

    def _resolve_overlay_idx(self, idx: Optional["INDEX_LIKE"]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _translate(self, overlay_positions: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _n_rows(self) -> int:  # pragma: no cover
        raise NotImplementedError