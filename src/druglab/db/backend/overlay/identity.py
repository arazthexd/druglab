"""
druglab.db.backend.overlay.identity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Schema identity helpers for safe overlay attach/detach operations.

Classes
-------
DetachedStateError
    Raised when a read operation on a detached overlay cannot be satisfied
    from either the delta or the prefetch cache.

SchemaIdentity
    A lightweight fingerprint attached to every ``BaseStorageBackend``
    instance.  Used by ``OverlayBackend.attach()`` to validate that the
    backend being re-attached is dimensionally and structurally compatible
    with what was detached.
"""

from __future__ import annotations

import hashlib
import uuid as _uuid_mod
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class DetachedStateError(RuntimeError):
    """
    Raised when an overlay is in the detached state and a read request cannot
    be satisfied from the delta or the prefetch cache.

    This typically means the caller should either:
    * re-attach the overlay with ``.attach(base_backend)``, or
    * ensure the required data was prefetched before detaching.
    """


# ---------------------------------------------------------------------------
# SchemaIdentity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SchemaIdentity:
    """
    Lightweight, immutable fingerprint of a ``BaseStorageBackend``.

    Fields
    ------
    uuid : str
        A random UUID generated at backend creation time.  Guaranteed unique
        across process boundaries.
    n_rows : int
        Row count at the moment the identity was captured.
    feature_names : tuple[str, ...]
        Sorted tuple of feature names present when the identity was captured.
    meta_cols : tuple[str, ...]
        Sorted tuple of metadata column names when the identity was captured.
    """

    uuid: str
    n_rows: int
    feature_names: tuple
    meta_cols: tuple

    # ------------------------------------------------------------------
    @classmethod
    def capture(cls, backend) -> "SchemaIdentity":
        """Snapshot the current schema of *backend*."""
        return cls(
            uuid=backend.schema_uuid,
            n_rows=len(backend),
            feature_names=tuple(sorted(backend.get_feature_names())),
            meta_cols=tuple(sorted(backend.get_metadata_columns())),
        )

    # ------------------------------------------------------------------
    def validate_compatible(self, other: "SchemaIdentity") -> None:
        """
        Raise ``ValueError`` if *other* is not compatible with this identity.

        Compatibility requires:
        * Same UUID (same backend instance or explicit clone).
        * Same row count.
        * Same (sorted) feature names.
        * Same (sorted) metadata columns.
        """
        errors: List[str] = []

        if self.uuid != other.uuid:
            errors.append(
                f"UUID mismatch: overlay expects '{self.uuid}', "
                f"got '{other.uuid}'."
            )

        if self.n_rows != other.n_rows:
            errors.append(
                f"Row count mismatch: overlay expects {self.n_rows}, "
                f"backend has {other.n_rows}."
            )

        if self.feature_names != other.feature_names:
            errors.append(
                f"Feature schema mismatch: "
                f"expected {list(self.feature_names)}, "
                f"got {list(other.feature_names)}."
            )

        if self.meta_cols != other.meta_cols:
            errors.append(
                f"Metadata schema mismatch: "
                f"expected {list(self.meta_cols)}, "
                f"got {list(other.meta_cols)}."
            )

        if errors:
            raise ValueError(
                "Cannot attach backend: schema incompatibility detected.\n"
                + "\n".join(f"  • {e}" for e in errors)
            )