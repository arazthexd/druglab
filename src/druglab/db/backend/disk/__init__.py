"""
druglab.db.backend.disk
~~~~~~~~~~~~~~~~~~~~~~~~
Out-of-core storage backends for DrugLab tables.

Classes
-------
ZarrFeatureStore
    Out-of-core feature store backed by Zarr v3 with an on-disk rollback journal.
"""

from .zarr import ZarrFeatureStore

__all__ = ["ZarrFeatureStore"]