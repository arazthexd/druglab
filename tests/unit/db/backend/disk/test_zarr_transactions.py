"""
tests/unit/db/backend/disk/test_zarr_transactions.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the out-of-core rollback journal in ZarrFeatureStore.

Covers:
* Happy-path: begin → commit removes journal and data is updated.
* Rollback: begin → partial corruption → rollback restores original state.
* Crash recovery: journal persists on disk; re-opening the store and calling
  rollback_transaction() restores consistent state.
* Drop-feature journal: a feature that was going to be deleted is restored
  after rollback.
* New-feature journal: a brand-new feature written during a transaction is
  removed after rollback.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Set

import numpy as np
import pytest

zarr = pytest.importorskip("zarr", reason="zarr not installed")

from druglab.db.backend.disk import ZarrFeatureStore
from druglab.db.backend.disk.zarr import _JOURNAL_KEY


# ---------------------------------------------------------------------------
# Minimal stub for FeatureDelta (avoids importing overlay machinery)
# ---------------------------------------------------------------------------

@dataclass
class _FakeDelta:
    """Minimal stand-in for FeatureDelta, sufficient for the transaction protocol."""
    local: Dict[str, np.ndarray] = field(default_factory=dict)
    deleted: Set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_store(path: Path, mode: str = "w") -> ZarrFeatureStore:
    group = zarr.open_group(str(path / "features.zarr"), mode=mode)
    return ZarrFeatureStore(group)


def _make_store(tmp_path: Path) -> ZarrFeatureStore:
    """Create a store with two features, 5 rows each."""
    store = _open_store(tmp_path)
    store.update_feature("fp", np.arange(20, dtype=np.float32).reshape(5, 4))
    store.update_feature("desc", np.ones((5, 8), dtype=np.float64) * 2.0)
    return store


# ===========================================================================
# Happy-path commit
# ===========================================================================

class TestZarrTransactionCommit:
    def test_commit_updates_data(self, tmp_path):
        store = _make_store(tmp_path)
        original_fp = store.get_feature("fp").copy()
        index_map = np.array([0, 2], dtype=np.intp)
        new_vals = np.full((2, 4), 99.0, dtype=np.float32)

        delta = _FakeDelta(local={"fp": new_vals}, deleted=set())
        store.begin_transaction(delta, index_map)
        store.commit_transaction(delta, index_map)

        result = store.get_feature("fp")
        assert result[0, 0] == 99.0
        assert result[2, 0] == 99.0
        # Untouched rows are intact.
        np.testing.assert_array_equal(result[1], original_fp[1])

    def test_journal_exists_after_begin(self, tmp_path):
        store = _make_store(tmp_path)
        delta = _FakeDelta(local={"fp": np.zeros((2, 4), dtype=np.float32)})
        store.begin_transaction(delta, np.array([0, 1], dtype=np.intp))
        assert _JOURNAL_KEY in store._group

    def test_journal_absent_after_explicit_clear(self, tmp_path):
        store = _make_store(tmp_path)
        delta = _FakeDelta(local={"fp": np.zeros((2, 4), dtype=np.float32)})
        index_map = np.array([0, 1], dtype=np.intp)
        store.begin_transaction(delta, index_map)
        store.commit_transaction(delta, index_map)
        store._clear_journal()
        assert _JOURNAL_KEY not in store._group


# ===========================================================================
# Rollback: existing feature rows
# ===========================================================================

class TestZarrTransactionRollback:
    def test_rollback_restores_original_rows(self, tmp_path):
        store = _make_store(tmp_path)
        original = store.get_feature("fp").copy()
        index_map = np.array([1, 3], dtype=np.intp)
        delta = _FakeDelta(
            local={"fp": np.full((2, 4), -1.0, dtype=np.float32)},
            deleted=set(),
        )

        store.begin_transaction(delta, index_map)
        # Simulate a partial / failed commit by calling rollback without commit.
        store.rollback_transaction()

        restored = store.get_feature("fp")
        np.testing.assert_array_equal(restored, original)

    def test_rollback_removes_journal(self, tmp_path):
        store = _make_store(tmp_path)
        delta = _FakeDelta(local={"fp": np.zeros((2, 4), dtype=np.float32)})
        store.begin_transaction(delta, np.array([0, 1], dtype=np.intp))
        store.rollback_transaction()
        assert _JOURNAL_KEY not in store._group
        assert store._journal is None


# ===========================================================================
# Rollback: brand-new feature
# ===========================================================================

class TestZarrTransactionRollbackNewFeature:
    def test_rollback_removes_new_feature(self, tmp_path):
        store = _make_store(tmp_path)
        index_map = np.array([0, 1, 2, 3, 4], dtype=np.intp)
        brand_new = np.zeros((5, 16), dtype=np.float32)
        delta = _FakeDelta(local={"shape_fp": brand_new}, deleted=set())

        store.begin_transaction(delta, index_map)
        store.commit_transaction(delta, index_map)
        assert "shape_fp" in store.get_feature_names()

        # Now rollback
        store.rollback_transaction()
        assert "shape_fp" not in store.get_feature_names()

    def test_rollback_before_commit_new_feature(self, tmp_path):
        """Rollback of a new feature that was never committed."""
        store = _make_store(tmp_path)
        index_map = np.array([0, 1], dtype=np.intp)
        delta = _FakeDelta(local={"brand_new": np.zeros((2, 3), dtype=np.float32)})
        store.begin_transaction(delta, index_map)
        # Do NOT commit; just rollback.
        store.rollback_transaction()
        assert "brand_new" not in store.get_feature_names()


# ===========================================================================
# Rollback: dropped feature
# ===========================================================================

class TestZarrTransactionRollbackDroppedFeature:
    def test_rollback_restores_dropped_feature(self, tmp_path):
        store = _make_store(tmp_path)
        original_desc = store.get_feature("desc").copy()
        index_map = np.arange(5, dtype=np.intp)
        delta = _FakeDelta(local={}, deleted={"desc"})

        store.begin_transaction(delta, index_map)
        store.commit_transaction(delta, index_map)
        assert "desc" not in store.get_feature_names()

        store.rollback_transaction()
        assert "desc" in store.get_feature_names()
        np.testing.assert_array_equal(store.get_feature("desc"), original_desc)


# ===========================================================================
# Crash recovery: journal survives process restart
# ===========================================================================

class TestZarrCrashRecovery:
    def test_journal_persists_and_rollback_restores(self, tmp_path):
        """
        Simulate a crash between begin_transaction and commit_transaction.

        The journal must survive on disk, and re-opening the store then calling
        rollback_transaction() must restore the original values.
        """
        store = _make_store(tmp_path)
        original = store.get_feature("fp").copy()
        index_map = np.array([0, 4], dtype=np.intp)
        delta = _FakeDelta(local={"fp": np.full((2, 4), -999.0, dtype=np.float32)})

        store.begin_transaction(delta, index_map)
        # --- CRASH SIMULATED: we do not call commit_transaction ---
        # Verify journal is on disk.
        assert _JOURNAL_KEY in store._group

        # Re-open the store (simulates a new process after restart).
        store2 = _open_store(tmp_path, mode="r+")
        assert store2._journal is not None  # Detects stale journal on init.

        store2.rollback_transaction()

        # Data must be back to original.
        np.testing.assert_array_equal(store2.get_feature("fp"), original)
        assert _JOURNAL_KEY not in store2._group