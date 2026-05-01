"""
tests/unit/db/backend/test_atomic_commit.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the atomic transaction protocol across all three storage domains.

Coverage
--------
* Unit tests for each store's journal (begin / commit / rollback)
* Integration test for CompositeStorageBackend.apply_deltas()
* "Poison pill" test: mid-commit failure rolls back all three domains
* Edge cases: new features, dropped features, empty deltas
"""

from __future__ import annotations

import copy
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from druglab.db.backend.memory import (
    EagerMemoryBackend,
    MemoryFeatureStore,
    MemoryMetadataStore,
    MemoryObjectStore,
)
from druglab.db.backend.overlay.deltas import FeatureDelta, MetadataDelta, ObjectDelta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backend(n: int = 3):
    """Return a populated EagerMemoryBackend with predictable state."""
    objects = [f"obj_{i}" for i in range(n)]
    metadata = pd.DataFrame({"label": [f"lbl_{i}" for i in range(n)], "score": [float(i) for i in range(n)]})
    features = {"fp": np.arange(n * 4, dtype=float).reshape(n, 4)}
    return EagerMemoryBackend(objects=objects, metadata=metadata, features=features)


def _index_map(*indices) -> np.ndarray:
    return np.array(indices, dtype=np.intp)


# ===========================================================================
# MemoryFeatureStore journal
# ===========================================================================

class TestMemoryFeatureStoreJournal:
    def _store(self):
        return MemoryFeatureStore(
            {"fp": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])}
        )

    def test_begin_journals_existing_rows(self):
        fs = self._store()
        delta = FeatureDelta()
        delta.set("fp", np.array([[99.0, 99.0]]))  # will update row 1
        idx = _index_map(1)

        fs.begin_transaction(delta, idx)

        assert "fp" in fs._journal
        entry = fs._journal["fp"]
        assert entry["existed"] is True
        np.testing.assert_array_equal(entry["old_values"], [[3.0, 4.0]])
        np.testing.assert_array_equal(entry["indices"], [1])

    def test_begin_journals_new_feature(self):
        fs = self._store()
        delta = FeatureDelta()
        delta.set("new_feat", np.array([[0.0]]))
        idx = _index_map(0)

        fs.begin_transaction(delta, idx)

        entry = fs._journal["new_feat"]
        assert entry["existed"] is False
        assert entry["old_values"] is None

    def test_begin_journals_dropped_feature(self):
        fs = self._store()
        delta = FeatureDelta()
        delta.delete("fp")
        idx = _index_map(0, 1)

        fs.begin_transaction(delta, idx)

        entry = fs._journal["fp"]
        assert entry["existed"] is True
        assert entry["indices"] is None  # full backup
        np.testing.assert_array_equal(entry["old_values"], fs._features["fp"])

    def test_commit_applies_update(self):
        fs = self._store()
        delta = FeatureDelta()
        delta.set("fp", np.array([[99.0, 99.0]]))
        idx = _index_map(1)

        fs.begin_transaction(delta, idx)
        fs.commit_transaction(delta, idx)

        np.testing.assert_array_equal(fs._features["fp"][1], [99.0, 99.0])
        # Journal is intentionally NOT cleared by commit_transaction; only apply_deltas
        # clears it after all three stores commit successfully.
        assert fs._journal is not None

    def test_commit_drops_feature(self):
        fs = self._store()
        delta = FeatureDelta()
        delta.delete("fp")
        idx = _index_map(0, 1, 2)

        fs.begin_transaction(delta, idx)
        fs.commit_transaction(delta, idx)

        assert "fp" not in fs._features
        assert fs._journal is not None  # cleared only by apply_deltas after full success

    def test_rollback_restores_updated_rows(self):
        fs = self._store()
        original = fs._features["fp"][1].copy()
        delta = FeatureDelta()
        delta.set("fp", np.array([[99.0, 99.0]]))
        idx = _index_map(1)

        fs.begin_transaction(delta, idx)
        # Simulate a partial write
        fs._features["fp"][1] = [99.0, 99.0]
        fs.rollback_transaction()

        np.testing.assert_array_equal(fs._features["fp"][1], original)
        assert fs._journal is None

    def test_rollback_removes_new_feature(self):
        fs = self._store()
        delta = FeatureDelta()
        delta.set("brand_new", np.array([[7.0, 8.0]]))
        idx = _index_map(0)

        fs.begin_transaction(delta, idx)
        # Simulate partial write of the new feature
        fs._features["brand_new"] = np.zeros((3, 2))
        fs.rollback_transaction()

        assert "brand_new" not in fs._features

    def test_rollback_restores_dropped_feature(self):
        fs = self._store()
        original = fs._features["fp"].copy()
        delta = FeatureDelta()
        delta.delete("fp")
        idx = _index_map(0, 1, 2)

        fs.begin_transaction(delta, idx)
        del fs._features["fp"]  # simulate drop
        fs.rollback_transaction()

        assert "fp" in fs._features
        np.testing.assert_array_equal(fs._features["fp"], original)

    def test_commit_without_begin_raises(self):
        fs = self._store()
        delta = FeatureDelta()
        with pytest.raises(RuntimeError, match="without beginning"):
            fs.commit_transaction(delta, _index_map(0))

    def test_rollback_without_begin_is_noop(self):
        fs = self._store()
        fs.rollback_transaction()  # should not raise


# ===========================================================================
# MemoryMetadataStore journal
# ===========================================================================

class TestMemoryMetadataStoreJournal:
    def _store(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
        return MemoryMetadataStore(df)

    def test_begin_journals_existing_col_rows(self):
        ms = self._store()
        delta = MetadataDelta()
        delta.ensure_local(3)
        delta.local["a"] = np.array([10.0, 20.0, 30.0])
        idx = _index_map(0, 2)

        ms.begin_transaction(delta, idx)

        entry = ms._journal["modified_cols"]["a"]
        assert entry["existed"] is True
        np.testing.assert_array_equal(entry["old_values"], [1.0, 3.0])
        np.testing.assert_array_equal(entry["indices"], [0, 2])

    def test_begin_journals_new_col(self):
        ms = self._store()
        delta = MetadataDelta()
        delta.ensure_local(3)
        delta.local["c"] = np.array([0.1, 0.2, 0.3])
        idx = _index_map(0, 1, 2)

        ms.begin_transaction(delta, idx)

        entry = ms._journal["modified_cols"]["c"]
        assert entry["existed"] is False

    def test_begin_journals_dropped_col(self):
        ms = self._store()
        delta = MetadataDelta()
        delta.deleted_cols.add("b")
        idx = _index_map(0)

        ms.begin_transaction(delta, idx)

        assert "b" in ms._journal["dropped_cols"]

    def test_commit_updates_rows(self):
        ms = self._store()
        # The delta local has N overlay rows; index_map maps them to base rows.
        # Replicate the pattern: 2-row overlay over rows [0, 2] of a 3-row base.
        delta = MetadataDelta()
        delta.ensure_local(2)
        delta.local["a"] = np.array([99.0, 88.0])  # 2 overlay rows
        idx = _index_map(0, 2)  # overlay[0]→base[0], overlay[1]→base[2]

        ms.begin_transaction(delta, idx)
        ms.commit_transaction(delta, idx)

        assert ms._metadata["a"].iloc[0] == 99.0
        assert ms._metadata["a"].iloc[2] == 88.0
        assert ms._metadata["a"].iloc[1] == 2.0  # untouched

    def test_commit_drops_col(self):
        ms = self._store()
        delta = MetadataDelta()
        delta.deleted_cols.add("b")
        idx = _index_map(0, 1, 2)

        ms.begin_transaction(delta, idx)
        ms.commit_transaction(delta, idx)

        assert "b" not in ms._metadata.columns

    def test_rollback_restores_updated_rows(self):
        ms = self._store()
        delta = MetadataDelta()
        delta.ensure_local(3)
        delta.local["a"] = np.array([99.0, 20.0, 99.0])
        idx = _index_map(0, 2)

        ms.begin_transaction(delta, idx)
        ms._metadata.iloc[0, ms._metadata.columns.get_loc("a")] = 99.0
        ms.rollback_transaction()

        assert ms._metadata["a"].iloc[0] == 1.0
        assert ms._metadata["a"].iloc[2] == 3.0

    def test_rollback_removes_new_col(self):
        ms = self._store()
        delta = MetadataDelta()
        delta.ensure_local(3)
        delta.local["c"] = np.array([0.1, 0.2, 0.3])
        idx = _index_map(0, 1, 2)

        ms.begin_transaction(delta, idx)
        ms._metadata["c"] = [0.1, 0.2, 0.3]  # simulate partial write
        ms.rollback_transaction()

        assert "c" not in ms._metadata.columns

    def test_rollback_restores_dropped_col(self):
        ms = self._store()
        delta = MetadataDelta()
        delta.deleted_cols.add("b")
        idx = _index_map(0, 1, 2)

        ms.begin_transaction(delta, idx)
        ms._metadata.drop(columns=["b"], inplace=True)  # simulate drop
        ms.rollback_transaction()

        assert "b" in ms._metadata.columns
        assert list(ms._metadata["b"]) == ["x", "y", "z"]

    def test_commit_without_begin_raises(self):
        ms = self._store()
        delta = MetadataDelta()
        with pytest.raises(RuntimeError, match="without beginning"):
            ms.commit_transaction(delta, _index_map(0))

    def test_rollback_without_begin_is_noop(self):
        ms = self._store()
        ms.rollback_transaction()  # should not raise


# ===========================================================================
# MemoryObjectStore journal
# ===========================================================================

class TestMemoryObjectStoreJournal:
    def _store(self):
        return MemoryObjectStore(["alpha", "beta", "gamma"])

    def test_begin_journals_mutated_positions(self):
        os_ = self._store()
        delta = ObjectDelta()
        delta.set(1, "BETA_NEW")  # overlay_idx=1 → base_pos=index_map[1]
        idx = _index_map(0, 1, 2)

        os_.begin_transaction(delta, idx)

        # overlay_idx=1, index_map[1]=1 → base_pos=1
        assert 1 in os_._journal
        assert os_._journal[1] == "beta"

    def test_commit_applies_update(self):
        os_ = self._store()
        delta = ObjectDelta()
        delta.set(0, "ALPHA_NEW")
        idx = _index_map(0, 1, 2)

        os_.begin_transaction(delta, idx)
        os_.commit_transaction(delta, idx)

        assert os_._objects[0] == "ALPHA_NEW"
        # Journal is intentionally NOT cleared by commit_transaction alone
        assert os_._journal is not None

    def test_rollback_restores_objects(self):
        os_ = self._store()
        delta = ObjectDelta()
        delta.set(0, "ALPHA_NEW")
        idx = _index_map(0, 1, 2)

        os_.begin_transaction(delta, idx)
        os_._objects[0] = "ALPHA_NEW"  # simulate partial write
        os_.rollback_transaction()

        assert os_._objects[0] == "alpha"
        assert os_._journal is None

    def test_commit_without_begin_raises(self):
        os_ = self._store()
        delta = ObjectDelta()
        with pytest.raises(RuntimeError, match="without beginning"):
            os_.commit_transaction(delta, _index_map(0))

    def test_rollback_without_begin_is_noop(self):
        os_ = self._store()
        os_.rollback_transaction()  # should not raise


# ===========================================================================
# CompositeStorageBackend.apply_deltas — happy path
# ===========================================================================

class TestApplyDeltasHappyPath:
    def test_apply_deltas_commits_all_domains(self):
        backend = _make_backend(3)
        idx = _index_map(1)

        obj_delta = ObjectDelta()
        obj_delta.set(0, "obj_PATCHED")  # overlay_idx=0 → base_pos=index_map[0]=1

        meta_delta = MetadataDelta()
        meta_delta.ensure_local(1)
        meta_delta.local["score"] = np.array([999.0])

        feat_delta = FeatureDelta()
        feat_delta.set("fp", np.array([[99.0, 99.0, 99.0, 99.0]]))

        backend.apply_deltas(obj_delta, meta_delta, feat_delta, idx)

        assert backend.get_objects(1) == "obj_PATCHED"
        assert backend.get_metadata(idx=idx)["score"].iloc[0] == 999.0
        np.testing.assert_array_equal(
            backend.get_feature("fp", idx=idx),
            [[99.0, 99.0, 99.0, 99.0]],
        )

    def test_apply_deltas_with_empty_deltas(self):
        backend = _make_backend(3)
        original_objs = backend.get_objects()
        original_meta = backend.get_metadata().copy()
        original_fp = backend.get_feature("fp").copy()

        backend.apply_deltas(ObjectDelta(), MetadataDelta(), FeatureDelta(), _index_map(0, 1, 2))

        assert backend.get_objects() == original_objs
        pd.testing.assert_frame_equal(backend.get_metadata(), original_meta)
        np.testing.assert_array_equal(backend.get_feature("fp"), original_fp)


# ===========================================================================
# POISON PILL TEST — mid-commit failure triggers full rollback
# ===========================================================================

class TestPoisonPillRollback:
    """
    Prove that a failure during the commit phase leaves the backend in exactly
    the same state it was in before apply_deltas() was called.
    """

    def test_feature_commit_failure_rolls_back_objects_and_metadata(self):
        """
        Scenario
        --------
        1. A valid ObjectDelta and MetadataDelta are provided.
        2. The FeatureDelta's commit_transaction is monkeypatched to raise.
        3. Assert that apply_deltas raises RuntimeError.
        4. Assert that objects and metadata are unchanged (rollback worked).
        """
        backend = _make_backend(3)
        idx = _index_map(1)

        # Snapshot state before apply_deltas
        original_obj_1 = backend.get_objects(1)
        original_score_1 = backend.get_metadata(idx=idx)["score"].iloc[0]
        original_fp_1 = backend.get_feature("fp", idx=idx).copy()

        obj_delta = ObjectDelta()
        obj_delta.set(0, "obj_PATCHED")  # overlay_idx=0, base_pos=index_map[0]=1

        meta_delta = MetadataDelta()
        meta_delta.ensure_local(1)
        meta_delta.local["score"] = np.array([999.0])

        feat_delta = FeatureDelta()
        feat_delta.set("fp", np.array([[99.0, 99.0, 99.0, 99.0]]))

        # Poison the feature store's commit_transaction
        original_commit = backend._feature_store.commit_transaction

        def poisoned_commit(delta, index_map):
            raise ValueError("💀 Simulated feature commit failure (poison pill)")

        with patch.object(backend._feature_store, "commit_transaction", poisoned_commit):
            with pytest.raises(RuntimeError, match="rolled back"):
                backend.apply_deltas(obj_delta, meta_delta, feat_delta, idx)

        # Critical assertions: state must be identical to pre-commit snapshot
        assert backend.get_objects(1) == original_obj_1, (
            f"Object at base_pos=1 was not rolled back! "
            f"Got: {backend.get_objects(1)!r}, expected: {original_obj_1!r}"
        )
        assert backend.get_metadata(idx=idx)["score"].iloc[0] == original_score_1, (
            f"Metadata score at row=1 was not rolled back! "
            f"Got: {backend.get_metadata(idx=idx)['score'].iloc[0]}, "
            f"expected: {original_score_1}"
        )
        np.testing.assert_array_equal(
            backend.get_feature("fp", idx=idx),
            original_fp_1,
            err_msg="Feature fp at row=1 was not rolled back!",
        )

    def test_metadata_commit_failure_rolls_back_objects(self):
        """
        Ensure that if the metadata commit fails (second in order), the
        already-committed object mutation is also rolled back.
        """
        backend = _make_backend(3)
        idx = _index_map(2)

        original_obj_2 = backend.get_objects(2)
        original_score_2 = backend.get_metadata(idx=idx)["score"].iloc[0]

        obj_delta = ObjectDelta()
        obj_delta.set(0, "obj_2_PATCHED")  # overlay_idx=0, base_pos=2

        meta_delta = MetadataDelta()
        meta_delta.ensure_local(1)
        meta_delta.local["score"] = np.array([777.0])

        feat_delta = FeatureDelta()

        def poisoned_meta_commit(delta, index_map):
            raise ValueError("💀 Simulated metadata commit failure")

        with patch.object(backend._metadata_store, "commit_transaction", poisoned_meta_commit):
            with pytest.raises(RuntimeError, match="rolled back"):
                backend.apply_deltas(obj_delta, meta_delta, feat_delta, idx)

        # Object must be rolled back even though its commit succeeded first
        assert backend.get_objects(2) == original_obj_2, (
            f"Object at base_pos=2 was not rolled back after metadata failure! "
            f"Got: {backend.get_objects(2)!r}, expected: {original_obj_2!r}"
        )
        assert backend.get_metadata(idx=idx)["score"].iloc[0] == original_score_2

    def test_object_commit_failure_leaves_all_unchanged(self):
        """
        If the very first commit (objects) fails, metadata and features must
        also be completely untouched.
        """
        backend = _make_backend(3)
        idx = _index_map(0)

        original_obj_0 = backend.get_objects(0)
        original_label_0 = backend.get_metadata(idx=idx)["label"].iloc[0]
        original_fp_0 = backend.get_feature("fp", idx=idx).copy()

        obj_delta = ObjectDelta()
        obj_delta.set(0, "obj_0_PATCHED")

        meta_delta = MetadataDelta()
        meta_delta.ensure_local(1)
        meta_delta.local["label"] = np.array(["PATCHED"])

        feat_delta = FeatureDelta()
        feat_delta.set("fp", np.array([[11.0, 22.0, 33.0, 44.0]]))

        def poisoned_obj_commit(delta, index_map):
            raise ValueError("💀 Simulated object commit failure")

        with patch.object(backend._object_store, "commit_transaction", poisoned_obj_commit):
            with pytest.raises(RuntimeError, match="rolled back"):
                backend.apply_deltas(obj_delta, meta_delta, feat_delta, idx)

        assert backend.get_objects(0) == original_obj_0
        assert backend.get_metadata(idx=idx)["label"].iloc[0] == original_label_0
        np.testing.assert_array_equal(backend.get_feature("fp", idx=idx), original_fp_0)

    def test_rollback_error_message_preserves_original_cause(self):
        """The RuntimeError wrapping must chain to the original exception."""
        backend = _make_backend(2)
        idx = _index_map(0)

        def poisoned(delta, index_map):
            raise ValueError("root cause sentinel")

        with patch.object(backend._feature_store, "commit_transaction", poisoned):
            with pytest.raises(RuntimeError) as exc_info:
                backend.apply_deltas(ObjectDelta(), MetadataDelta(), FeatureDelta(), idx)

        assert exc_info.value.__cause__ is not None
        assert "root cause sentinel" in str(exc_info.value.__cause__)

    def test_new_feature_rolled_back_on_failure(self):
        """
        A brand-new feature (not present in the base before apply_deltas)
        must be removed on rollback, leaving no trace.
        """
        backend = _make_backend(2)
        idx = _index_map(0, 1)

        assert "brand_new" not in backend.get_feature_names()

        feat_delta = FeatureDelta()
        feat_delta.set("brand_new", np.array([[1.0, 2.0], [3.0, 4.0]]))

        # Poison metadata so the rollback is triggered after features are committed
        # We need to poison *after* feature commit succeeds, so poison metadata.
        original_meta_commit = backend._metadata_store.commit_transaction

        def poisoned_meta(delta, index_map):
            raise ValueError("💀 Simulated post-feature failure")

        with patch.object(backend._metadata_store, "commit_transaction", poisoned_meta):
            with pytest.raises(RuntimeError, match="rolled back"):
                backend.apply_deltas(ObjectDelta(), MetadataDelta(), feat_delta, idx)

        assert "brand_new" not in backend.get_feature_names(), (
            "Brand-new feature 'brand_new' was not cleaned up during rollback!"
        )

    def test_dropped_feature_restored_on_failure(self):
        """
        A feature that existed before and was dropped in the delta must be
        fully restored if the commit fails downstream.
        """
        backend = _make_backend(2)
        original_fp = backend.get_feature("fp").copy()
        idx = _index_map(0, 1)

        feat_delta = FeatureDelta()
        feat_delta.delete("fp")

        def poisoned_meta(delta, index_map):
            raise ValueError("💀 Simulated post-feature failure")

        with patch.object(backend._metadata_store, "commit_transaction", poisoned_meta):
            with pytest.raises(RuntimeError, match="rolled back"):
                backend.apply_deltas(ObjectDelta(), MetadataDelta(), feat_delta, idx)

        assert "fp" in backend.get_feature_names(), (
            "Dropped feature 'fp' was not restored during rollback!"
        )
        np.testing.assert_array_equal(backend.get_feature("fp"), original_fp)


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_apply_deltas_single_row_backend(self):
        backend = EagerMemoryBackend(
            objects=["solo"],
            metadata=pd.DataFrame({"v": [42.0]}),
            features={"f": np.array([[1.0]])},
        )
        idx = _index_map(0)

        feat_delta = FeatureDelta()
        feat_delta.set("f", np.array([[7.0]]))

        backend.apply_deltas(ObjectDelta(), MetadataDelta(), feat_delta, idx)

        np.testing.assert_array_equal(backend.get_feature("f"), [[7.0]])

    def test_apply_deltas_multi_feature_partial_update(self):
        """Updating one of two features leaves the other intact."""
        backend = EagerMemoryBackend(
            objects=["a", "b"],
            metadata=pd.DataFrame({"x": [1, 2]}),
            features={
                "fp1": np.array([[1.0, 2.0], [3.0, 4.0]]),
                "fp2": np.array([[10.0], [20.0]]),
            },
        )
        idx = _index_map(0)

        feat_delta = FeatureDelta()
        feat_delta.set("fp1", np.array([[99.0, 99.0]]))  # only fp1

        backend.apply_deltas(ObjectDelta(), MetadataDelta(), feat_delta, idx)

        np.testing.assert_array_equal(backend.get_feature("fp1", idx=idx), [[99.0, 99.0]])
        np.testing.assert_array_equal(backend.get_feature("fp2", idx=idx), [[10.0]])  # untouched

    def test_journal_is_none_after_successful_commit(self):
        backend = _make_backend(2)
        idx = _index_map(0)

        feat_delta = FeatureDelta()
        feat_delta.set("fp", np.array([[55.0, 55.0, 55.0, 55.0]]))

        backend.apply_deltas(ObjectDelta(), MetadataDelta(), feat_delta, idx)

        # apply_deltas explicitly clears all journals after ALL three stores succeed
        assert backend._feature_store._journal is None
        assert backend._metadata_store._journal is None
        assert backend._object_store._journal is None

    def test_journal_is_none_after_rollback(self):
        backend = _make_backend(2)
        idx = _index_map(0)

        feat_delta = FeatureDelta()
        feat_delta.set("fp", np.array([[55.0, 55.0, 55.0, 55.0]]))

        def poisoned(delta, index_map):
            raise ValueError("poison")

        with patch.object(backend._feature_store, "commit_transaction", poisoned):
            with pytest.raises(RuntimeError):
                backend.apply_deltas(ObjectDelta(), MetadataDelta(), feat_delta, idx)

        # All journals should be cleared after rollback too
        assert backend._feature_store._journal is None
        assert backend._metadata_store._journal is None
        assert backend._object_store._journal is None