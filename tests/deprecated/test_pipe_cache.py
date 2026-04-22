import pytest

from druglab.pipe.cache import DictCache

class TestDictCacheLRU:
    """
    REGRESSION: Before the fix, DictCache.get() never moved the accessed
    entry to the 'most-recently-used' position.  This meant eviction was
    effectively FIFO regardless of access patterns — a cache hit on an old
    entry would not protect it from eviction.
    """
 
    def test_get_promotes_entry_to_mru(self):
        """
        After accessing key 'a', it should be the last to be evicted.
 
        Old behaviour: FIFO — 'a' (inserted first) is evicted when 'd' is added.
        New behaviour: LRU  — 'a' is promoted by get(), so 'b' is evicted instead.
        """
        cache = DictCache(max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
 
        # Access 'a' — should protect it from being the next victim
        _ = cache.get("a")
 
        # Adding 'd' must evict the LRU entry, which is now 'b' (not 'a')
        cache.set("d", 4)
 
        # REGRESSION: old code would evict 'a' here, returning None
        assert cache.get("a") == 1, (
            "SAFETY-01 regression: get() must promote 'a' so it is not "
            "evicted before 'b'."
        )
        assert cache.get("b") is None, (
            "SAFETY-01: 'b' should have been evicted as the LRU entry."
        )
 
    def test_get_on_missing_key_returns_none(self):
        """get() on a non-existent key must return None, not raise."""
        cache = DictCache(max_size=5)
        assert cache.get("nonexistent") is None
 
    def test_set_updates_recency_on_overwrite(self):
        """Re-setting an existing key should also mark it as most-recently-used."""
        cache = DictCache(max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
 
        # Re-set 'a' — should move it to MRU
        cache.set("a", 99)
 
        # Adding 'd' should now evict 'b' (oldest after 'a' was refreshed)
        cache.set("d", 4)
 
        assert cache.get("a") == 99
        assert cache.get("b") is None
 
    def test_max_size_enforced(self):
        cache = DictCache(max_size=10)
        for i in range(25):
            cache.set(f"k{i}", i)
        assert len(cache._store) <= 10
 
    def test_max_size_zero_raises(self):
        with pytest.raises(ValueError):
            DictCache(max_size=0)
 
    def test_clear_empties_store(self):
        cache = DictCache(max_size=5)
        for i in range(5):
            cache.set(f"k{i}", i)
        cache.clear()
        assert len(cache._store) == 0