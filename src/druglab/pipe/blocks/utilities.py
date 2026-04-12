from typing import Optional, List
import random

from druglab.db import BaseTable, MoleculeTable
from druglab.io import BatchReader, EagerReader
from druglab.pipe.base import BaseBlock
from druglab.pipe.archetypes import IOBlock

# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

class MemoryIOBlock(IOBlock):
    """
    A testing/utility block that doesn't read from disk, but yields chunks 
    from an already loaded BaseTable. Perfect for triggering pipeline batch mode.
    """
    
    def __init__(self, table: BaseTable, batch_size: int = 1000, **kwargs):
        super().__init__(batch_size=batch_size, **kwargs)
        self.table = table
        
    def yield_batches(self):
        n = len(self.table)
        for i in range(0, n, self.batch_size):
            yield self.table[i : i + self.batch_size]
            
    def _load_table(self):
        return self.table
    
class MoleculeFileReaderBlock(IOBlock):
    """
    Reads molecule files given as a list of paths
    """

    def __init__(self, paths: List[str], batch_size: Optional[int] = None, **kwargs):
        super().__init__(batch_size=batch_size, **kwargs)
        self.paths = paths

    def yield_batches(self):
        if self.batch_size is None or self.batch_size < 1:
            raise ValueError("MoleculeFileReaderBlock.yield_batches requires a positive batch_size.")
    
        reader = BatchReader(self.paths, batch_size=self.batch_size)
        for records in reader:
            table = MoleculeTable.from_records(records)
            yield table
    
    def _load_table(self):
        reader = EagerReader(self.paths)
        records = reader.read()
        return MoleculeTable.from_records(records)
            
# ---------------------------------------------------------------------------
# Sampling / Utilities
# ---------------------------------------------------------------------------
    
class SamplerBlock(BaseBlock):
    """
    Randomly subsamples the table to at most ``max_size`` rows.
 
    Useful for debugging pipelines on large datasets or creating
    reproducible mini-batches for quick experiments.
 
    The block is a no-op when the table already has <= ``max_size`` rows.
 
    Parameters
    ----------
    max_size : int
        Maximum number of rows to keep.
    random_seed : int or None
        Seed for the random number generator.  Pass *None* for a
        non-deterministic sample.  Default None.
    """
 
    def __init__(
        self,
        max_size: int,
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = max_size
        self.random_seed = random_seed
 
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_size": self.max_size, "random_seed": self.random_seed})
        return cfg
 
    def _process(self, table: Optional[BaseTable]) -> BaseTable:
        if table is None:
            raise ValueError("SamplerBlock requires an input table.")
 
        n = len(table)
        if n <= self.max_size:
            return table  # nothing to do
 
        rng = random.Random(self.random_seed)
        indices = sorted(rng.sample(range(n), self.max_size))
 
        sampled = table.subset(indices, copy_objects=False)
        return sampled