from druglab.db import BaseTable
from druglab.pipe.archetypes import IOBlock

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