"""
druglab.pipe.pipeline
~~~~~~~~~~~~~~~~~~~~~
Pipeline orchestrator that manages block execution and batch boundaries.
"""

from typing import List, Optional, Iterator, TypeVar
from druglab.db.table import BaseTable
from druglab.pipe.base import BaseBlock

TableT = TypeVar("T", bound="BaseTable")

class Pipeline:
    """
    Orchestrates a sequence of BaseBlocks. 
    Handles transparent chunking and concatenation when a block declares a batch_size.
    """

    def __init__(self, steps: List[BaseBlock]):
        self.steps = steps

    def run(self, table: Optional[TableT] = None) -> TableT:
        """Run the pipeline on an initial table (or None if starting with an IO block)."""
        return self._run_steps(self.steps, table)

    def _run_steps(self, steps: List[BaseBlock], table: Optional[TableT]) -> TableT:
        if not steps:
            return table

        current_step = steps[0]
        remaining_steps = steps[1:]

        # --- BATCH BOUNDARY DETECTED ---
        if current_step.batch_size is not None and current_step.batch_size > 0:
            processed_batches = []
            
            if table is None:
                # Scenario A: It's an IO Block generating data in batches
                for out_batch in current_step.yield_batches():
                    # Pass the generated batch down the rest of the pipeline
                    final_batch = self._run_steps(remaining_steps, out_batch)
                    if final_batch is not None and len(final_batch) > 0:
                        processed_batches.append(final_batch)
            else:
                # Scenario B: It's a normal block forcing a chunking of the current table
                for chunk in self._chunk_table(table, current_step.batch_size):
                    out_batch = current_step.run(chunk)
                    # Pass the chunked batch down the rest of the pipeline
                    final_batch = self._run_steps(remaining_steps, out_batch)
                    if final_batch is not None and len(final_batch) > 0:
                        processed_batches.append(final_batch)

            # Recombine the universe
            if not processed_batches:
                return table.__class__() if table is not None else None
                
            return processed_batches[0].__class__.concat(processed_batches)

        # --- NORMAL EXECUTION ---
        else:
            out_table = current_step.run(table)
            return self._run_steps(remaining_steps, out_table)

    def _chunk_table(self, table: TableT, batch_size: int) -> Iterator[TableT]:
        """Helper to yield slices of a BaseTable."""
        n = len(table)
        for i in range(0, n, batch_size):
            yield table[i : i + batch_size]