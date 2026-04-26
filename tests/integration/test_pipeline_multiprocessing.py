"""Integration coverage for ItemBlock scatter-gather multiprocessing."""

from __future__ import annotations

import json
from typing import Any, List

import numpy as np
import pandas as pd

from druglab.db.backend import EagerMemoryBackend
from druglab.db.table.base import BaseTable
from druglab.pipe.base import ItemBlock


class DummyTable(BaseTable[dict]):
    def _serialize_object(self, obj: dict) -> bytes:
        return json.dumps(obj).encode("utf-8")

    @classmethod
    def _deserialize_object(cls, raw: bytes) -> dict:
        return json.loads(raw.decode("utf-8"))

    def _object_type_name(self) -> str:
        return "dict"


class ParallelMutationBlock(ItemBlock):
    required_features = ["seed_feature"]
    required_metadata = ["seed_meta"]

    def _process_item(self, item: dict) -> Any:
        value = int(item["value"])
        return value * 2, f"v{value}"

    def _apply_results(self, table: BaseTable, results: List[Any]) -> BaseTable:
        doubled = np.asarray([x[0] for x in results], dtype=np.int64).reshape(-1, 1)
        labels = pd.Series([x[1] for x in results], name="worker_tag")

        table.add_metadata_column("worker_tag", labels)
        table.update_feature("double_feature", doubled)
        return table


def _make_table(n_rows: int = 48) -> DummyTable:
    values = list(range(n_rows))
    return DummyTable(
        objects=[{"value": i} for i in values],
        metadata=pd.DataFrame({"seed_meta": values}),
        features={"seed_feature": np.asarray(values, dtype=np.int64).reshape(n_rows, 1)},
    )


def test_itemblock_scatter_gather_multiprocessing_commits_worker_deltas() -> None:
    table = _make_table()
    block = ParallelMutationBlock(n_workers=4, copy_table=False)

    out = block.run(table)

    assert out is table
    assert isinstance(out.backend, EagerMemoryBackend)
    assert out.n == 48
    assert len(out.get_objects()) == 48

    expected_double = np.arange(48, dtype=np.int64).reshape(48, 1) * 2
    np.testing.assert_array_equal(out.get_feature("double_feature"), expected_double)

    expected_labels = [f"v{i}" for i in range(48)]
    assert out.get_metadata(cols=["worker_tag"])["worker_tag"].tolist() == expected_labels