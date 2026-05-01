from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

class BaseMetadataStore(ABC):
    @abstractmethod
    def get_metadata(self, idx=None, cols: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def add_metadata_column(self, name: str, value, idx=None, na: Any = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_metadata(self, values, idx=None) -> None:
        raise NotImplementedError

    @abstractmethod
    def drop_metadata_columns(self, cols=None) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_metadata_columns(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def n_rows(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def append(self, data: Any) -> None:
        """Append net-new rows to the end of the store."""
        raise NotImplementedError

    @abstractmethod
    def gather_materialized_state(self, index_map: Optional[np.ndarray] = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path, **kwargs) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Path, **kwargs) -> "BaseMetadataStore":
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Transaction protocol
    # ------------------------------------------------------------------

    def begin_transaction(self, delta: Any, index_map: np.ndarray) -> None:
        """Prepare for a commit: validate the delta and snapshot rows to journal."""
        raise NotImplementedError

    def commit_transaction(self, delta: Any, index_map: np.ndarray) -> None:
        """Apply the delta to the underlying storage."""
        raise NotImplementedError

    def rollback_transaction(self) -> None:
        """Revert any changes made during the transaction using the rollback journal."""
        raise NotImplementedError