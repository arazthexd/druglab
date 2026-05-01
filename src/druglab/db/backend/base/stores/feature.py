from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

class BaseFeatureStore(ABC):
    @abstractmethod
    def get_feature(self, name: str, idx=None) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def update_feature(self, name: str, array: np.ndarray, idx=None, na: Any = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def drop_feature(self, name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_feature_shape(self, name: str) -> tuple:
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
    def load(cls, path: Path, **kwargs) -> "BaseFeatureStore":
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