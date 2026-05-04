from typing import Optional, Dict, List
from typing_extensions import Self
import numpy as np
import pandas as pd

from ..types import IdxLike
from .base import BaseEngine

class PandasEngine(BaseEngine[pd.DataFrame]):
    """
    In-memory DataFrame engine. Perfect for testing, small datasets, 
    or fully RAM-bound workflows.
    """
    
    def __init__(
        self, 
        _store: Optional[Dict[str, pd.DataFrame]] = None, 
        _masks: Optional[Dict[str, IdxLike]] = None,
        _is_view: bool = False
    ):
        # 1. State Sharing
        # If _store is passed, this is a View! It shares the parent's memory dict.
        self._store = _store if _store is not None else {}
        self._masks = _masks if _masks is not None else {}
        self._is_view = _is_view

    def _get_table_name(self, namespace: str, what: str) -> str:
        return f"{namespace}_{what}"

    def materialize(
        self, 
        namespace: str, 
        what: str, 
        rows: Optional[IdxLike] = None, 
        *args, 
        **kwargs
    ) -> pd.DataFrame:
        table_name = self._get_table_name(namespace, what)

        if table_name not in self._store:
            raise KeyError(f"Table '{table_name}' does not exist in the Pandas engine.")
            
        df = self._store[table_name]

        # Get current mask for the given namespace
        current_mask = self._masks.get(namespace)
        
        # Combine view masks
        if rows is not None:
            active_mask = self._combine_masks(current_mask, rows)
        else:
            active_mask = current_mask
        
        if active_mask is None:
            return df.copy()
            
        # --- NATIVE PANDAS PUSHDOWN ---
        if isinstance(active_mask, slice):
            return df.iloc[active_mask].copy()
            
        elif isinstance(active_mask, (list, np.ndarray)):
            # Pandas .iloc natively supports list/array indexing
            return df.iloc[active_mask].copy()
            
        raise ValueError(f"Unsupported row mask type: {type(active_mask)}")

    def spawn_view(self, namespace: str, rows: IdxLike) -> Self:
        new_masks = self._masks.copy()
        current_mask = new_masks.get(namespace)
        new_masks[namespace] = self._combine_masks(current_mask, rows)
        return self.__class__(
            _store=self._store,
            _masks=new_masks,
            _is_view=True
        )

    def write(self, namespace: str, what: str, data: pd.DataFrame, **kwargs) -> None:
        if self._is_view:
            raise PermissionError("Cannot write data through a restricted view. Write to the root engine.")
            
        if data is None or data.empty:
            return
            
        table_name = self._get_table_name(namespace, what)
        
        if table_name not in self._store:
            # CREATE PATH
            self._store[table_name] = data.copy()
        else:
            # APPEND PATH
            self._store[table_name] = pd.concat(
                [self._store[table_name], data], 
                ignore_index=True
            )

    def export(self, target: Optional[str] = None, namespaces: Optional[List[str]] = None) -> Self:
        """
        Exports the current in-memory view to a brand new isolated in-memory engine.
        Target is mostly symbolic here since it's RAM-bound, but maintains the API.
        """
        new_engine = PandasEngine()
        
        for tbl_name in self._store.keys():
            try:
                namespace, what = tbl_name.split("_", 1)
            except ValueError:
                continue

            # Skip if we are filtering for a specific namespace!
            if namespaces and namespace not in namespaces:
                continue
                
            df_subset = self.materialize(namespace, what)
            
            if "_row_id" in df_subset.columns:
                df_subset["_row_id"] = np.arange(len(df_subset))
                
            new_engine.write(namespace, what, df_subset)
            
        return new_engine

    def _combine_masks(self, current_mask: Optional[IdxLike], new_mask: IdxLike) -> IdxLike:
        if current_mask is None:
            return new_mask
        curr_arr = np.array(current_mask) if not isinstance(current_mask, slice) else np.arange(current_mask.stop)[current_mask]
        if isinstance(new_mask, slice):
            return curr_arr[new_mask]
        else:
            return curr_arr[np.array(new_mask)]