from __future__ import annotations

from typing import Optional, List

import pandas as pd

def try_numerize_df(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
    """
    Attempt to numerize columns in a DataFrame.

    Parameters
    ----------
    columns : Optional[List[str]], default None
        The column(s) to numerize. If None, numerize all columns.
    """
    if columns is None:
        columns = df.columns.tolist()
    else:
        if isinstance(columns, str):
            columns = [columns]

    for col in columns:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(self._metadata[col])
            except (ValueError, TypeError):
                pass