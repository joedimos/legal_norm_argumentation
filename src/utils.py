from __future__ import annotations

from typing import Iterable
import pandas as pd


def flatten_text_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    cols = list(columns)
    result["_combined_text"] = result[cols].fillna("").astype(str).agg("\n".join, axis=1)
    return result
