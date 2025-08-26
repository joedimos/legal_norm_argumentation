import logging
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("legal-norm-extraction")


def flatten_text_columns(df, text_cols):
    """Concatenate multiple potential text columns into a single string column."""
    import pandas as pd
    df = df.copy()
    df["_combined_text"] = df[text_cols].fillna("").agg(" ".join, axis=1)
    return df
