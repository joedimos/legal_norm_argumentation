from __future__ import annotations

import logging
from typing import Optional, Tuple

import pandas as pd

try:
    from .originality import LogicBasedLegalReasoner
    from .utils import flatten_text_columns
except ImportError:  # pragma: no cover
    from originality import LogicBasedLegalReasoner
    from utils import flatten_text_columns

logger = logging.getLogger(__name__)
PREFERRED_TEXT_COLS = ["text", "decision_text", "content", "case_body", "full_text"]


def load_dataset(csv_path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="warn")
        text_col = next((column for column in PREFERRED_TEXT_COLS if column in df.columns), None)
        if text_col is None:
            candidates = [column for column in df.columns if "text" in column.lower() or "body" in column.lower()]
            if not candidates:
                return None, None
            df = flatten_text_columns(df, candidates)
            text_col = "_combined_text"
        return df, text_col
    except Exception as exc:
        logger.exception("Failed to load dataset: %s", exc)
        return None, None


def run_pipeline(csv_path: str, out_csv: str = "results_summary.csv", max_cases: Optional[int] = 100) -> pd.DataFrame:
    df, text_col = load_dataset(csv_path)
    if df is None or text_col is None:
        raise ValueError("Dataset load failed or no text-like column was found")

    count = min(max_cases, len(df)) if max_cases is not None else len(df)
    results = []
    for row_index in range(count):
        text = str(df.iloc[row_index][text_col]) if pd.notna(df.iloc[row_index][text_col]) else ""
        if not text.strip():
            results.append({"row": row_index, "status": "empty"})
            continue
        analysis = LogicBasedLegalReasoner().analyze_legal_case(text[:20000], run_pyreason=False)
        results.append(
            {
                "row": row_index,
                "status": "ok",
                "arguments_count": len(analysis["arguments"]),
                "attacks_count": len(analysis["attacks"]),
                "preferred_extensions": len(analysis["aaf"]["preferred"]),
                "stable_extensions": len(analysis["aaf"]["stable"]),
            }
        )
    output = pd.DataFrame(results)
    output.to_csv(out_csv, index=False)
    return output
