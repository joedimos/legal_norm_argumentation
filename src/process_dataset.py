import os
import pandas as pd
import logging
from .utils import flatten_text_columns
from .originality import LogicBasedLegalReasoner
from typing import Optional

logger = logging.getLogger(__name__)

PREFERRED_TEXT_COLS = ['text', 'decision_text', 'content', 'case_body', 'full_text']

def load_dataset(csv_path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='warn')
        text_col = next((c for c in PREFERRED_TEXT_COLS if c in df.columns), None)
        if text_col is None:
            # try to merge common text columns if present
            candidates = [c for c in df.columns if 'text' in c.lower() or 'body' in c.lower()]
            if not candidates:
                logger.error("No text-like column found in CSV.")
                return None
            df = flatten_text_columns(df, candidates)
            text_col = "_combined_text"
        logger.info("Loaded dataset with shape: %s", df.shape)
        return df, text_col
    except Exception as e:
        logger.exception("Failed to load dataset: %s", e)
        return None, None

def run_pipeline(csv_path: str, out_csv: str = "results_summary.csv", max_cases: Optional[int] = 100):
    loader = load_dataset(csv_path)
    if loader is None:
        logger.error("Dataset load failed.")
        return
    df, text_col = loader
    n = min(max_cases, len(df)) if max_cases else len(df)
    reasoner = LogicBasedLegalReasoner()
    results = []
    for i in range(n):
        case = df.iloc[i]
        text = str(case[text_col]) if pd.notna(case[text_col]) else ""
        if not text.strip():
            logger.warning("Empty text at row %d; skipping", i)
            results.append({"row": i, "status": "empty"})
            continue
        logger.info("Processing row %d", i)
        try:
            analysis = reasoner.analyze_legal_case(text[:20000])  # limit length for speed
            results.append({
                "row": i,
                "arguments_count": len(analysis["arguments"]),
                "attacks_count": len(analysis["attacks"]),
                "preferred_extensions": len(analysis["aaf"].get("preferred", [])),
                "stable_extensions": len(analysis["aaf"].get("stable", []))
            })
        except Exception as e:
            logger.exception("Error processing row %d: %s", i, e)
            results.append({"row": i, "status": "error", "error": str(e)})
    # save
    outdf = pd.DataFrame(results)
    outdf.to_csv(out_csv, index=False)
    logger.info("Saved summary to %s", out_csv)
