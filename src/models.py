from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
        logger.info("spaCy model en_core_web_sm not loaded; entity extraction disabled")
except Exception:
    spacy = None
    nlp = None


def load_transformer_classifier(model_name: str = "nlpaueb/legal-bert-base-uncased") -> Optional[object]:
    """Load an optional classifier lazily; never make extraction depend on it."""
    try:
        from transformers import pipeline
        return pipeline("text-classification", model=model_name, truncation=True)
    except Exception as exc:
        logger.warning("Transformer classifier unavailable: %s", exc)
        return None
