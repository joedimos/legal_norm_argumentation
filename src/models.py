from typing import Optional
import spacy
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
  
    nlp = None
    logger.warning("spaCy model 'en_core_web_sm' not loaded. Run `python -m spacy download en_core_web_sm`.")

def load_transformer_classifier(model_name: str = "nlpaueb/legal-bert-base-uncased") -> Optional[object]:
    """
    Returns a transformers pipeline or model object for norm classification.
    For now this is a placeholder to be swapped with a fine-tuned model.
    """
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        pipe = pipeline("text-classification", model=model_name, truncation=True)
        return pipe
    except Exception as e:
        logger.warning("Transformer classifier not available: %s", e)
        return None
