from typing import List, Dict, Tuple, Optional
import re
import logging
from .models import nlp, load_transformer_classifier

logger = logging.getLogger(__name__)

DEONTIC_KEYWORDS = {
    "obligation": ["shall", "must", "is required to", "required to"],
    "permission": ["may", "is permitted to", "is allowed to"],
    "prohibition": ["shall not", "must not", "is prohibited", "is forbidden"]
}

# Simple rule-based extractor
def rule_based_norms(text: str) -> List[Dict]:
    doc_text = text.replace("\n", " ")
    sentences = re.split(r'(?<=[\.\;\:])\s+', doc_text)
    norms = []
    for s in sentences:
        s_lower = s.lower()
        for modality, kws in DEONTIC_KEYWORDS.items():
            for kw in kws:
                if kw in s_lower:
                    # naive agent extraction: up to first verb phrase (improve with SRL)
                    agent = extract_agent_simple(s)
                    action = extract_action_simple(s, kw)
                    norms.append({
                        "sentence": s.strip(),
                        "modality": modality,
                        "trigger": kw,
                        "agent": agent,
                        "action": action,
                    })
                    break
    return norms

def extract_agent_simple(sentence: str) -> str:
    # heuristic: look for "X shall/may/..." pattern => agent is the phrase before keyword
    m = re.search(r'(.{1,80}?)\b(shall|must|may|is required to|is permitted to|shall not|must not)\b', sentence, flags=re.I)
    if m:
        return m.group(1).strip(",;: ")
    # fallback: return empty
    return ""

def extract_action_simple(sentence: str, trigger: str) -> str:
    # action candidate is remainder after trigger
    idx = sentence.lower().find(trigger)
    if idx >= 0:
        return sentence[idx + len(trigger):].strip()
    return ""

# Optional transformer-based classifier wrapper
_transformer = None
def classify_sentence_modality(sentence: str) -> Optional[Dict]:
    global _transformer
    if _transformer is None:
        _transformer = load_transformer_classifier()
    if _transformer is None:
        return None
    try:
        res = _transformer(sentence, top_k=3)
        return res
    except Exception as e:
        logger.error("Transformer classification failed: %s", e)
        return None

# High-level API
def extract_norms(text: str, use_transformer: bool = False) -> List[Dict]:
    norms = rule_based_norms(text)
    if use_transformer:
        for n in norms:
            try:
                cls = classify_sentence_modality(n["sentence"])
                n["transformer"] = cls
            except Exception:
                n["transformer"] = None
    return norms
