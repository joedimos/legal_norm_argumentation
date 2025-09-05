from typing import List, Dict, Optional
import re
import logging
from .models import nlp, load_transformer_classifier
from pyreason import PyReason, Fact, Rule, Query

logger = logging.getLogger(__name__)

# -------------------------
# Keywords
# -------------------------
DEONTIC_KEYWORDS = {
    "obligation": ["shall", "must", "is required to", "required to"],
    "permission": ["may", "is permitted to", "is allowed to"],
    "prohibition": ["shall not", "must not", "is prohibited", "is forbidden"]
}


# -------------------------
# Rule-based norm extractor
# -------------------------
def rule_based_norms(text: str) -> List[Dict]:
    doc_text = text.replace("\n", " ")
    sentences = re.split(r'(?<=[\.\;\:])\s+', doc_text)
    norms = []
    for s in sentences:
        s_lower = s.lower()
        for modality, kws in DEONTIC_KEYWORDS.items():
            for kw in kws:
                if kw in s_lower:
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
    m = re.search(
        r'(.{1,80}?)\b(shall|must|may|is required to|is permitted to|shall not|must not)\b',
        sentence,
        flags=re.I,
    )
    if m:
        return m.group(1).strip(",;: ")
    return ""


def extract_action_simple(sentence: str, trigger: str) -> str:
    idx = sentence.lower().find(trigger)
    if idx >= 0:
        return sentence[idx + len(trigger):].strip()
    return ""


# -------------------------
# Optional transformer-based classifier
# -------------------------
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


# -------------------------
# PyReason Bridge
# -------------------------
class PyReasonNormBridge:
    """
    Converts extracted norms into PyReason facts/rules and evaluates them.
    """

    def __init__(self):
        self.engine = PyReason()
        self.facts: List[Fact] = []
        self.rules: List[Rule] = []

    def add_norms(self, norms: List[Dict]) -> None:
        """
        Add norms as PyReason facts, using modality/agent/action encoding.
        Example predicate:
          obligation(agent_X, action_Y).
        """
        for n in norms:
            agent = n["agent"].replace(" ", "_").lower() or "unspecified_agent"
            action = n["action"].replace(" ", "_").lower() or "unspecified_action"
            modality = n["modality"]

            predicate = f"{modality}({agent},{action})"
            fact = Fact(predicate=predicate, truth_value=1.0, meta=n)
            self.facts.append(fact)
            self.engine.add_fact(fact)

    def add_conflict_rules(self) -> None:
        """
        Add some simple deontic logic consistency rules:
        - prohibition(agent, action) -> not permission(agent, action)
        - obligation(agent, action) -> not prohibition(agent, action)
        """
        self.rules.append(Rule("prohibition(A, X) -> ~permission(A, X)"))
        self.rules.append(Rule("obligation(A, X) -> ~prohibition(A, X)"))

        for r in self.rules:
            self.engine.add_rule(r)

    def query(self, predicate: str) -> float:
        q = Query(predicate)
        return self.engine.evaluate_query(q)


# -------------------------
# High-level API
# -------------------------
def extract_norms(text: str, use_transformer: bool = False, use_pyreason: bool = True) -> List[Dict]:
    norms = rule_based_norms(text)

    # Optionally enrich with transformer classification
    if use_transformer:
        for n in norms:
            try:
                cls = classify_sentence_modality(n["sentence"])
                n["transformer"] = cls
            except Exception:
                n["transformer"] = None

    # Optionally integrate with PyReason
    if use_pyreason and norms:
        bridge = PyReasonNormBridge()
        bridge.add_norms(norms)
        bridge.add_conflict_rules()

        # Run queries for each norm type
        for n in norms:
            agent = n["agent"].replace(" ", "_").lower() or "unspecified_agent"
            action = n["action"].replace(" ", "_").lower() or "unspecified_action"
            modality = n["modality"]

            predicate = f"{modality}({agent},{action})"
            try:
                n["pyreason_eval"] = bridge.query(predicate)
            except Exception as e:
                logger.error("PyReason query failed: %s", e)
                n["pyreason_eval"] = None

    return norms
