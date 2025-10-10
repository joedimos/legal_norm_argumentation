from typing import List, Dict, Optional
import re
import logging

from .models import nlp, load_transformer_classifier

from pyreason import PyReason, Fact, Rule, Query

logger = logging.getLogger(__name__)

# Keywords

DEONTIC_KEYWORDS: Dict[str, List[str]] = {
    "obligation": ["shall", "must", "is required to", "required to"],
    "permission": ["may", "is permitted to", "is allowed to"],
    "prohibition": ["shall not", "must not", "is prohibited", "is forbidden"],
}


# Rule-based norm extractor

def rule_based_norms(text: str) -> List[Dict[str, str]]:
    """
    Extracts deontic norms (obligation, permission, prohibition) from text using keyword heuristics.
    """
    doc_text = text.replace("\n", " ")
    sentences = re.split(r'(?<=[\.\;\:])\s+', doc_text)
    norms: List[Dict[str, str]] = []

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
                    break  # prevent duplicate modality hits for same sentence
    return norms


def extract_agent_simple(sentence: str) -> str:
    """
    Heuristic extraction of the agent in a deontic sentence (the subject before the modal verb).
    """
    m = re.search(
        r'(.{1,80}?)\b(shall|must|may|is required to|is permitted to|shall not|must not)\b',
        sentence,
        flags=re.I,
    )
    if m:
        return m.group(1).strip(",;: ")
    return ""


def extract_action_simple(sentence: str, trigger: str) -> str:
    """
    Heuristic extraction of the action following a modal trigger.
    """
    idx = sentence.lower().find(trigger)
    if idx >= 0:
        return sentence[idx + len(trigger):].strip()
    return ""


# Optional transformer-based classifier

_transformer = None


def classify_sentence_modality(sentence: str) -> Optional[Dict]:
    """
    Optionally classifies a sentence’s modality using a transformer-based classifier.
    Returns a dictionary of classification results if available.
    """
    global _transformer
    if _transformer is None:
        _transformer = load_transformer_classifier()

    if _transformer is None:
        logger.warning("Transformer classifier not available.")
        return None

    try:
        return _transformer(sentence, top_k=3)
    except Exception as e:
        logger.error("Transformer classification failed: %s", e)
        return None


# PyReason Bridge

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
        - prohibition(A, X) -> ~permission(A, X)
        - obligation(A, X) -> ~prohibition(A, X)
        """
        conflict_rules = [
            Rule("prohibition(A, X) -> ~permission(A, X)"),
            Rule("obligation(A, X) -> ~prohibition(A, X)")
        ]

        self.rules.extend(conflict_rules)
        for r in conflict_rules:
            self.engine.add_rule(r)

    def query(self, predicate: str) -> Optional[float]:
        """
        Query the reasoning engine for a given predicate and return its evaluated truth value.
        """
        try:
            q = Query(predicate)
            return self.engine.evaluate_query(q)
        except Exception as e:
            logger.error("PyReason query failed: %s", e)
            return None


# High-level API

def extract_norms(
    text: str,
    use_transformer: bool = False,
    use_pyreason: bool = True
) -> List[Dict]:
    """
    Extracts and optionally classifies deontic norms from text.
    Optionally evaluates them using PyReason.
    """
    norms = rule_based_norms(text)

    # Optionally enrich with transformer classification
    if use_transformer:
        for n in norms:
            n["transformer"] = classify_sentence_modality(n["sentence"])

    # Optionally integrate with PyReason
    if use_pyreason and norms:
        bridge = PyReasonNormBridge()
        bridge.add_norms(norms)
        bridge.add_conflict_rules()

        for n in norms:
            agent = n["agent"].replace(" ", "_").lower() or "unspecified_agent"
            action = n["action"].replace(" ", "_").lower() or "unspecified_action"
            modality = n["modality"]

            predicate = f"{modality}({agent},{action})"
            n["pyreason_eval"] = bridge.query(predicate)

    return norms
