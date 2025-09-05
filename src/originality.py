from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field
import logging

# External dependencies
import networkx as nx
from pyreason import PyReason, Fact, Rule, Query

# Local imports
from .argumentation import LegalArgument, Sequent
from .norm_extraction import extract_norms
from .models import nlp

logger = logging.getLogger(__name__)

# --- Keywords ---
ORIGINALITY_KEYWORDS = {
    "empreinte de la personnalité", "author's personality",
    "choix créatifs libres", "free creative choices",
    "dictated by technical"
    
    
class PyReasonBridge:
    """
    Bridge layer between our LegalArgument/Sequent representation
    and the PyReason reasoning engine.
    """

    def __init__(self):
        self.engine = PyReason()
        self.facts: Dict[str, Fact] = {}
        self.rules: Dict[str, Rule] = {}

    def add_argument_as_fact(self, arg: LegalArgument) -> None:
        """
        Translate a LegalArgument into a PyReason Fact.
        """
        fact_id = f"fact_{arg.id}"
        fact = Fact(
            predicate=arg.conclusion.replace(" ", "_").lower(),
            truth_value=(sum(arg.confidence) / 2.0),
            meta={"premise": arg.premise, "id": arg.id}
        )
        self.facts[arg.id] = fact
        self.engine.add_fact(fact)

    def add_sequent_as_rule(self, sequent: Sequent) -> None:
        """
        Translate a Sequent into a PyReason Rule.
        """
        antecedent_preds = [
            self.facts[a].predicate for a in sequent.antecedents if a in self.facts
        ]
        consequent_preds = [
            self.facts[c].predicate for c in sequent.consequents if c in self.facts
        ]
        if not antecedent_preds or not consequent_preds:
            return

        rule_expr = " & ".join(antecedent_preds) + " -> " + " & ".join(consequent_preds)
        rule = Rule(rule_expr, weight=sequent.confidence)
        self.rules[rule_expr] = rule
        self.engine.add_rule(rule)

    def query(self, predicate: str) -> float:
        """
        Query the truth value of a predicate after reasoning.
        """
        q = Query(predicate)
        return self.engine.evaluate_query(q)


# -------------------------
# OriginalityAnnotator
# -------------------------
class OriginalityAnnotator:
    """
    Extracts originality-related claims from text, builds legal arguments,
    constructs sequents, and attaches them to the reasoner.
    """

    def __init__(self, reasoner: "LogicBasedLegalReasoner"):
        self.reasoner = reasoner
        self._current_arguments: Dict[str, LegalArgument] = {}
        self._current_attacks: Set[Tuple[str, str]] = set()
        self.sequents: List[Sequent] = []

    # -------------------------
    # Claim extraction
    # -------------------------
    def extract_originality_claims(self, text: str) -> List[Dict]:
        claims = []
        t = text.lower()

        if any(kw in t for kw in ORIGINALITY_KEYWORDS):
            if "empreinte de la personnalité" in t or "author's personality" in t:
                claims.append({
                    "premise": "Work shows the imprint of the author's personality.",
                    "conclusion": "The work is original.",
                    "confidence": 0.88,
                })
            if "choix créatifs libres" in t or "free creative choices" in t:
                claims.append({
                    "premise": "Author made free and creative choices.",
                    "conclusion": "The work is original.",
                    "confidence": 0.91,
                })
        if "dictated by technical function" in t:
            claims.extend([
                {
                    "premise": "Form is dictated by technical function.",
                    "conclusion": "The work is NOT original.",
                    "confidence": 0.95,
                },
                {
                    "premise": "Functional elements lack originality.",
                    "conclusion": "No copyright protection for functional aspects.",
                    "confidence": 0.90,
                },
            ])
        return claims

    # -------------------------
    # Argument construction
    # -------------------------
    def _create_arguments_from_claims(self, claims: List[Dict]) -> Dict[str, LegalArgument]:
        arguments: Dict[str, LegalArgument] = {}
        for i, claim in enumerate(claims):
            conf = claim.get("confidence", 0.8)
            low, high = conf * 0.8, conf

            supports_originality = "not" not in claim["conclusion"].lower()
            if supports_originality:
                orig_id = f"is_original_{i}"
                arguments[orig_id] = LegalArgument(
                    id=orig_id,
                    premise=claim["premise"],
                    conclusion=claim["conclusion"],
                    confidence=(low, high),
                    supporting_args=set(),
                    attacking_args=set(),
                    sequent_type="hypothesis",
                )
                prot_id = f"copyright_protected_{i}"
                arguments[prot_id] = LegalArgument(
                    id=prot_id,
                    premise=f"Based on {orig_id}",
                    conclusion="Copyright protection likely applies.",
                    confidence=(low * 0.9, high),
                    supporting_args={orig_id},
                    attacking_args=set(),
                    sequent_type="conclusion",
                )
            else:
                not_id = f"not_original_{i}"
                arguments[not_id] = LegalArgument(
                    id=not_id,
                    premise=claim["premise"],
                    conclusion=claim["conclusion"],
                    confidence=(low, high),
                    supporting_args=set(),
                    attacking_args=set(),
                    sequent_type="hypothesis",
                )
                notprot_id = f"copyright_not_protected_{i}"
                arguments[notprot_id] = LegalArgument(
                    id=notprot_id,
                    premise=f"Based on {not_id}",
                    conclusion="Copyright protection likely does not apply.",
                    confidence=(low * 0.9, high),
                    supporting_args={not_id},
                    attacking_args=set(),
                    sequent_type="conclusion",
                )

        self._link_attacks(arguments)
        return arguments

    def _link_attacks(self, arguments: Dict[str, LegalArgument]) -> None:
        ids = list(arguments.keys())
        for i, a_id in enumerate(ids):
            for b_id in ids[i + 1:]:
                a, b = arguments[a_id], arguments[b_id]
                if (
                    ("original" in a.conclusion.lower() and "not original" in b.conclusion.lower())
                    or ("original" in b.conclusion.lower() and "not original" in a.conclusion.lower())
                ):
                    a.attacking_args.add(b.id)
                    b.attacking_args.add(a.id)

    # -------------------------
    # Sequent construction
    # -------------------------
    def construct_sequents(self) -> None:
        args = self._current_arguments
        originality_args = {k for k in args if "is_original" in k}
        protection_args = {k for k in args if "copyright_protected" in k}
        no_protection_args = {k for k in args if "copyright_not_protected" in k}
        not_original_args = {k for k in args if "not_original" in k}

        sequents = []
        for orig in originality_args:
            conf = args[orig].confidence[0]
            sequents.append(
                Sequent(
                    antecedents={orig},
                    consequents=protection_args,
                    confidence=conf * 0.9,
                )
            )
            sequents.append(
                Sequent(
                    antecedents=no_protection_args,
                    consequents=not_original_args,
                    confidence=conf * 0.9,
                    is_contrapositive=True,
                )
            )
        self.sequents = sequents

    # -------------------------
    # Orchestration
    # -------------------------
    def annotate_originality(self, text: str) -> Dict:
        self._current_arguments.clear()
        self._current_attacks.clear()
        self.sequents = []

        claims = self.extract_originality_claims(text)
        if not claims:
            return {"arguments": {}, "attacks": set(), "sequents": [], "norms": []}

        self._current_arguments = self._create_arguments_from_claims(claims)
        for arg in self._current_arguments.values():
            for attacked in arg.attacking_args:
                if attacked in self._current_arguments:
                    self._current_attacks.add((arg.id, attacked))

        self.construct_sequents()

        for arg in self._current_arguments.values():
            self.reasoner.add_argument(arg)

        norms = extract_norms(text, use_transformer=False)

        return {
            "arguments": self._current_arguments,
            "attacks": self._current_attacks,
            "sequents": self.sequents,
            "norms": norms,
        }


# -------------------------
# LogicBasedLegalReasoner
# -------------------------
class LogicBasedLegalReasoner:
    """
    Coordinates argument construction, applies IP-specific rules,
    and performs reasoning via PyReason.
    """

    def __init__(self):
        self.arguments: Dict[str, LegalArgument] = {}
        self.annotator = OriginalityAnnotator(self)
        self.pyreason = PyReasonBridge()

        self.ip_rules = {
            "imprint_of_personality": lambda arg_id: self._arg_confidence_mean(arg_id) > 0.7,
            "creative_choices": lambda arg_id: self._arg_confidence_mean(arg_id) > 0.7,
            "technical_function": lambda arg_id: self._arg_confidence_mean(arg_id) > 0.8,
        }

    def add_argument(self, arg: LegalArgument) -> None:
        self.arguments[arg.id] = arg
        self.pyreason.add_argument_as_fact(arg)

    def _arg_confidence_mean(self, arg_id: str) -> float:
        arg = self.arguments.get(arg_id)
        if not arg:
            return 0.0
        return sum(arg.confidence) / 2.0

    def analyze_legal_case(self, text: str) -> Dict:
        result = self.annotator.annotate_originality(text)

        for s in result["sequents"]:
            self.pyreason.add_sequent_as_rule(s)

        # Run PyReason reasoning
        originality_score = self.pyreason.query("the_work_is_original.")
        protection_score = self.pyreason.query("copyright_protection_likely_applies.")

        return {
            "arguments": result["arguments"],
            "attacks": list(result["attacks"]),
            "sequents": [s.__dict__ for s in result["sequents"]],
            "norms": result["norms"],
            "pyreason_eval": {
                "originality_score": originality_score,
                "protection_score": protection_score,
            },
        }