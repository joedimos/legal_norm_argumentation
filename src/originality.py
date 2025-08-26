from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
import logging
from .argumentation import LegalArgument, Sequent, OptimizedAAF
from .norm_extraction import extract_norms
from .models import nlp
import networkx as nx

logger = logging.getLogger(__name__)

# --- Keywords ---
ORIGINALITY_KEYWORDS = {
    "empreinte de la personnalité", "author's personality",
    "choix créatifs libres", "free creative choices",
    "dictated by technical function"
}


# -------------------------
# OriginalityAnnotator
# -------------------------
class OriginalityAnnotator:
    """
    Extracts originality-related claims from text, builds legal arguments,
    constructs sequents, and attaches them to a legal reasoner.
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
        """
        Identify candidate originality claims in the text.
        Returns a list of dicts with premise, conclusion, and confidence.
        """
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
        """
        Convert extracted claims into LegalArgument objects, adding
        supporting and attacking relationships.
        """
        arguments: Dict[str, LegalArgument] = {}

        for i, claim in enumerate(claims):
            conf = claim.get("confidence", 0.8)
            low, high = conf * 0.8, conf

            supports_originality = "not" not in claim["conclusion"].lower()

            if supports_originality:
                # Originality hypothesis
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
                # Protection conclusion
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
                # Non-originality hypothesis
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
                # No protection conclusion
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

        # Create basic attack links between opposing claims
        self._link_attacks(arguments)
        return arguments

    def _link_attacks(self, arguments: Dict[str, LegalArgument]) -> None:
        """Establish mutual attacks between originality and non-originality claims."""
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
        """Generate sequents linking originality claims to copyright protection outcomes."""
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
        """
        Full pipeline: extract claims, build arguments, construct attacks and sequents,
        and attach arguments to the global reasoner.
        """
        # Reset state
        self._current_arguments.clear()
        self._current_attacks.clear()
        self.sequents = []

        claims = self.extract_originality_claims(text)
        if not claims:
            return {"arguments": {}, "attacks": set(), "sequents": [], "norms": []}

        # Build arguments & attacks
        self._current_arguments = self._create_arguments_from_claims(claims)
        for arg in self._current_arguments.values():
            for attacked in arg.attacking_args:
                if attacked in self._current_arguments:
                    self._current_attacks.add((arg.id, attacked))

        # Construct sequents
        self.construct_sequents()

        # Register with reasoner
        for arg in self._current_arguments.values():
            self.reasoner.add_argument(arg)

        # Norm extraction
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
    and performs AAF reasoning with the OptimizedAAF backend.
    """

    def __init__(self):
        self.arguments: Dict[str, LegalArgument] = {}
        self.annotator = OriginalityAnnotator(self)

        # High-level interpretive rules
        self.ip_rules = {
            "imprint_of_personality": lambda arg_id: self._arg_confidence_mean(arg_id) > 0.7,
            "creative_choices": lambda arg_id: self._arg_confidence_mean(arg_id) > 0.7,
            "technical_function": lambda arg_id: self._arg_confidence_mean(arg_id) > 0.8,
        }

    def add_argument(self, arg: LegalArgument) -> None:
        """Register a new argument in the global pool."""
        self.arguments[arg.id] = arg

    def _arg_confidence_mean(self, arg_id: str) -> float:
        """Compute the mean confidence for a given argument."""
        arg = self.arguments.get(arg_id)
        if not arg:
            return 0.0
        return sum(arg.confidence) / 2.0

    def analyze_legal_case(self, text: str) -> Dict:
        """
        Main entrypoint: extract arguments, build AAF, compute extensions,
        and evaluate sequents.
        """
        result = self.annotator.annotate_originality(text)

        # AAF reasoning
        args_ids = set(result["arguments"].keys())
        aaf = OptimizedAAF(args_ids, result["attacks"])
        aaf_summary = {
            "preferred": [list(s) for s in aaf.get_preferred_extensions()],
            "stable": [list(s) for s in aaf.get_stable_extensions()],
            "has_cycle": aaf.has_cycle(),
        }

        # Evaluate sequents (naive: min confidence propagation)
        sequent_eval = []
        for s in result["sequents"]:
            min_ant = min((self._arg_confidence_mean(a) for a in s.antecedents), default=0.0)
            min_con = min((self._arg_confidence_mean(c) for c in s.consequents), default=0.0)
            sequent_eval.append({
                "antecedents": list(s.antecedents),
                "consequents": list(s.consequents),
                "confidence": s.confidence,
                "valid": (min_con >= min_ant - 0.2),
            })

        return {
            "arguments": result["arguments"],
            "attacks": list(result["attacks"]),
            "sequents": sequent_eval,
            "aaf": aaf_summary,
            "norms": result["norms"],
        }
