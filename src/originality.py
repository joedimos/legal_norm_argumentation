from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
import logging
from .argumentation import LegalArgument, Sequent, OptimizedAAF
from .norm_extraction import extract_norms
from .models import nlp
import networkx as nx

logger = logging.getLogger(__name__)

# Keywords from your earlier set
ORIGINALITY_KEYWORDS = {
    "empreinte de la personnalité", "author's personality",
    "choix créatifs libres", "free creative choices",
    "dictated by technical function"
}

# --- OriginalityAnnotator ---
class OriginalityAnnotator:
    def __init__(self, reasoner: "LogicBasedLegalReasoner"):
        self.reasoner = reasoner
        self._current_arguments: Dict[str, LegalArgument] = {}
        self._current_attacks: Set[Tuple[str, str]] = set()
        self.sequents: List[Sequent] = []

    def extract_originality_claims(self, text: str) -> List[Dict]:
        claims = []
        t = text.lower()
        if any(kw in t for kw in ORIGINALITY_KEYWORDS):
            if "empreinte de la personnalité" in t or "author's personality" in t:
                claims.append({
                    "premise": "Work shows the imprint of the author's personality.",
                    "conclusion": "The work is original.",
                    "confidence": 0.88
                })
            if "choix créatifs libres" in t or "free creative choices" in t:
                claims.append({
                    "premise": "Author made free and creative choices.",
                    "conclusion": "The work is original.",
                    "confidence": 0.91
                })
        if "dictated by technical function" in t:
            claims.append({
                "premise": "Form is dictated by technical function.",
                "conclusion": "The work is NOT original.",
                "confidence": 0.95
            })
            claims.append({
                "premise": "Functional elements lack originality.",
                "conclusion": "No copyright protection for functional aspects.",
                "confidence": 0.90
            })
        return claims

    def _create_arguments_from_claims(self, claims: List[Dict]) -> Dict[str, LegalArgument]:
        arguments: Dict[str, LegalArgument] = {}
        for i, c in enumerate(claims):
            base = f"claim_{i}"
            conf = c.get("confidence", 0.8)
            low = conf * 0.8
            high = conf
            # create support or attack depending on conclusion
            supports_originality = "not" not in c["conclusion"].lower()
            if supports_originality:
                orig_id = f"is_original_{i}"
                arguments[orig_id] = LegalArgument(
                    id=orig_id,
                    premise=c["premise"],
                    conclusion=c["conclusion"],
                    confidence=(low, high),
                    supporting_args=set(),
                    attacking_args=set(),
                    sequent_type="hypothesis"
                )
                prot_id = f"copyright_protected_{i}"
                arguments[prot_id] = LegalArgument(
                    id=prot_id,
                    premise=f"Based on {orig_id}",
                    conclusion="Copyright protection likely applies.",
                    confidence=(low * 0.9, high),
                    supporting_args={orig_id},
                    attacking_args=set(),
                    sequent_type="conclusion"
                )
            else:
                not_id = f"not_original_{i}"
                arguments[not_id] = LegalArgument(
                    id=not_id,
                    premise=c["premise"],
                    conclusion=c["conclusion"],
                    confidence=(low, high),
                    supporting_args=set(),
                    attacking_args=set(),
                    sequent_type="hypothesis"
                )
                notprot = f"copyright_not_protected_{i}"
                arguments[notprot] = LegalArgument(
                    id=notprot,
                    premise=f"Based on {not_id}",
                    conclusion="Copyright protection likely does not apply.",
                    confidence=(low * 0.9, high),
                    supporting_args={not_id},
                    attacking_args=set(),
                    sequent_type="conclusion"
                )
        # create basic attack links between opposing claims
        ids = list(arguments.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = arguments[ids[i]]
                b = arguments[ids[j]]
                if ("original" in a.conclusion.lower() and "not original" in b.conclusion.lower()) or \
                   ("original" in b.conclusion.lower() and "not original" in a.conclusion.lower()):
                    a.attacking_args.add(b.id)
                    b.attacking_args.add(a.id)
        return arguments

    def construct_sequents(self):
        sequents = []
        args = self._current_arguments
        originality_args = {k for k in args if "is_original" in k}
        protection_args = {k for k in args if "copyright_protected" in k}
        no_protection_args = {k for k in args if "copyright_not_protected" in k}
        not_original_args = {k for k in args if "not_original" in k}
        for orig in originality_args:
            conf = args[orig].confidence[0]
            sequents.append(Sequent(antecedents={orig}, consequents=protection_args, confidence=conf * 0.9))
            sequents.append(Sequent(antecedents=no_protection_args, consequents=not_original_args, confidence=conf * 0.9, is_contrapositive=True))
        self.sequents = sequents

    def annotate_originality(self, text: str) -> Dict:
        claims = self.extract_originality_claims(text)
        if not claims:
            return {"arguments": {}, "attacks": set(), "sequents": [], "norms": []}
        self._current_arguments = self._create_arguments_from_claims(claims)
        # identify attacks set
        for aid, a in self._current_arguments.items():
            for attacked in a.attacking_args:
                if attacked in self._current_arguments:
                    self._current_attacks.add((aid, attacked))
        self.construct_sequents()
        # attach to global reasoner
        for arg in self._current_arguments.values():
            self.reasoner.add_argument(arg)
        # also call norm extraction to enrich arguments
        norms = extract_norms(text, use_transformer=False)
        return {
            "arguments": self._current_arguments,
            "attacks": self._current_attacks,
            "sequents": self.sequents,
            "norms": norms
        }

# --- LogicBasedLegalReasoner ---
class LogicBasedLegalReasoner:
    def __init__(self):
        self.arguments: Dict[str, LegalArgument] = {}
        # high-level rule mapping
        self.ip_rules = {
            "imprint_of_personality": lambda arg_id: self._arg_confidence_mean(arg_id) > 0.7,
            "creative_choices": lambda arg_id: self._arg_confidence_mean(arg_id) > 0.7,
            "technical_function": lambda arg_id: self._arg_confidence_mean(arg_id) > 0.8
        }
        self.annotator = OriginalityAnnotator(self)

    def add_argument(self, arg: LegalArgument):
        self.arguments[arg.id] = arg

    def _arg_confidence_mean(self, arg_id: str) -> float:
        a = self.arguments.get(arg_id)
        if not a:
            return 0.0
        return (a.confidence[0] + a.confidence[1]) / 2.0

    def analyze_legal_case(self, text: str) -> Dict:
        result = self.annotator.annotate_originality(text)
        # AAF analysis
        args_ids = set(result["arguments"].keys())
        attacks = result["attacks"]
        aaf = OptimizedAAF(args_ids, attacks)
        aaf_summary = {
            "preferred": [list(s) for s in aaf.get_preferred_extensions()],
            "stable": [list(s) for s in aaf.get_stable_extensions()],
            "has_cycle": aaf.has_cycle()
        }
        # Evaluate sequents (naive)
        sequent_eval = []
        for s in result["sequents"]:
            min_ant = min([self._arg_confidence_mean(a) for a in s.antecedents]) if s.antecedents else 0.0
            min_con = min([self._arg_confidence_mean(c) for c in s.consequents]) if s.consequents else 0.0
            sequent_eval.append({
                "antecedents": list(s.antecedents), "consequents": list(s.consequents),
                "confidence": s.confidence, "valid": (min_con >= min_ant - 0.2)
            })
        return {
            "arguments": result["arguments"],
            "attacks": list(attacks),
            "sequents": sequent_eval,
            "aaf": aaf_summary,
            "norms": result["norms"]
        }
