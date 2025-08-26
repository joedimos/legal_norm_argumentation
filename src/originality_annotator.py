from typing import List, Dict, Set, Tuple
from .argumentation import LegalArgument, Sequent, OptimizedAAF, PyReasonConnector
from .norm_extraction import extract_norms
import logging

logger = logging.getLogger(__name__)


class OriginalityAnnotator:
    """
    Annotates text for originality-related claims, constructs arguments and attacks,
    builds sequents, and integrates with PyReason for reasoning.
    """

    def __init__(self, reasoner: "LogicBasedLegalReasoner"):
        self.reasoner = reasoner
        self._current_arguments: Dict[str, LegalArgument] = {}
        self._current_attacks: Set[Tuple[str, str]] = set()
        self.sequents: List[Sequent] = []

    # --- Extract claims ---
    def extract_originality_claims(self, text: str) -> List[Dict]:
        t = text.lower()
        claims = []

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
            claims.extend([
                {
                    "premise": "Form is dictated by technical function.",
                    "conclusion": "The work is NOT original.",
                    "confidence": 0.95
                },
                {
                    "premise": "Functional elements lack originality.",
                    "conclusion": "No copyright protection for functional aspects.",
                    "confidence": 0.90
                }
            ])
        return claims

    # --- Build LegalArgument objects ---
    def _create_arguments_from_claims(self, claims: List[Dict]) -> Dict[str, LegalArgument]:
        arguments: Dict[str, LegalArgument] = {}
        for i, c in enumerate(claims):
            conf = c.get("confidence", 0.8)
            low, high = conf * 0.8, conf

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
                    confidence=(low*0.9, high),
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
                notprot_id = f"copyright_not_protected_{i}"
                arguments[notprot_id] = LegalArgument(
                    id=notprot_id,
                    premise=f"Based on {not_id}",
                    conclusion="Copyright protection likely does not apply.",
                    confidence=(low*0.9, high),
                    supporting_args={not_id},
                    attacking_args=set(),
                    sequent_type="conclusion"
                )

        # link opposing claims as attacks
        self._link_attacks(arguments)
        return arguments

    def _link_attacks(self, arguments: Dict[str, LegalArgument]) -> None:
        ids = list(arguments.keys())
        for i, a_id in enumerate(ids):
            for b_id in ids[i+1:]:
                a, b = arguments[a_id], arguments[b_id]
                if ("original" in a.conclusion.lower() and "not original" in b.conclusion.lower()) or \
                   ("original" in b.conclusion.lower() and "not original" in a.conclusion.lower()):
                    a.attacking_args.add(b.id)
                    b.attacking_args.add(a.id)

    # --- Construct sequents ---
    def construct_sequents(self) -> None:
        args = self._current_arguments
        originality_args = {k for k in args if "is_original" in k}
        protection_args = {k for k in args if "copyright_protected" in k}
        no_protection_args = {k for k in args if "copyright_not_protected" in k}
        not_original_args = {k for k in args if "not_original" in k}

        sequents = []
        for orig in originality_args:
            conf = args[orig].confidence[0]
            sequents.append(Sequent(
                antecedents={orig},
                consequents=protection_args,
                confidence=conf*0.9
            ))
            sequents.append(Sequent(
                antecedents=no_protection_args,
                consequents=not_original_args,
                confidence=conf*0.9,
                is_contrapositive=True
            ))
        self.sequents = sequents

    # --- Full annotation + PyReason integration ---
    def annotate_originality(self, text: str, run_pyreason: bool = True) -> Dict:
        self._current_arguments.clear()
        self._current_attacks.clear()
        self.sequents = []

        claims = self.extract_originality_claims(text)
        if not claims:
            return {"arguments": {}, "attacks": set(), "sequents": [], "norms": []}

        self._current_arguments = self._create_arguments_from_claims(claims)

        # fill attacks
        for arg in self._current_arguments.values():
            for attacked in arg.attacking_args:
                if attacked in self._current_arguments:
                    self._current_attacks.add((arg.id, attacked))

        # build sequents
        self.construct_sequents()

        # register arguments with global reasoner
        for arg in self._current_arguments.values():
            self.reasoner.add_argument(arg)

        # extract norms
        norms = extract_norms(text, use_transformer=False)

        # optionally run PyReason reasoning
        pyreason_results = None
        if run_pyreason:
            connector = PyReasonConnector(self._current_arguments, self._current_attacks)
            pyreason_results = connector.run_reasoning()

        return {
            "arguments": self._current_arguments,
            "attacks": self._current_attacks,
            "sequents": self.sequents,
            "norms": norms,
            "pyreason": pyreason_results
        }
