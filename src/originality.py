from __future__ import annotations

try:
    from .argumentation import LegalArgument, OptimizedAAF
    from .norm_extraction import extract_norms
except ImportError:  # pragma: no cover
    from argumentation import LegalArgument, OptimizedAAF
    from norm_extraction import extract_norms


class LogicBasedLegalReasoner:
    """Small copyright-originality case study built on the shared AAF core.

    These keyword-derived claims are research hypotheses, not legal holdings.
    """

    def __init__(self) -> None:
        self.arguments: dict[str, LegalArgument] = {}

    def _claims(self, text: str) -> list[tuple[str, str, float]]:
        lowered = text.lower()
        claims: list[tuple[str, str, float]] = []
        if "author's personality" in lowered or "empreinte de la personnalité" in lowered:
            claims.append(("Work shows the author's personality.", "The work is original.", 0.88))
        if "free creative choices" in lowered or "choix créatifs libres" in lowered:
            claims.append(("Author made free creative choices.", "The work is original.", 0.91))
        if "dictated by technical function" in lowered or "functional elements lack originality" in lowered:
            claims.append(("Form is dictated by technical function.", "The work is not original.", 0.95))
        return claims

    def analyze_legal_case(
        self, text: str, timesteps: int = 10, run_pyreason: bool = False, *, pyreason_timeout_seconds: int = 60,
    ) -> dict:
        # Reset per case: dataset runs must not leak arguments from prior rows.
        self.arguments = {}
        attacks: set[tuple[str, str]] = set()
        supports: set[tuple[str, str]] = set()

        for index, (premise, conclusion, confidence) in enumerate(self._claims(text)):
            arg_id = f"originality_{index}"
            self.arguments[arg_id] = LegalArgument(
                id=arg_id,
                premise=premise,
                conclusion=conclusion,
                confidence=(round(confidence * 0.8, 4), confidence),
            )

        ids = list(self.arguments)
        for i, left_id in enumerate(ids):
            for right_id in ids[i + 1 :]:
                left = self.arguments[left_id].conclusion.lower()
                right = self.arguments[right_id].conclusion.lower()
                if ("not original" in left and "original" in right) or ("not original" in right and "original" in left):
                    attacks.add((left_id, right_id))
                    attacks.add((right_id, left_id))

        aaf = OptimizedAAF(self.arguments.values(), attacks=attacks, supports=supports)
        pyreason_results = aaf.run_pyreason(timesteps=timesteps, timeout_seconds=pyreason_timeout_seconds) if run_pyreason else None
        originality_support = [a for a in self.arguments.values() if "not original" not in a.conclusion.lower()]
        originality_score = (
            sum((arg.confidence[0] + arg.confidence[1]) / 2 for arg in originality_support) / len(originality_support)
            if originality_support else 0.0
        )

        return {
            "arguments": {
                key: {
                    "id": arg.id,
                    "premise": arg.premise,
                    "conclusion": arg.conclusion,
                    "confidence": arg.confidence,
                }
                for key, arg in self.arguments.items()
            },
            "attacks": sorted(attacks),
            "supports": sorted(supports),
            "norms": extract_norms(text, source_id="originality-case"),
            "aaf": aaf.export_to_dict(),
            "pyreason_results": pyreason_results,
            "pyreason_eval": {
                "originality_score": originality_score,
                "protection_score": originality_score,
                "basis": "keyword_heuristic_not_pyreason_or_legal_judgment",
                "review_status": "unreviewed",
                "decision_authorized": False,
                "legal_certification": False,
            },
        }
