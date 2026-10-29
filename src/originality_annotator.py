"""Compatibility wrapper for the consolidated originality implementation."""

try:
    from .originality import LogicBasedLegalReasoner
except ImportError:  # pragma: no cover
    from originality import LogicBasedLegalReasoner


class OriginalityAnnotator:
    def __init__(self, reasoner: LogicBasedLegalReasoner):
        self.reasoner = reasoner

    def annotate_originality(self, text: str, run_pyreason: bool = False, *, pyreason_timeout_seconds: int = 60):
        return self.reasoner.analyze_legal_case(
            text, run_pyreason=run_pyreason, pyreason_timeout_seconds=pyreason_timeout_seconds,
        )
