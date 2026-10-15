from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Sequence

from .compliance import evaluate
from .normative_model import FactContext, NormInterpretation


@dataclass(frozen=True)
class ComplianceEvent:
    timestamp: str
    action: str
    actor: Optional[str] = None
    facts: Mapping[str, bool] = None


class ComplianceMonitor:
    """Stateful monitor that evaluates a growing behavioural trace after each event."""

    def __init__(self, norms: Sequence[NormInterpretation], initial_facts: Optional[Mapping[str, bool]] = None) -> None:
        self.norms = tuple(norms)
        self.facts: Dict[str, bool] = dict(initial_facts or {})
        self.actions: List[str] = []
        self.events: List[ComplianceEvent] = []
        self.reports: List[Dict] = []

    def ingest(self, action: str, *, actor: Optional[str] = None, facts: Optional[Mapping[str, bool]] = None, at_date: Optional[str] = None) -> Dict:
        if facts:
            self.facts.update(facts)
        self.actions.append(action)
        event = ComplianceEvent(datetime.now(timezone.utc).isoformat(), action, actor, dict(facts or {}))
        self.events.append(event)
        report = evaluate(self.norms, FactContext(facts=dict(self.facts), actor=actor, observed_actions=tuple(self.actions), at_date=at_date))
        snapshot = {"event": asdict(event), "report": report}
        self.reports.append(snapshot)
        return snapshot

    def export(self) -> Dict:
        return {"events": [asdict(item) for item in self.events], "reports": list(self.reports)}
