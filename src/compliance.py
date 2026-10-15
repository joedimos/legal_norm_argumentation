from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence

from .defeasible import reason
from .normative_model import FactContext, Modality, NormInterpretation


@dataclass(frozen=True)
class ComplianceFinding:
    norm_id: str
    status: str
    modality: str
    subject: str
    action: str
    explanation: str


def _action_seen(norm: NormInterpretation, context: FactContext) -> bool:
    return norm.action in context.observed_actions


def evaluate(norms: Sequence[NormInterpretation], context: FactContext) -> Dict:
    reasoning = reason(norms, context)
    applicable = {item["norm_id"] for item in reasoning["conclusions"] if item["status"] == "applicable"}
    findings: List[ComplianceFinding] = []

    for norm in norms:
        activation_status = reasoning["activations"][norm.id]["status"]
        if norm.id not in applicable:
            if activation_status == "undetermined":
                findings.append(ComplianceFinding(norm.id, "unknown", norm.modality.value, norm.subject, norm.action, "Applicability cannot be established from available facts."))
            continue

        performed = _action_seen(norm, context)
        if norm.modality == Modality.OBLIGATION:
            status = "satisfied" if performed else "violated"
            explanation = "Required action observed." if performed else "Required action not observed."
        elif norm.modality == Modality.PROHIBITION:
            status = "violated" if performed else "satisfied"
            explanation = "Prohibited action observed." if performed else "Prohibited action not observed."
        elif norm.modality == Modality.PERMISSION:
            status = "permitted" if performed else "not_exercised"
            explanation = "Permitted action observed." if performed else "Permission exists but was not exercised."
        else:
            status = "advisory_met" if performed else "advisory_unmet"
            explanation = "Recommendation is advisory rather than a hard compliance constraint."
        findings.append(ComplianceFinding(norm.id, status, norm.modality.value, norm.subject, norm.action, explanation))

    violations = [asdict(item) for item in findings if item.status == "violated"]
    unknown = [asdict(item) for item in findings if item.status == "unknown"]
    return {
        "status": "non_compliant" if violations else "indeterminate" if unknown or reasoning["unresolved_conflicts"] else "compliant",
        "findings": [asdict(item) for item in findings],
        "violations": violations,
        "reasoning": reasoning,
    }


def generate_counterexamples(norms: Sequence[NormInterpretation], fact_names: Sequence[str], actor: Optional[str] = None, max_contexts: int = 256) -> List[Dict]:
    """Bounded counterexample generation over a finite boolean context space."""
    if 2 ** len(fact_names) > max_contexts:
        raise ValueError("counterexample search space exceeds max_contexts")
    actions = sorted({norm.action for norm in norms})
    counterexamples: List[Dict] = []
    for values in product((False, True), repeat=len(fact_names)):
        facts = dict(zip(fact_names, values))
        traces = [()] + [(action,) for action in actions]
        for trace in traces:
            context = FactContext(facts=facts, actor=actor, observed_actions=trace)
            report = evaluate(norms, context)
            if report["status"] == "non_compliant":
                counterexamples.append({"facts": facts, "observed_actions": list(trace), "violations": report["violations"]})
    return counterexamples
