from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import product

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


def evaluate(norms: Sequence[NormInterpretation], context: FactContext, *, backend="core", pyreason_timeout_seconds=60,
             pyreason_program=None, pyreason_worker_mode="isolated") -> dict:
    if pyreason_program is not None and backend != "pyreason":
        raise ValueError("graph inference requires the PyReason backend")
    if pyreason_worker_mode not in {"isolated", "pooled"}:
        raise ValueError("unsupported PyReason worker mode")
    reasoning = reason(norms, context, backend=backend, pyreason_timeout_seconds=pyreason_timeout_seconds,
                       pyreason_program=pyreason_program, pyreason_worker_mode=pyreason_worker_mode)
    applicable = {item["norm_id"] for item in reasoning["conclusions"] if item["status"] == "applicable"}
    findings: list[ComplianceFinding] = []

    for norm in norms:
        activation_status = reasoning["activations"][norm.id]["status"]
        if norm.id not in applicable:
            if activation_status == "undetermined" or any(
                c["norm_id"] == norm.id and c["status"] == "undetermined" for c in reasoning["conclusions"]
            ):
                findings.append(ComplianceFinding(norm.id, "unknown", norm.modality.value, norm.subject, norm.action, "Applicability cannot be established from available facts."))
            continue

        if norm.deadline or norm.object is not None:
            findings.append(ComplianceFinding(norm.id, "unknown", norm.modality.value, norm.subject, norm.action,
                                              "Deadline/object requires reviewed temporal, structured action observations."))
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
        "status": ("indeterminate" if reasoning.get("backend", {}).get("status", "computed") != "computed"
                   else "non_compliant" if violations else "indeterminate" if unknown or reasoning["unresolved_conflicts"] else "compliant"),
        "findings": [asdict(item) for item in findings],
        "violations": violations,
        "reasoning": reasoning,
    }


def generate_counterexamples(norms: Sequence[NormInterpretation], fact_names: Sequence[str], actor: str | None = None, max_contexts: int = 256) -> list[dict]:
    """Bounded counterexample generation over a finite boolean context space."""
    if 2 ** len(fact_names) > max_contexts:
        raise ValueError("counterexample search space exceeds max_contexts")
    actions = sorted({norm.action for norm in norms})
    counterexamples: list[dict] = []
    for values in product((False, True), repeat=len(fact_names)):
        facts = dict(zip(fact_names, values, strict=True))
        traces = [()] + [(action,) for action in actions]
        for trace in traces:
            context = FactContext(facts=facts, actor=actor, observed_actions=trace)
            report = evaluate(norms, context)
            if report["status"] == "non_compliant":
                counterexamples.append({"facts": facts, "observed_actions": list(trace), "violations": report["violations"]})
    return counterexamples
