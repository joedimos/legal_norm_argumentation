from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

from .normative_model import FactContext, NormInterpretation
from .priority import PriorityDecision, resolve_conflicts


@dataclass(frozen=True)
class Activation:
    norm_id: str
    status: str
    reasons: Tuple[str, ...]


def activation(norm: NormInterpretation, context: FactContext) -> Activation:
    effective = norm.authority.is_effective(context.at_date)
    if effective is False:
        return Activation(norm.id, "inactive", ("outside_temporal_scope",))

    condition_values = [predicate.evaluate(context.facts) for predicate in norm.conditions]
    if any(value is False for value in condition_values):
        return Activation(norm.id, "inactive", ("condition_false",))
    if any(value is None for value in condition_values):
        return Activation(norm.id, "undetermined", ("condition_unknown",))

    exception_values = [predicate.evaluate(context.facts) for predicate in norm.exceptions]
    if any(value is True for value in exception_values):
        return Activation(norm.id, "defeated", ("exception_applies",))
    if any(value is None for value in exception_values):
        return Activation(norm.id, "undetermined", ("exception_unknown",))
    return Activation(norm.id, "active", ("conditions_satisfied",))


def reason(norms: Sequence[NormInterpretation], context: FactContext) -> Dict:
    activations = {norm.id: activation(norm, context) for norm in norms}
    active = [norm for norm in norms if activations[norm.id].status == "active"]
    priority = resolve_conflicts(active)
    defeated_by_priority: Dict[str, PriorityDecision] = {
        decision.loser_id: decision
        for decision in priority
        if decision.status == "resolved" and decision.loser_id is not None
    }

    conclusions: List[Dict] = []
    for norm in active:
        if norm.id in defeated_by_priority:
            decision = defeated_by_priority[norm.id]
            conclusions.append({"norm_id": norm.id, "status": "defeated", "reason": decision.principle, "defeated_by": decision.winner_id})
        else:
            conclusions.append({"norm_id": norm.id, "status": "applicable", "modality": norm.modality.value, "subject": norm.subject, "action": norm.action})

    unresolved = [asdict(item) for item in priority if item.status == "unresolved"]
    return {
        "activations": {key: asdict(value) for key, value in activations.items()},
        "conclusions": conclusions,
        "priority_decisions": [asdict(item) for item in priority],
        "unresolved_conflicts": unresolved,
    }
