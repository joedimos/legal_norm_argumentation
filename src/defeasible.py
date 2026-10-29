from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from .normative_model import FactContext, NormInterpretation


@dataclass(frozen=True)
class Activation:
    norm_id: str
    status: str
    reasons: tuple[str, ...]


def activation(norm: NormInterpretation, context: FactContext) -> Activation:
    effective = norm.authority.is_effective(context.at_date)
    if effective is False:
        return Activation(norm.id, "inactive", ("outside_temporal_scope",))

    if effective is None and (norm.authority.effective_from or norm.authority.effective_to):
        return Activation(norm.id, "undetermined", ("evaluation_date_unknown",))
    if context.actor is not None and context.actor != norm.subject:
        return Activation(norm.id, "inactive", ("different_subject",))

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


def reason(norms: Sequence[NormInterpretation], context: FactContext, *, backend="core", pyreason_timeout_seconds=60) -> dict:
    from .argumentation import OptimizedAAF
    from .conflicts import detect_conflicts
    from .priority import resolve_pair

    if len({norm.id for norm in norms}) != len(norms):
        raise ValueError("norm ids must be unique")
    backend_report = None
    if backend == "core":
        activations = {norm.id: activation(norm, context) for norm in norms}
    elif backend == "pyreason":
        from .pyreason_normative import run_activation_backend

        backend_report = run_activation_backend(norms, context, timeout_seconds=pyreason_timeout_seconds)
        activations = ({key: Activation(key, item["status"], tuple(item["reasons"]))
                        for key, item in backend_report["activations"].items()}
                       if backend_report["status"] == "computed" else
                       {norm.id: Activation(norm.id, "undetermined", ("pyreason_computation_incomplete",)) for norm in norms})
    else:
        raise ValueError("unsupported normative reasoning backend")
    active = {norm.id: norm for norm in norms if activations[norm.id].status == "active"}
    conflicts = detect_conflicts(list(active.values()))
    attacks = set()
    priority = []
    for conflict in conflicts:
        decision = resolve_pair(active[conflict.left_id], active[conflict.right_id])
        priority.append(decision)
        if decision.status == "resolved":
            attacks.add((decision.winner_id, decision.loser_id))
        else:
            attacks.update({(conflict.left_id, conflict.right_id), (conflict.right_id, conflict.left_id)})
    framework = OptimizedAAF(active, attacks=attacks)
    labels = framework.grounded_labeling()
    conclusions = []
    for norm in active.values():
        label = labels[norm.id]
        conclusions.append({
            "norm_id": norm.id, "status": {"in": "applicable", "out": "defeated", "undec": "undetermined"}[label],
            "modality": norm.modality.value, "subject": norm.subject, "action": norm.action,
            "reason": "grounded_defeat_semantics",
            "defeated_by": sorted(a for a, b in attacks if b == norm.id and labels[a] == "in"),
        })
    unresolved = [c.to_dict() for c in conflicts
                  if labels[c.left_id] == "undec" and labels[c.right_id] == "undec"]
    return {
        "activations": {key: asdict(value) for key, value in activations.items()},
        "conclusions": conclusions,
        "priority_decisions": [asdict(item) for item in priority],
        "structured_conflicts": [c.to_dict() for c in conflicts],
        "unresolved_conflicts": unresolved,
        "argumentation": {"semantics": "grounded", "labels": labels, "defeats": sorted(map(list, attacks))},
        **({"backend": backend_report} if backend_report is not None else {}),
    }
