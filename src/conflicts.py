"""Reviewable conflicts with explicit target, overlap, and activation evidence."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import combinations

from .normative_model import NormInterpretation


def scope_literals(norm: NormInterpretation) -> set[tuple[str, bool]]:
    return {(p.name, not p.negated) for p in norm.conditions} | {(p.name, p.negated) for p in norm.exceptions}


def scopes_overlap(left: NormInterpretation, right: NormInterpretation) -> bool:
    if (left.subject, left.action, left.object) != (right.subject, right.action, right.object):
        return False
    la, ra = left.authority, right.authority
    # Different stated jurisdictions need an explicit conflict-of-laws mapping.
    if la.jurisdiction and ra.jurisdiction and la.jurisdiction != ra.jurisdiction:
        return False
    if ((la.effective_to and ra.effective_from and la.effective_to < ra.effective_from)
            or (ra.effective_to and la.effective_from and ra.effective_to < la.effective_from)):
        return False
    literals = scope_literals(left) | scope_literals(right)
    return not any((name, not value) in literals for name, value in literals)


@dataclass(frozen=True)
class NormConflict:
    id: str
    left_id: str
    right_id: str
    kind: str
    subject: str
    action: str
    object: str | None
    activation_witness: dict[str, bool]
    temporal_overlap: tuple[int, int] | None = None
    severity: str = "incompatible"

    def to_dict(self):
        return asdict(self)


def detect_conflicts(norms) -> list[NormConflict]:
    if len({n.id for n in norms}) != len(norms):
        raise ValueError("norm ids must be unique")
    result = []
    for left, right in combinations(sorted(norms, key=lambda n: n.id), 2):
        pair = {left.modality.value, right.modality.value}
        if pair not in ({"obligation", "prohibition"}, {"permission", "prohibition"}):
            continue
        if not scopes_overlap(left, right):
            continue
        kind = "permission_prohibition" if "permission" in pair else "obligation_prohibition"
        identity = json.dumps([left.id, right.id, left.semantic_hash(), right.semantic_hash(), kind])
        result.append(NormConflict("conflict-" + hashlib.sha256(identity.encode()).hexdigest()[:20],
                                   left.id, right.id, kind, left.subject, left.action, left.object,
                                   dict(sorted(scope_literals(left) | scope_literals(right)))))
    return result


def detect_temporal_conflicts(norms) -> list[NormConflict]:
    from dataclasses import replace

    if len({n.norm.id for n in norms}) != len(norms):
        raise ValueError("norm ids must be unique")
    result = []
    for left, right in combinations(sorted(norms, key=lambda n: n.norm.id), 2):
        # Predicates at different activation steps are independent propositions.
        literals = {(n.start, name, value) for n in (left, right)
                    for name, value in scope_literals(n.norm)}
        if any((step, name, not value) in literals for step, name, value in literals):
            continue
        if any(n.norm.authority.is_effective(n.activation_date) is False for n in (left, right)):
            continue
        base = detect_conflicts([replace(n.norm, conditions=(), exceptions=()) for n in (left, right)])
        if not base:
            continue
        conflict = base[0]
        start, end = max(left.start, right.start), min(left.end, right.end)
        if start > end:
            continue
        prohibition = left if left.norm.modality.value == "prohibition" else right
        other = right if prohibition is left else left
        severity = "incompatible"
        if other.norm.modality.value == "obligation" and other.mode == "achievement" and not (
            prohibition.start <= other.start and prohibition.end >= other.end
        ):
            severity = "potential"
        identity = json.dumps([left.norm.semantic_hash(), right.norm.semantic_hash(), conflict.id,
                               left.start, left.end, left.mode, left.activation_date,
                               right.start, right.end, right.mode, right.activation_date])
        result.append(replace(conflict, id="conflict-" + hashlib.sha256(identity.encode()).hexdigest()[:20],
                              temporal_overlap=(start, end), severity=severity,
                              activation_witness={f"{step}:{name}": value for step, name, value in sorted(literals)}))
    return result
