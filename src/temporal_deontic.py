"""Explicit discrete-time achievement/maintenance norms over observed traces.

Applicability is sampled once at start; exceptions do not retroactively cancel
an obligation. Windows are inclusive. This is an operational deontic fragment,
not an axiomatization of all legal modalities or recurring obligations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .normative_model import Modality, NormInterpretation
from .temporal import Formula, atom, combine, evaluate_formula, validate_trace


def _fold(op, values):
    if not values:
        return Formula("true" if op == "and" else "false")
    if len(values) == 1:
        return values[0]
    midpoint = len(values) // 2
    return combine(op, _fold(op, values[:midpoint]), _fold(op, values[midpoint:]))


@dataclass(frozen=True)
class TemporalNorm:
    norm: NormInterpretation
    start: int
    end: int
    mode: str = "achievement"
    activation_date: str | None = None

    def __post_init__(self):
        if type(self.start) is not int or type(self.end) is not int or not 0 <= self.start <= self.end <= 32:
            raise ValueError("norm window must satisfy 0 <= start <= end <= 32")
        if self.mode not in {"achievement", "maintenance"}:
            raise ValueError("mode must be achievement or maintenance")
        if self.norm.deadline:
            raise ValueError("translate the textual deadline into a reviewed window and clear deadline first")
        if self.norm.modality == Modality.RECOMMENDATION:
            raise ValueError("recommendations are not hard temporal norms")
        authority = self.norm.authority
        if (authority.effective_from or authority.effective_to) and not self.activation_date:
            raise ValueError("dated norms require an explicit activation_date")
        authority.is_effective(self.activation_date)  # validate the supplied date
        if len(self.norm.conditions) + len(self.norm.exceptions) > 64:
            raise ValueError("at most 64 activation predicates are supported")

    @property
    def action_atom(self) -> str:
        # Namespaced tuple avoids subject/object/action collisions and free-text aliasing.
        return "action:" + json.dumps([self.norm.subject, self.norm.action, self.norm.object],
                                      ensure_ascii=False, separators=(",", ":"))

    def guard(self) -> Formula:
        if self.norm.authority.is_effective(self.activation_date) is False:
            return Formula("false")
        predicates = []
        for predicate in self.norm.conditions:
            value = atom("fact:" + predicate.name)
            predicates.append(combine("not", value) if predicate.negated else value)
        for predicate in self.norm.exceptions:
            value = atom("fact:" + predicate.name)
            predicates.append(value if predicate.negated else combine("not", value))
        return _fold("and", predicates)

    def constraint(self) -> Formula:
        """Return a hard constraint; verification horizon must be > end.

        Permission contributes no duty. Its conflicts remain a separate normative
        analysis, because strong permission and prohibition can disagree even when
        their hard action constraints are satisfiable.
        """
        if self.norm.modality == Modality.PERMISSION:
            return Formula("true")
        effect = atom(self.action_atom)
        maintenance = self.mode == "maintenance" or self.norm.modality == Modality.PROHIBITION
        if self.norm.modality == Modality.PROHIBITION:
            effect = combine("not", effect)
        requirement = effect
        for _ in range(self.end - self.start):
            requirement = combine("and" if maintenance else "or", effect, combine("next", requirement))
        result = combine("or", combine("not", self.guard()), requirement)
        for _ in range(self.start):
            result = combine("next", result)
        return result

    def evaluate(self, trace, *, closed: bool = True) -> dict:
        validate_trace(trace)
        if type(closed) is not bool:
            raise ValueError("closed must be boolean")

        def finding(status, explanation):
            return {"norm_id": self.norm.id, "status": status, "explanation": explanation,
                    "window": [self.start, self.end], "mode": self.mode, "action_atom": self.action_atom}

        if len(trace) <= self.start:
            return finding("unknown" if closed else "pending", "activation_state_not_observed")
        active = evaluate_formula(self.guard(), trace, self.start)
        if active is False:
            return finding("inactive", "condition_false_exception_true_or_outside_effective_dates")
        if active is None:
            return finding("unknown", "activation_facts_missing")
        if self.norm.modality == Modality.PERMISSION:
            return finding("permitted", "permission_does_not_impose_a_duty")
        values = [state.get(self.action_atom) for state in trace[self.start:self.end + 1]]
        maintenance = self.mode == "maintenance" or self.norm.modality == Modality.PROHIBITION
        if self.norm.modality == Modality.PROHIBITION:
            values = [None if v is None else not v for v in values]
        if maintenance and False in values:
            return finding("violated", "maintenance_counterexample_observed")
        if not maintenance and True in values:
            return finding("satisfied", "achievement_witness_observed")
        if len(trace) <= self.end:
            return finding("unknown" if closed else "pending", "window_not_fully_observed")
        if None in values:
            return finding("unknown", "action_observations_missing")
        return finding("satisfied" if maintenance else "violated", "window_complete")


def verify_temporal_norm(norm: TemporalNorm, *, horizon: int, **kwargs):
    from .model_checking import verify

    if horizon <= norm.end:
        raise ValueError("verification horizon must include the entire normative window")
    return verify(norm.constraint(), horizon=horizon, **kwargs)
