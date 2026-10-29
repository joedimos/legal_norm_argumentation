"""Optional finite-trace SAT oracle with guarded assumptions and unsat cores."""

from __future__ import annotations

from functools import cache

from .phase4 import BoundHit


class Z3Oracle:
    def __init__(self, hard, assumptions, names, horizon, budget):
        try:
            import z3
        except ImportError as exc:
            raise BoundHit("z3_unavailable_install_verification_extra") from exc
        self.z3, self.budget, self.horizon = z3, budget, horizon
        self.context = z3.Context()
        self.variables = {
            (name, t): z3.Bool(f"p{i}_{t}", ctx=self.context) for i, name in enumerate(names) for t in range(horizon)
        }
        self.names = names
        self.solver = z3.Solver(ctx=self.context)
        self.labels = [z3.Bool(f"assumption_{i}", ctx=self.context) for i in range(len(assumptions))]

        @cache
        def encode(f, t):
            budget.tick()
            op, a = f.op, f.args
            if op == "atom":
                return self.variables[f.name, t]
            if op in {"true", "false"}:
                return z3.BoolVal(op == "true", ctx=self.context)
            if op == "not":
                return z3.Not(encode(a[0], t))
            if op in {"and", "or"}:
                return (z3.And if op == "and" else z3.Or)(encode(a[0], t), encode(a[1], t))
            if op in {"next", "weak_next"}:
                return encode(a[0], t + 1) if t + 1 < horizon else z3.BoolVal(op == "weak_next", ctx=self.context)
            if op in {"always", "eventually"}:
                return (z3.And if op == "always" else z3.Or)([encode(a[0], i) for i in range(t, horizon)])
            return z3.Or(
                [z3.And([encode(a[1], j)] + [encode(a[0], k) for k in range(t, j)]) for j in range(t, horizon)]
            )

        self.solver.add([encode(f, 0) for f in hard])
        for label, f in zip(self.labels, assumptions, strict=True):
            self.solver.add(z3.Implies(label, encode(f, 0)))
        self.last_core = []

    def check(self, retained):
        from time import monotonic

        self.budget.tick()
        remaining = self.budget.timeout_ms - int((monotonic() - self.budget.started) * 1000)
        if remaining <= 0:
            raise BoundHit("timeout")
        self.solver.set(timeout=remaining)
        answer = self.solver.check(*[self.labels[i] for i in sorted(retained)])
        self.budget.tick()
        if answer == self.z3.unknown:
            raise BoundHit("z3_unknown:" + self.solver.reason_unknown())
        if answer == self.z3.unsat:
            core = {str(v) for v in self.solver.unsat_core()}
            self.last_core = [i for i, label in enumerate(self.labels) if str(label) in core]
            return False, None
        m = self.solver.model()
        trace = [
            {name: self.z3.is_true(m.eval(self.variables[name, t], model_completion=True)) for name in self.names}
            for t in range(self.horizon)
        ]
        return True, trace
