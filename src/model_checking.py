"""Independent exhaustive and SMT encodings of bounded finite-trace validity.

Verification checks A => property for ALL traces of exactly `horizon` states.
Inconsistent assumptions are reported separately, never as a vacuous proof.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from functools import cache
from itertools import product
from time import monotonic

from .temporal import MAX_STEPS, Formula, evaluate_formula, validate_formula, validate_trace


@dataclass(frozen=True)
class VerificationResult:
    status: str
    backend: str
    horizon: int
    atoms: tuple[str, ...]
    reason: str
    checked_traces: int = 0
    counterexample: tuple[dict[str, bool], ...] | None = None
    backend_version: str | None = None
    semantics: str = "LTLf-complete-valuations-v1"
    observation_constraints: tuple[tuple[int, str, bool], ...] = ()
    unknown_cells: int | None = None

    def to_dict(self):
        return asdict(self)


def verify(
    property: Formula, *, horizon: int, assumptions: Sequence[Formula] = (),
    backend: str = "exhaustive", max_traces: int = 65536, timeout_ms: int = 5000,
    observations: Sequence[Mapping[str, bool | None]] | None = None,
) -> VerificationResult:
    if type(horizon) is not int or not 1 <= horizon <= MAX_STEPS:
        raise ValueError(f"horizon must be 1-{MAX_STEPS}")
    if type(timeout_ms) is not int or not 1 <= timeout_ms <= 60000:
        raise ValueError("timeout_ms must be 1-60000")
    if type(max_traces) is not int or not 1 <= max_traces <= 1_000_000:
        raise ValueError("max_traces must be 1-1000000")
    if len(assumptions) > 64:
        raise ValueError("at most 64 assumptions are supported")
    names = validate_formula(property)
    for assumption in assumptions:
        names |= validate_formula(assumption)
    if observations is not None:
        if not isinstance(observations, (list, tuple)):
            raise ValueError("observations must be a trace array")
        validate_trace(observations)
        if len(observations) > horizon:
            raise ValueError("observations extend beyond the verification horizon")
        for state in observations:
            for name in state:
                if not name.strip() or len(name) > 512:
                    raise ValueError("observation atoms require 1-512 characters")
                names.add(name)
    fixed = {(name, t): value for t, state in enumerate(observations or ())
             for name, value in state.items() if value is not None}
    names = tuple(sorted(names))
    if len(names) > 64:
        raise ValueError("at most 64 atoms are supported")
    if backend not in {"exhaustive", "z3"}:
        raise ValueError("backend must be exhaustive or z3")
    constraints = tuple(sorted((t, name, value) for (name, t), value in fixed.items()))
    unknown_cells = len(names) * horizon - len(fixed)
    if backend == "z3":
        return replace(_verify_z3(property, horizon, assumptions, names, timeout_ms, fixed),
                       observation_constraints=constraints, unknown_cells=unknown_cells)

    def result(status, reason, checked=0, trace=None):
        return VerificationResult(status, backend, horizon, names, reason, checked, trace,
                                  observation_constraints=constraints, unknown_cells=unknown_cells)

    unknowns = [(name, t) for t in range(horizon) for name in names if (name, t) not in fixed]
    bits = len(unknowns)
    if bits > max_traces.bit_length() or 2 ** bits > max_traces:
        return result("unknown", "state_space_limit")
    started = monotonic()
    feasible = 0
    checked = 0
    for values in product((False, True), repeat=bits):
        if (monotonic() - started) * 1000 >= timeout_ms:
            return result("unknown", "timeout", checked)
        assignment = {**fixed, **dict(zip(unknowns, values, strict=True))}
        trace = tuple({name: assignment[name, t] for name in names} for t in range(horizon))
        checked += 1
        if not all(evaluate_formula(a, trace) is True for a in assumptions):
            continue
        feasible += 1
        if evaluate_formula(property, trace) is False:
            return result("counterexample", "property_false", checked, trace)
    if not feasible:
        return result("inconsistent_assumptions", "no_admissible_trace", checked)
    return result("verified_bounded", "exhaustive_within_declared_horizon", checked)


def _verify_z3(property, horizon, assumptions, names, timeout_ms, fixed):
    try:
        import z3
    except ImportError:
        return VerificationResult("unavailable", "z3", horizon, names, "install_the_verification_extra")
    version = z3.get_version_string()
    def node_count(node):
        return 1 + sum(node_count(arg) for arg in node.args)
    encoding_cost = sum(node_count(f) for f in [property, *assumptions]) * horizon * horizon
    if encoding_cost > 250_000:
        return VerificationResult("unknown", "z3", horizon, names, "encoding_budget", backend_version=version)

    def result(status, reason, trace=None):
        return VerificationResult(status, "z3", horizon, names, reason,
                                  counterexample=trace, backend_version=version)

    # Separate solver context avoids sharing mutable Z3 state between requests.
    context = z3.Context()
    variables = {(name, t): z3.Bool(f"a{index}_t{t}", ctx=context)
                 for index, name in enumerate(names) for t in range(horizon)}
    started = monotonic()

    @cache
    def encode(node, t):
        if (monotonic() - started) * 1000 >= timeout_ms:
            raise TimeoutError("encoding_timeout")
        op, args = node.op, node.args
        if op in {"true", "false"}:
            return z3.BoolVal(op == "true", ctx=context)
        if op == "atom":
            return variables[node.name, t]
        if op == "not":
            return z3.Not(encode(args[0], t))
        if op in {"and", "or"}:
            return (z3.And if op == "and" else z3.Or)(encode(args[0], t), encode(args[1], t))
        if op in {"next", "weak_next"}:
            return (z3.BoolVal(op == "weak_next", ctx=context) if t + 1 == horizon
                    else encode(args[0], t + 1))
        if op in {"eventually", "always"}:
            return (z3.Or if op == "eventually" else z3.And)([encode(args[0], i) for i in range(t, horizon)])
        # Independent witness-based encoding, unlike the evaluator recurrence.
        return z3.Or([z3.And(encode(args[1], j),
                            (z3.And([encode(args[0], k) for k in range(t, j)])
                             if j > t else z3.BoolVal(True, ctx=context)))
                      for j in range(t, horizon)])

    solver = z3.Solver(ctx=context)
    try:
        solver.add([variables[key] == value for key, value in fixed.items()])
        solver.add([encode(a, 0) for a in assumptions])
        negated_property = z3.Not(encode(property, 0))
    except TimeoutError:
        return result("unknown", "encoding_timeout")

    def check():
        remaining = timeout_ms - int((monotonic() - started) * 1000)
        if remaining <= 0:
            return None
        solver.set(timeout=remaining)
        return solver.check()

    feasible = check()
    if feasible is None or feasible == z3.unknown:
        return result("unknown", "timeout" if feasible is None else solver.reason_unknown())
    if feasible == z3.unsat:
        return result("inconsistent_assumptions", "no_admissible_trace")
    solver.add(negated_property)
    answer = check()
    if answer is None or answer == z3.unknown:
        return result("unknown", "timeout" if answer is None else solver.reason_unknown())
    if answer == z3.unsat:
        return result("verified_bounded", "unsat_negated_property_within_declared_horizon")
    model = solver.model()
    trace = tuple({name: z3.is_true(model.eval(variables[name, t], model_completion=True))
                   for name in names} for t in range(horizon))
    # Replay solver evidence through the independent interpreter before release.
    if (evaluate_formula(property, trace) is not False or not all(
        evaluate_formula(a, trace) is True for a in assumptions
    ) or any(trace[t][name] != value for (name, t), value in fixed.items())):
        return result("unknown", "counterexample_replay_failed")
    return result("counterexample", "property_false", trace)


def verify_request(payload: Mapping) -> dict:
    if not isinstance(payload, Mapping) or set(payload) - {
        "property", "horizon", "assumptions", "backend", "max_traces", "timeout_ms", "observations"
    }:
        raise ValueError("invalid verification request fields")
    if "property" not in payload or "horizon" not in payload:
        raise ValueError("property and horizon are required")
    assumptions = payload.get("assumptions", [])
    if not isinstance(assumptions, list) or len(assumptions) > 64:
        raise ValueError("assumptions must be a list of at most 64 formulas")
    return verify(Formula.from_dict(payload["property"]), horizon=payload["horizon"],
                  assumptions=[Formula.from_dict(a) for a in assumptions],
                  backend=payload.get("backend", "exhaustive"),
                  max_traces=payload.get("max_traces", 65536),
                  timeout_ms=payload.get("timeout_ms", 5000),
                  observations=payload.get("observations")).to_dict()
