"""Subset-minimal correction/unsat diagnosis, not automatic legal correction.

The expected-verdict formula is supplied by a reviewer. A correction witnesses
existential satisfiability after relaxation, not universal legal compliance.
"""

from __future__ import annotations

from itertools import combinations, product

from .phase4 import BoundHit, Budget, envelope, fields, integer
from .temporal import Formula, evaluate_formula, validate_formula

KINDS = {"background_fact", "predicate_grounding", "interpretation_selection", "horizon_closure", "boolean_completion"}


def assumption_report(request, *, backend=None, max_mcs=None):
    fields(
        request,
        "vle-assumptions/v1",
        {
            "expected",
            "hard",
            "assumptions",
            "horizon",
            "backend",
            "max_mcs",
            "max_mus",
            "max_states",
            "max_traces",
            "timeout_ms",
        },
        {"expected", "assumptions", "horizon"},
    )
    backend = backend or request.get("backend", "exhaustive")
    if backend not in {"exhaustive", "z3"}:
        raise ValueError("invalid backend")
    horizon = integer(request["horizon"], 1, 33, "horizon")
    cap = integer(max_mcs if max_mcs is not None else request.get("max_mcs", 32), 1, 256, "max_mcs")
    mus_cap = integer(request.get("max_mus", 32), 1, 256, "max_mus")
    trace_cap = integer(request.get("max_traces", 65536), 1, 1000000, "max_traces")
    budget = Budget(max_states=request.get("max_states", 65536), timeout_ms=request.get("timeout_ms", 5000))
    raw = request["assumptions"]
    if not isinstance(raw, list) or len(raw) > 16:
        raise ValueError("assumptions require 0-16 entries")
    parsed = []
    for item in raw:
        if (
            not isinstance(item, dict)
            or set(item) != {"id", "kind", "formula", "source_span", "description"}
            or not isinstance(item["id"], str)
            or not item["id"].strip()
            or item["kind"] not in KINDS
            or not isinstance(item["description"], str)
            or not item["description"].strip()
        ):
            raise ValueError("invalid assumption")
        span = item["source_span"]
        if span is not None:
            if (
                not isinstance(span, dict)
                or set(span) != {"source_id", "start", "end", "quote"}
                or not isinstance(span["source_id"], str)
                or not isinstance(span["quote"], str)
            ):
                raise ValueError("invalid assumption source span")
            integer(span["start"], 0, 1000000, "start")
            integer(span["end"], span["start"], 1000000, "end")
            if span["end"] - span["start"] != len(span["quote"]):
                raise ValueError("quote length must match half-open span")
        parsed.append(Formula.from_dict(item["formula"]))
    if len({a["id"] for a in raw}) != len(raw):
        raise ValueError("duplicate assumption id")
    # Sorting removes incidental caller ordering from ranked review entry points.
    order = sorted(range(len(raw)), key=lambda i: raw[i]["id"])
    raw, parsed = [raw[i] for i in order], [parsed[i] for i in order]
    hard_raw = request.get("hard", [])
    if not isinstance(hard_raw, list) or len(hard_raw) > 32:
        raise ValueError("hard constraints require 0-32 formulas")
    hard = [Formula.from_dict(f) for f in hard_raw] + [Formula.from_dict(request["expected"])]
    names = tuple(sorted(set().union(*(validate_formula(f) for f in [*hard, *parsed]))))
    if len(names) > 64:
        raise ValueError("at most 64 atoms")
    mcs, mus, table, core = [], [], {}, []
    all_ids = frozenset(range(len(raw)))

    def result(status, reason, truncated=False):
        from .review import assumption_review_prompts

        return envelope(
            "vle-assumption-report/v1",
            request,
            status=status,
            reason=reason,
            backend=backend,
            horizon=horizon,
            assumptions=raw,
            expected=request["expected"],
            minimal_corrections=mcs if not truncated else [],
            minimal_unsatisfiable=mus if not truncated else [],
            solver_unsat_core=[raw[i]["id"] for i in core],
            truncated=truncated,
            minimality_established=not truncated,
            review_prompts=assumption_review_prompts(mcs) if not truncated else [],
            bounds={
                **budget.to_dict(),
                "max_assumptions": 16,
                "max_mcs": cap,
                "max_mus": mus_cap,
                "max_traces": trace_cap,
            },
            semantics="bounded_existential_restoration_not_universal_compliance",
        )

    try:
        if backend == "z3":
            from .assumptions_z3 import Z3Oracle

            oracle = Z3Oracle(hard, parsed, names, horizon, budget)
            check = oracle.check
        else:
            if len(names) * horizon > trace_cap.bit_length() or 2 ** (len(names) * horizon) > trace_cap:
                raise BoundHit("trace_limit")

            def check(retained):
                for values in product((False, True), repeat=len(names) * horizon):
                    budget.tick()
                    trace = [{name: values[t * len(names) + i] for i, name in enumerate(names)} for t in range(horizon)]
                    if all(evaluate_formula(f, trace) is True for f in [*hard, *(parsed[i] for i in sorted(retained))]):
                        return True, trace
                return False, None

        for size in range(len(raw) + 1):
            for subset in combinations(range(len(raw)), size):
                budget.tick()
                retained = frozenset(subset)
                sat, witness = check(retained)
                if sat and not all(evaluate_formula(f, witness) is True for f in [*hard, *(parsed[i] for i in subset)]):
                    raise BoundHit("witness_replay_failed")
                table[retained] = (sat, witness)
        if backend == "z3":
            check(all_ids)
            core = oracle.last_core
        if not table[frozenset()][0]:
            # Expected outcome conflicts with hard constraints even with all
            # soft assumptions removed. Do not invent a statutory fix.
            mus.append({"assumption_ids": [], "relative_to_hard_and_expected": True})
            return result("hard_conflict", "no_relaxation_of_declared_assumptions_restores_expected")
        for subset in sorted(table, key=lambda s: (len(s), sorted(s))):
            budget.tick()
            sat, _ = table[subset]
            if not sat and all(table[subset - {i}][0] for i in subset):
                mus.append(
                    {"assumption_ids": [raw[i]["id"] for i in sorted(subset)], "relative_to_hard_and_expected": True}
                )
                if len(mus) >= mus_cap:
                    raise BoundHit("mus_cap")
            removed = all_ids - subset
            if sat and all(not table[subset | {i}][0] for i in removed):
                ids = [raw[i]["id"] for i in sorted(removed)]
                mcs.append(
                    {
                        "assumption_ids": ids,
                        "assumptions": [raw[i] for i in sorted(removed)],
                        "witness": table[subset][1],
                        "verdict_change": "inconsistent_to_expected_feasible"
                        if not table[all_ids][0]
                        else "expected_already_feasible",
                    }
                )
                if len(mcs) >= cap:
                    raise BoundHit("mcs_cap")
        mcs.sort(key=lambda row: (len(row["assumption_ids"]), row["assumption_ids"]))
        return result(
            "expected_feasible" if table[all_ids][0] else "corrections_found", "complete_within_recorded_bounds"
        )
    except BoundHit as exc:
        return result("unknown", str(exc), truncated=True)
