"""Exact, bounded minimax observation planning for finite Boolean completions.

Queries reveal one true Boolean cell at a fixed, positive cost. This is neither
an expected-cost optimizer nor a model of noisy/legal/causal evidence collection.
"""
from __future__ import annotations

from itertools import product
from math import inf
from time import monotonic

from .audit_bundle import canonical_json, digest
from .temporal import MAX_STEPS, Formula, evaluate_formula, validate_formula, validate_trace


class _LimitReached(Exception):
    pass


def plan_evidence(request: dict) -> dict:
    allowed = {"schema", "property", "horizon", "assumptions", "observations", "queries",
               "max_completions", "max_states", "timeout_ms"}
    if (not isinstance(request, dict) or request.get("schema") != "vle-evidence-plan/v1"
            or set(request) - allowed or not {"property", "horizon"} <= set(request)):
        raise ValueError("invalid vle-evidence-plan/v1 request")
    property = Formula.from_dict(request["property"])
    names = validate_formula(property)
    assumptions = request.get("assumptions", [])
    if not isinstance(assumptions, list) or len(assumptions) > 64:
        raise ValueError("assumptions must be an array of at most 64 formulas")
    assumptions = [Formula.from_dict(item) for item in assumptions]
    for item in assumptions:
        names |= validate_formula(item)
    horizon = request["horizon"]
    if type(horizon) is not int or not 1 <= horizon <= MAX_STEPS:
        raise ValueError("invalid horizon")
    max_completions = request.get("max_completions", 1024)
    max_states = request.get("max_states", 10000)
    timeout_ms = request.get("timeout_ms", 5000)
    for name, value, upper in (("max_completions", max_completions, 4096),
                               ("max_states", max_states, 100000), ("timeout_ms", timeout_ms, 60000)):
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f"{name} must be between 1 and {upper}")
    observed = request.get("observations")
    if observed is not None:
        if not isinstance(observed, list):
            raise ValueError("observations must be a trace array")
        validate_trace(observed)
        if len(observed) > horizon:
            raise ValueError("observations extend beyond the horizon")
        for state in observed:
            names.update(state)
    fixed = {(t, name): value for t, state in enumerate(observed or ())
             for name, value in state.items() if value is not None}
    raw_queries = request.get("queries")
    queries = []
    if raw_queries is not None:
        if not isinstance(raw_queries, list) or len(raw_queries) > 12:
            raise ValueError("queries must be an array of at most 12 cells")
        for query in raw_queries:
            if not isinstance(query, dict) or set(query) != {"step", "atom", "cost"}:
                raise ValueError("a query requires step, atom, and cost")
            step, name, cost = query["step"], query["atom"], query["cost"]
            if type(step) is not int or not 0 <= step < horizon:
                raise ValueError("query step is outside the horizon")
            if not isinstance(name, str) or not name.strip() or len(name) > 512:
                raise ValueError("query atom requires 1-512 characters")
            if type(cost) is not int or not 1 <= cost <= 100000:
                raise ValueError("query costs must be positive integers at most 100000")
            if (step, name) in fixed:
                raise ValueError("cannot query an already observed Boolean cell")
            names.add(name)
            queries.append((step, name, cost))
        if len({(t, n) for t, n, _ in queries}) != len(queries):
            raise ValueError("duplicate query cell")
    if len(names) > 64 or any(not n.strip() or len(n) > 512 for n in names):
        raise ValueError("at most 64 atoms with 1-512 characters are supported")
    names = sorted(names)
    unknowns = [(t, n) for t in range(horizon) for n in names if (t, n) not in fixed]
    if raw_queries is None:
        queries = [(t, n, 1) for t, n in unknowns]
    queries.sort()
    request_hash = digest(canonical_json(request))
    result = {"schema": "vle-evidence-plan-result/v1", "request_sha256": request_hash,
              "legal_certification": False, "objective": "minimum_worst_case_sum_of_query_costs",
              "scope": {"horizon": horizon, "atoms": names, "unknown_cells": len(unknowns),
                        "queries": [{"step": t, "atom": n, "cost": c} for t, n, c in queries],
                        "max_completions": max_completions, "max_states": max_states,
                        "timeout_ms": timeout_ms, "semantics": "noiseless_finite_boolean_queries_v1"},
              "optimal_in_scope": False, "plan": None, "worst_case_cost": None}
    started = monotonic()
    states = {}
    feasible = []

    def check_limit():
        if (monotonic() - started) * 1000 >= timeout_ms:
            raise _LimitReached("timeout")

    indices = {cell: i for i, cell in enumerate(unknowns)}

    def trace(values):
        return [{n: fixed[t, n] if (t, n) in fixed else values[indices[t, n]] for n in names}
                for t in range(horizon)]

    try:
        bits = len(unknowns)
        if bits > max_completions.bit_length() or 2 ** bits > max_completions:
            raise _LimitReached("completion_space_limit")
        # Query limits also apply when candidates are inferred automatically.
        if len(queries) > 12:
            raise _LimitReached("query_space_limit")
        true_mask = 0
        for values in product((False, True), repeat=bits):
            check_limit()
            complete = trace(values)
            if not all(evaluate_formula(a, complete) is True for a in assumptions):
                continue
            if evaluate_formula(property, complete) is True:
                true_mask |= 1 << len(feasible)
            feasible.append(values)
        result["feasible_completions"] = len(feasible)
        if not feasible:
            return {**result, "status": "inconsistent_evidence", "reason": "no_admissible_completion"}
        masks = [sum(1 << i for i, a in enumerate(feasible) if a[indices[t, n]]) for t, n, _ in queries]

        def solve(state):
            check_limit()
            if state in states:
                return states[state][0]
            if len(states) >= max_states:
                raise _LimitReached("planning_state_limit")
            # Reserve the state before recursion so the bound counts live states.
            states[state] = (inf, None)
            positives = state & true_mask
            if not positives or positives == state:
                states[state] = (0, "true" if positives else "false")
                return 0
            best, choice = inf, None
            for index, ((_, _, cost), mask) in enumerate(zip(queries, masks, strict=True)):
                low, high = state & ~mask, state & mask
                if not low or not high:
                    continue  # This cell is already determined under this evidence.
                left = solve(low)
                if left == inf:
                    continue
                value = cost + max(left, solve(high))
                if value < best:
                    best, choice = value, index
            states[state] = best, choice
            return best

        root = (1 << len(feasible)) - 1
        cost = solve(root)
        result["planning_states"] = len(states)
        if cost == inf:
            # Find opposite-label completions indistinguishable by ALL allowed queries.
            seen = {}
            for i, assignment in enumerate(feasible):
                check_limit()
                signature = tuple(assignment[indices[t, n]] for t, n, _ in queries)
                truth = bool((1 << i) & true_mask)
                if signature in seen and seen[signature][0] != truth:
                    other = seen[signature][1]
                    return {**result, "status": "unresolvable", "reason": "indistinguishable_opposite_verdicts",
                            "indistinguishable_witnesses": [trace(other), trace(assignment)]}
                seen[signature] = truth, assignment
            raise RuntimeError("unresolvable state has no indistinguishability witness")
        nodes, ids = [], {}

        def emit(state):
            check_limit()
            if state in ids:
                return ids[state]
            node_id = f"n{len(ids)}"
            ids[state] = node_id
            value, choice = states[state]
            node = {"id": node_id, "completion_count": state.bit_count(), "worst_case_cost": value}
            nodes.append(node)
            if choice in {"true", "false"}:
                node["verdict"] = "definitely_true" if choice == "true" else "definitely_false"
            else:
                t, name, price = queries[choice]
                node.update({"step": t, "atom": name, "cost": price,
                             "on_false": emit(state & ~masks[choice]), "on_true": emit(state & masks[choice])})
            return node_id

        root_id = emit(root)
        return {**result, "status": "already_decided" if cost == 0 else "planned",
                "optimal_in_scope": True, "worst_case_cost": cost, "plan": {"root": root_id, "nodes": nodes}}
    except _LimitReached as exc:
        return {**result, "status": "unknown", "reason": str(exc), "planning_states": len(states)}
