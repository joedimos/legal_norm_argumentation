"""Interpretation-family analysis and factual evidence planning over finite worlds."""
from __future__ import annotations

from itertools import product
from math import inf
from time import monotonic

from .audit_bundle import canonical_json, digest
from .family_contract import family_fingerprint, grounding_fingerprint, parse_family
from .observation_strategy import PlanningLimit, optimal_strategy
from .temporal import evaluate_formula


def analyze_family(request: dict) -> dict:
    parsed = parse_family(request)
    members = []
    for interpretation in sorted(parsed["interpretations"], key=lambda item: item["id"]):
        for behavior in sorted(parsed["behaviors"], key=lambda item: item["id"] or ""):
            iid, bid = interpretation["id"], behavior["id"]
            key = iid if bid is None else f"{len(iid)}:{iid}{len(bid)}:{bid}"
            members.append({**interpretation, "id": key, "interpretation_id": iid, "behavior_id": bid,
                            "assumptions": interpretation["assumptions"] + behavior["assumptions"]})
    names = set(parsed["names"])
    horizon, fixed = request["horizon"], parsed["fixed"]
    limits = {"max_completions": (1024, 4096), "max_worlds": (8192, 65536),
              "max_states": (10000, 100000), "timeout_ms": (5000, 60000)}
    options = {}
    for key, (default, upper) in limits.items():
        value = request.get(key, default)
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f"{key} must be 1-{upper}")
        options[key] = value
    raw_queries = request.get("queries")
    queries = []
    if raw_queries is not None:
        if not isinstance(raw_queries, list) or len(raw_queries) > 12:
            raise ValueError("queries must be an array of at most 12 factual cells")
        for query in raw_queries:
            if not isinstance(query, dict) or set(query) != {"step", "atom", "cost"}:
                raise ValueError("queries require step, atom, cost; interpretations are not factual queries")
            t, name, price = query["step"], query["atom"], query["cost"]
            if type(t) is not int or not 0 <= t < horizon:
                raise ValueError("query step outside horizon")
            if not isinstance(name, str) or not name.strip() or len(name) > 512:
                raise ValueError("query atom requires 1-512 characters")
            if "behavior" in request and name.startswith("response:"):
                raise ValueError("hypothetical response actions are not factual queries")
            if type(price) is not int or not 1 <= price <= 100000:
                raise ValueError("query price requires a positive integer at most 100000")
            if (t, name) in fixed:
                raise ValueError("query cell is already observed")
            names.add(name)
            queries.append((t, name, price))
        if len(set((t, n) for t, n, _ in queries)) != len(queries):
            raise ValueError("duplicate query cell")
    if len(names) > 64:
        raise ValueError("at most 64 atoms are supported")
    names = sorted(names)
    unknowns = [(t, n) for t in range(horizon) for n in names if (t, n) not in fixed]
    indices = {cell: i for i, cell in enumerate(unknowns)}
    if raw_queries is None:
        queries = [(t, n, 1) for t, n in unknowns
                   if "behavior" not in request or not n.startswith("response:")]
    queries.sort()
    started = monotonic()
    worlds = []
    counts = {item["id"]: {"feasible": 0, "satisfied": 0, "violated": 0} for item in members}
    positive = 0
    disagreement = None
    example_true = example_false = None

    def check_limit():
        if (monotonic() - started) * 1000 >= options["timeout_ms"]:
            raise PlanningLimit("timeout")

    def trace(values):
        return [{n: fixed[t, n] if (t, n) in fixed else values[indices[t, n]] for n in names}
                for t in range(horizon)]

    def witness(world):
        index, values, truth = world
        return {"interpretation_id": members[index]["interpretation_id"],
                "behavior_id": members[index]["behavior_id"], "trace": trace(values), "satisfies": truth}

    result = {"schema": "vle-family-result/v1", "request_sha256": digest(canonical_json(request)),
              "family_fingerprint": family_fingerprint(request), "grounding_fingerprint": grounding_fingerprint(request),
              "review_findings": parsed["findings"], "reviews_authenticated": False,
              "deployment_authorized": False, "legal_certification": False,
              "scope": {"horizon": horizon, "atoms": names, "unknown_cells": len(unknowns),
                        "interpretation_ids": sorted({i["interpretation_id"] for i in members}),
                        "behavior_ids": [b["id"] for b in parsed["behaviors"]],
                        "model_ids": {i["id"]: {"interpretation_id": i["interpretation_id"],
                                               "behavior_id": i["behavior_id"]} for i in members},
                        "observations": request.get("observations"), "context": request["context"],
                        "queries": [{"step": t, "atom": n, "cost": c} for t, n, c in queries], **options,
                        "semantics": "reviewed-family-finite-completions-v1"},
              "analysis_basis": "explicit_caller_models_and_observation_assumptions"}

    def finish(analysis):
        return {**result, "status": "review_required" if parsed["findings"] else analysis["status"],
                "analysis": analysis}

    try:
        bits = len(unknowns)
        if bits > options["max_completions"].bit_length() or 2 ** bits > options["max_completions"]:
            raise PlanningLimit("completion_space_limit")
        if len(queries) > 12:
            raise PlanningLimit("query_space_limit")
        for values in product((False, True), repeat=bits):
            check_limit()
            complete = trace(values)
            at_assignment = []
            for index, member in enumerate(members):
                check_limit()
                if not all(evaluate_formula(a, complete) is True for a in member["assumptions"]):
                    continue
                if len(worlds) >= options["max_worlds"]:
                    raise PlanningLimit("world_space_limit")
                truth = evaluate_formula(member["property"], complete)
                world = index, values, truth
                if truth:
                    positive |= 1 << len(worlds)
                    example_true = example_true or world
                else:
                    example_false = example_false or world
                worlds.append(world)
                at_assignment.append(world)
                count = counts[member["id"]]
                count["feasible"] += 1
                count["satisfied" if truth else "violated"] += 1
            if disagreement is None and {w[2] for w in at_assignment} == {False, True}:
                disagreement = [next(w for w in at_assignment if w[2]), next(w for w in at_assignment if not w[2])]
        empty = [name for name, count in counts.items() if not count["feasible"]]
        base = {"members": counts, "world_count": len(worlds),
                "satisfying_witness": witness(example_true) if example_true is not None else None,
                "violating_witness": witness(example_false) if example_false is not None else None}
        if empty:
            return finish({**base, "status": "inconsistent_family", "infeasible_interpretations": sorted({m["interpretation_id"] for m in members if m["id"] in empty}),
                           "infeasible_models": empty, "plan": None, "reason": "an_interpretation_has_no_admissible_completion"})
        if disagreement:
            # The same complete facts cannot be separated by ANY factual query.
            first, second = (members[w[0]] for w in disagreement)
            disagreement_type = ("interpretation_disagreement" if first["behavior_id"] == second["behavior_id"]
                                 else "grounding_disagreement" if first["interpretation_id"] == second["interpretation_id"]
                                 else "model_disagreement")
            return finish({**base, "status": disagreement_type, "plan": None,
                           "indistinguishable_witnesses": [witness(w) for w in disagreement],
                           "reason": "opposite_verdicts_under_identical_complete_facts"})
        masks = [sum(1 << i for i, (_, v, _) in enumerate(worlds) if v[indices[t, n]]) for t, n, _ in queries]
        strategy = optimal_strategy(world_count=len(worlds), true_mask=positive, masks=masks,
                                    queries=queries, max_states=options["max_states"], check_limit=check_limit)
        if strategy["cost"] == inf:
            seen = {}
            for world in worlds:
                check_limit()
                signature = tuple(world[1][indices[t, n]] for t, n, _ in queries)
                previous = seen.get(signature)
                if previous is not None and previous[2] != world[2]:
                    factual_cells = [cell for cell in unknowns if not cell[1].startswith("response:")]
                    same_facts = all(previous[1][indices[cell]] == world[1][indices[cell]] for cell in factual_cells)
                    old, new = members[previous[0]], members[world[0]]
                    state = "unresolvable_queries"
                    if "behavior" in request and same_facts:
                        state = ("grounding_disagreement" if old["interpretation_id"] == new["interpretation_id"]
                                 and old["behavior_id"] != new["behavior_id"] else "behavior_uncertainty")
                    return finish({**base, "status": state, "plan": None,
                                   "indistinguishable_witnesses": [witness(previous), witness(world)],
                                   "reason": "allowed_queries_cannot_resolve_factual_uncertainty"})
                seen[signature] = world
            raise RuntimeError("unresolvable family has no witness")
        status = "clarification_needed" if strategy["cost"] else "robust_satisfied" if positive else "robust_violated"
        return finish({**base, "status": status, "plan": strategy["plan"], "worst_case_cost": strategy["cost"],
                       "optimal_in_scope": True, "planning_states": strategy["states"]})
    except PlanningLimit as exc:
        return finish({"status": "unknown", "reason": str(exc), "plan": None, "optimal_in_scope": False})


def run_family_benchmark(corpus):
    if (not isinstance(corpus, dict) or set(corpus) != {"schema", "annotation_status", "cases"}
            or corpus["schema"] != "vle-family-corpus/v1" or corpus["annotation_status"] != "synthetic_unreviewed"):
        raise ValueError("expected an explicitly synthetic_unreviewed family corpus")
    cases = corpus["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 128:
        raise ValueError("corpus requires 1-128 cases")
    rows, seen = [], set()
    for case in cases:
        if (not isinstance(case, dict) or set(case) != {"id", "request", "expected_analysis_status"}
                or not isinstance(case["id"], str) or not case["id"].strip() or case["id"] in seen
                or not isinstance(case["request"], dict) or not isinstance(case["expected_analysis_status"], str)):
            raise ValueError("invalid or duplicate case")
        if case["request"].get("reviews"):
            raise ValueError("synthetic corpus must not pretend to contain independent approvals")
        seen.add(case["id"])
        result = analyze_family(case["request"])
        actual = result["analysis"]["status"]
        rows.append({"id": case["id"], "actual": actual, "expected": case["expected_analysis_status"],
                     "passed": actual == case["expected_analysis_status"], "result": result})
    return {"schema": "vle-family-benchmark/v1", "annotation_status": corpus["annotation_status"],
            "corpus_sha256": digest(canonical_json(corpus)), "case_count": len(rows),
            "all_passed": all(row["passed"] for row in rows), "cases": rows, "legal_certification": False,
            "scope": "synthetic_semantic_regression_not_legal_accuracy_or_expert_validation"}
