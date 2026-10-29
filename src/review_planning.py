"""Finite interpretive-question minimax planning. Suggestions never answer reviews."""

from __future__ import annotations

from itertools import product
from math import inf

from .certificates import certify
from .family_contract import parse_family
from .observation_strategy import PlanningLimit, optimal_strategy
from .phase4 import BoundHit, Budget, envelope, fields, integer
from .temporal import Formula, evaluate_formula

KINDS = {"predicate_alignment", "interpretation_selection", "assumption_confirmation", "span_grounding"}


def family_worlds(family, budget):
    parsed = parse_family(family)
    names = sorted(parsed["names"])
    h = family["horizon"]
    fixed = parsed["fixed"]
    free = [(t, n) for t in range(h) for n in names if (t, n) not in fixed]
    if len(free) > 12:
        raise BoundHit("family_completion_limit")
    result = []
    for values in product((False, True), repeat=len(free)):
        assignment = {**fixed, **dict(zip(free, values, strict=True))}
        trace = [{n: assignment[t, n] for n in names} for t in range(h)]
        for member in sorted(parsed["interpretations"], key=lambda m: m["id"]):
            for behavior in parsed["behaviors"]:
                budget.tick()
                if not all(evaluate_formula(a, trace) is True for a in member["assumptions"] + behavior["assumptions"]):
                    continue
                result.append(
                    {
                        "id": f"w{len(result)}",
                        "interpretation_id": member["id"],
                        "property": member["property"].to_dict(),
                        "trace": trace,
                    }
                )
                if len(result) > 4096:
                    raise BoundHit("world_limit")
    return result, parsed["findings"]


def plan_review(request):
    fields(
        request,
        "vle-review-plan/v1",
        {"worlds", "family", "questions", "max_states", "timeout_ms", "max_worlds"},
        {"questions"},
    )
    if ("worlds" in request) == ("family" in request):
        raise ValueError("one explicit worlds array or interpretation family required")
    budget = Budget(max_states=request.get("max_states", 65536), timeout_ms=request.get("timeout_ms", 5000))
    maximum = integer(request.get("max_worlds", 4096), 1, 4096, "max_worlds")
    result = {
        "plan": None,
        "optimal_in_scope": False,
        "worst_case_cost": None,
        "review_prompts": [],
        "certificates": [],
        "suggested_order_only": True,
        "default_costs_are_assumptions": True,
        "plan_optimality_independently_certified": False,
    }

    def finish(status, reason, **extra):
        if status != "unknown":
            budget.tick()
        return envelope(
            "vle-review-plan-result/v1",
            request,
            **{
                **result,
                **extra,
                "status": status,
                "reason": reason,
                "bounds": {**budget.to_dict(), "max_worlds": maximum, "max_questions": 12},
            },
        )

    try:
        worlds, findings = family_worlds(request["family"], budget) if "family" in request else (request["worlds"], [])
        if not isinstance(worlds, list) or not worlds:
            raise ValueError("worlds must be a nonempty array")
        if len(worlds) > maximum:
            raise BoundHit("world_limit")
        ids = []
        truth = []
        certificates = []
        for world in worlds:
            if (
                not isinstance(world, dict)
                or set(world) - {"id", "interpretation_id", "property", "trace"}
                or not {"id", "property", "trace"} <= set(world)
            ):
                raise ValueError("worlds carry explicit property/trace, never trusted verdict labels")
            if not isinstance(world["id"], str) or not world["id"] or world["id"] in ids:
                raise ValueError("invalid duplicate world id")
            ids.append(world["id"])
            budget.tick()
            f = Formula.from_dict(world["property"])
            value = evaluate_formula(f, world["trace"])
            if value is None:
                raise BoundHit("unknown_world_verdict")
            truth.append(value)
            certified = certify(
                {
                    "property": world["property"],
                    "horizon": len(world["trace"]),
                    "observations": world["trace"],
                    "binding": {"world_id": world["id"], "interpretation_id": world.get("interpretation_id")},
                }
            )
            if certified["status"] == "unknown":
                raise BoundHit("world_certificate_unknown")
            certificates.append(certified["certificate"])
            budget.tick()
        questions = request["questions"]
        if not isinstance(questions, list) or len(questions) > 12:
            raise ValueError("at most12 review questions")
        queries = []
        for q in questions:
            if (
                not isinstance(q, dict)
                or set(q) - {"id", "kind", "domain", "answers", "by_interpretation", "cost", "prompt"}
                or not {"id", "kind", "domain", "prompt"} <= set(q)
                or q["kind"] not in KINDS
                or not isinstance(q["id"], str)
                or not isinstance(q["prompt"], str)
                or not q["prompt"].strip()
            ):
                raise ValueError("invalid review question")
            domain = q["domain"]
            if (
                not isinstance(domain, list)
                or not 2 <= len(domain) <= 16
                or any(not isinstance(a, str) for a in domain)
                or len(set(domain)) != len(domain)
            ):
                raise ValueError("finite unique answer domain required")
            if ("answers" in q) == ("by_interpretation" in q):
                raise ValueError("one explicit answer partition is required")
            mapping = q.get("answers", {})
            if "answers" in q and (not isinstance(mapping, dict) or set(mapping) != set(ids)):
                raise ValueError("partition must cover every world exactly")
            answers = [
                q["by_interpretation"].get(w.get("interpretation_id")) if "by_interpretation" in q else mapping[w["id"]]
                for w in worlds
            ]
            if any(a not in domain for a in answers):
                raise ValueError("partition contains undefined answer")
            price = integer(q.get("cost", 1), 1, 100000, "review cost")
            queries.append({**q, "cost": price, "values": answers})
        if len({q["id"] for q in queries}) != len(queries):
            raise ValueError("duplicate question id")
        sorted_queries = sorted(queries, key=lambda q: q["id"])
        memo = {}

        def solve(state):
            budget.tick()
            if state in memo:
                return memo[state][0]
            if len({truth[i] for i in state}) == 1:
                memo[state] = (0, None)
                return 0
            best, choice = inf, None
            for q in sorted_queries:
                parts = [tuple(i for i in state if q["values"][i] == a) for a in q["domain"]]
                parts = [s for s in parts if s]
                if len(parts) < 2:
                    continue
                value = q["cost"] + max(solve(s) for s in parts)
                if value < best:
                    best, choice = value, q
            memo[state] = (best, choice)
            return best

        root = tuple(range(len(worlds)))
        cost = solve(root)
        if all(len(q["domain"]) == 2 for q in sorted_queries):
            # Differential cross-check using the unchanged Boolean factual kernel.
            masks = [sum(1 << i for i, a in enumerate(q["values"]) if a == q["domain"][1]) for q in sorted_queries]
            other = optimal_strategy(
                world_count=len(worlds),
                true_mask=sum(1 << i for i, t in enumerate(truth) if t),
                masks=masks,
                queries=[(0, q["id"], q["cost"]) for q in sorted_queries],
                max_states=budget.max_states,
                check_limit=budget.tick,
            )
            if other["cost"] != cost:
                raise BoundHit("minimax_kernel_disagreement")
        if cost == inf:
            for i in root:
                for j in root:
                    budget.tick()
                    if truth[i] != truth[j] and all(q["values"][i] == q["values"][j] for q in sorted_queries):
                        return finish(
                            "unresolvable",
                            "available_questions_cannot_separate_opposite_verdicts",
                            opposite_verdict_witnesses=[worlds[i], worlds[j]],
                            certificates=[certificates[i], certificates[j]],
                        )
            raise BoundHit("unresolvable_witness_missing")
        nodes = []
        node_ids = {}

        def emit(state):
            budget.tick()
            if state in node_ids:
                return node_ids[state]
            id = "n" + str(len(nodes))
            node_ids[state] = id
            node = {"id": id, "world_ids": [ids[i] for i in state], "worst_case_cost": memo[state][0]}
            nodes.append(node)
            choice = memo[state][1]
            if choice is None:
                node["conditional_verdict"] = "satisfied" if truth[state[0]] else "violated"
            else:
                node["question_id"] = choice["id"]
                node["branches"] = {
                    a: emit(tuple(i for i in state if choice["values"][i] == a))
                    for a in choice["domain"]
                    if any(choice["values"][i] == a for i in state)
                }
            return id

        plan = {"root": emit(root), "nodes": nodes}
        budget.tick()
        result.update(plan=plan, optimal_in_scope=True, worst_case_cost=cost, certificates=certificates)
        from .review import suggested_review_order

        result["review_prompts"] = suggested_review_order(plan, queries)
        return finish(
            "already_decided" if cost == 0 else "planned",
            "minimum_worst_case_review_cost_in_explicit_model",
            review_findings=findings,
            worlds=worlds,
            question_costs={q["id"]: q["cost"] for q in queries},
        )
    except (BoundHit, PlanningLimit) as exc:
        result.update(plan=None, optimal_in_scope=False, worst_case_cost=None, review_prompts=[], certificates=[])
        return finish("unknown", str(exc))
