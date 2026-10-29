"""Frozen finite-oracle simulation; no people, inferred priors or average-only claims."""

from __future__ import annotations

import random

from .review_planning import plan_review
from .temporal import Formula, evaluate_formula


def simulate(corpus):
    rows = []
    for c in corpus["cases"]:
        request = c["request"]
        r = plan_review(request)
        worlds = request["worlds"]
        queries = request["questions"]
        truths = [evaluate_formula(Formula.from_dict(w["property"]), w["trace"]) for w in worlds]

        def fixed(order, actual, worlds=worlds, truths=truths):
            remaining = list(range(len(worlds)))
            cost = count = 0
            for q in order:
                if len({truths[i] for i in remaining}) == 1:
                    break
                answer = q["answers"][worlds[actual]["id"]]
                remaining = [i for i in remaining if q["answers"][worlds[i]["id"]] == answer]
                cost += q.get("cost", 1)
                count += 1
            return {"questions": count, "cost": cost, "determinate": len({truths[i] for i in remaining}) == 1}

        distributions = {"fixed": [fixed(queries, i) for i in range(len(worlds))], "random": [], "minimax": []}
        for seed in range(8):
            order = list(queries)
            random.Random(seed).shuffle(order)
            distributions["random"].append({"seed": seed, "worlds": [fixed(order, i) for i in range(len(worlds))]})
        if r["status"] in {"planned", "already_decided"}:
            nodes = {n["id"]: n for n in r["plan"]["nodes"]}
            byq = {q["id"]: q for q in queries}
            for w in worlds:
                node = nodes[r["plan"]["root"]]
                cost = count = 0
                while "question_id" in node:
                    q = byq[node["question_id"]]
                    cost += q.get("cost", 1)
                    count += 1
                    node = nodes[node["branches"][q["answers"][w["id"]]]]
                distributions["minimax"].append({"questions": count, "cost": cost, "determinate": True})
        rows.append(
            {
                "id": c["id"],
                "status": r["status"],
                "equal_cost": len({q.get("cost", 1) for q in queries}) <= 1,
                "optimal_worst_case_cost": r["worst_case_cost"],
                "distributions": distributions,
            }
        )
    return {
        "schema": "vle-review-simulation/v1",
        "synthetic": True,
        "review_status": "unreviewed",
        "real_reviewers": 0,
        "rows": rows,
        "decision_authorized": False,
        "legal_certification": False,
        "scope": "finite_noiseless_oracles_not_real_reviewers",
    }


if __name__ == "__main__":
    import json
    from pathlib import Path

    print(json.dumps(simulate(json.loads(Path("benchmarks/review-planning-v1.json").read_text())), indent=2))
