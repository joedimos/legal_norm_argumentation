"""Fixed planted constraints and explicit unavailable empirical inventory."""

from __future__ import annotations

from .obstruction import obstruction
from .temporal import atom, combine


def equal(a, b):
    return combine("or", combine("and", a, b), combine("and", combine("not", a), combine("not", b)))


def planted(size, seed, kind):
    names = [atom(f"p{seed}_{i}") for i in range(size)]
    forms = [equal(names[i], names[i + 1]) for i in range(size - 1)] + [combine("not", equal(names[-1], names[0]))]
    cells = [{"id": f"n{i}", "property": f.to_dict()} for i, f in enumerate(forms)]
    refs = {}
    if kind == "reference_loop":
        for i, c in enumerate(cells):
            ref = "ref" + str((i + 1) % size)
            c["references"] = [ref]
            refs[ref] = {"source_id": cells[(i + 1) % size]["id"], "text": "synthetic"}
    return {
        "schema": "vle-obstruction/v1",
        "cells": cells,
        "horizon": 1,
        "assume_reviewed_demo": True,
        "reference_index": refs,
    }


def run_study(backend="exhaustive"):
    rows = []
    for kind, size in [("triangle", 3), ("four_cycle", 4), ("exception_scope_loop", 3), ("reference_loop", 3)]:
        for seed in range(8):
            request = planted(size, seed, kind)
            r = obstruction(request, backend=backend)
            rows.append(
                {
                    "kind": kind,
                    "seed": seed,
                    "pair_conflicts": sum(not p["consistent"] for p in r["pair_checks"]),
                    "status": r["status"],
                    "minimal_family": r["minimal_family"],
                    "expected_family": [c["id"] for c in request["cells"]],
                    "existing_pairwise_applicable": r.get("existing_pairwise_applicable"),
                }
            )
    p, q = atom("p"), atom("q")
    chain = {
        "schema": "vle-obstruction/v1",
        "horizon": 1,
        "assume_reviewed_demo": True,
        "cells": [{"id": str(i), "property": f.to_dict()} for i, f in enumerate([p, equal(p, q), combine("not", q)])],
    }
    counterexample = obstruction(chain, backend=backend)
    return {
        "schema": "vle-obstruction-study/v1",
        "synthetic": True,
        "review_status": "unreviewed",
        "backend": backend,
        "F1": {
            "cases": len(rows),
            "passed": sum(
                r["status"] == "cyclic_obstruction"
                and r["pair_conflicts"] == 0
                and r["minimal_family"] == r["expected_family"]
                for r in rows
            ),
            "rows": rows,
        },
        "F2_unqualified_pair_SAT_sufficiency": False,
        "acyclic_counterexample": counterexample,
        "F3": {
            "status": "not_run_missing_reviewed_data",
            "inventory": "no local ingested SQLite/snapshot corpus located",
            "eligible_documents": None,
            "eligible_norms": None,
            "cyclic_structures": None,
            "obstructions": None,
        },
        "decision_authorized": False,
        "legal_certification": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_study(), indent=2))
