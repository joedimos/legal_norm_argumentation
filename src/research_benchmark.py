"""Versioned synthetic regression corpus; no claims of real-world legal accuracy."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .model_checking import verify_request
from .preferences import Preference, analyze_preferences
from .research import run_research
from .temporal import Formula, evaluate_formula


def run_benchmarks(path: str | Path, *, backend: str = "exhaustive") -> dict:
    raw = Path(path).read_bytes()
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("benchmark corpus exceeds 4 MiB")
    corpus = json.loads(raw)
    if corpus.get("schema") != "vle-benchmark/v1":
        raise ValueError("unsupported benchmark schema")
    cases = corpus.get("cases", [])
    if not isinstance(cases, list) or not 1 <= len(cases) <= 1000:
        raise ValueError("benchmark requires 1-1000 cases")
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("benchmark case ids must be unique")
    rows = []
    groups = defaultdict(lambda: {"passed": 0, "total": 0})
    for case in cases:
        kind = case["kind"]
        request = case["input"]
        if kind == "temporal":
            actual = evaluate_formula(Formula.from_dict(request["formula"]), request["trace"])
        elif kind == "verification":
            actual = verify_request({**request, "backend": backend})["status"]
        elif kind == "research":
            actual = run_research(request)["status"]
        elif kind == "preferences":
            actual = analyze_preferences(request["arguments"], {tuple(e) for e in request["attacks"]},
                                         [Preference(**p) for p in request.get("preferences", [])],
                                         semantics=request.get("semantics", "grounded"))["extensions"]
        else:
            raise ValueError(f"unsupported benchmark kind: {kind}")
        passed = actual == case["expected"]
        group = groups[case["family"]]
        group["total"] += 1
        group["passed"] += int(passed)
        rows.append({"id": case["id"], "family": case["family"], "passed": passed,
                     "actual": actual, "expected": case["expected"]})
    return {"schema": "vle-benchmark-result/v1", "corpus_sha256": hashlib.sha256(raw).hexdigest(),
            "corpus_version": corpus["version"], "backend": backend,
            "scope": "synthetic_semantic_regression_only", "case_count": len(rows),
            "passed": sum(r["passed"] for r in rows), "all_passed": all(r["passed"] for r in rows),
            "by_family": dict(sorted(groups.items())), "cases": rows}
