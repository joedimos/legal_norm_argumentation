"""Bounded uncertainty and normative-amendment experiments over reviewed formulas.

These are explicit finite-model analyses, not automatic legal interpretation.
No completeness, minimality, novelty or legal-certification claim is inferred.
"""
from __future__ import annotations

from collections.abc import Mapping

from .audit_bundle import canonical_json, digest
from .model_checking import verify
from .temporal import Formula, combine, evaluate_formula

SCHEMA = "vle-experiment/v1"
COMMON = {"schema", "kind", "horizon", "assumptions", "observations", "backend", "max_traces", "timeout_ms"}


def _implication(left: Formula, right: Formula) -> Formula:
    return combine("or", combine("not", left), right)


def run_experiment(request: Mapping) -> dict:
    """Run two bounded checks and retain their evidence, including failures.

    The timeout is per verification call (at most two calls). An unknown or
    unavailable backend never becomes a definite or equivalent result.
    """
    if not isinstance(request, dict) or request.get("schema") != SCHEMA:
        raise ValueError("expected vle-experiment/v1 request")
    kind = request.get("kind")
    if kind not in {"uncertainty", "amendment"}:
        raise ValueError("kind must be uncertainty or amendment")
    required = {"property"} if kind == "uncertainty" else {"before", "after"}
    if set(request) - (COMMON | required) or not required <= set(request):
        raise ValueError("missing or unsupported experiment fields")
    assumptions = request.get("assumptions", [])
    if not isinstance(assumptions, list) or len(assumptions) > 64:
        raise ValueError("assumptions must be a list of at most 64 formulas")
    options = {
        "horizon": request.get("horizon"),
        "assumptions": [Formula.from_dict(a) for a in assumptions],
        "observations": request.get("observations"),
        "backend": request.get("backend", "exhaustive"),
        "max_traces": request.get("max_traces", 65536),
        "timeout_ms": request.get("timeout_ms", 5000),
    }
    # Hash the entire request, including uncertainty, constraints and budgets.
    request_digest = digest(canonical_json(request))
    if kind == "uncertainty":
        result = _uncertainty(Formula.from_dict(request["property"]), options)
    else:
        result = _amendment(Formula.from_dict(request["before"]), Formula.from_dict(request["after"]), options)
    return {
        "schema": "vle-experiment-result/v1", "kind": kind, **result,
        "request_sha256": request_digest,
        "scope": {"horizon": options["horizon"], "backend": options["backend"],
                  "timeout_ms_per_check": options["timeout_ms"],
                  "max_traces_per_check": options["max_traces"],
                  "observed_steps": len(options["observations"] or ()),
                  "semantics": "LTLf-completions-at-exact-horizon-v1",
                  "assumptions": assumptions, "observations": options["observations"],
                  "unobserved_values": "all_consistent_boolean_completions"},
        "legal_certification": False,
    }


def _uncertainty(property: Formula, options: dict) -> dict:
    positive = verify(property, **options)
    checks = {"all_completions_satisfy": positive.to_dict()}
    status = positive.status
    true_witness = false_witness = None
    if status == "inconsistent_assumptions":
        verdict = "inconsistent_evidence"
    elif status == "verified_bounded":
        verdict = "definitely_true"
    else:
        negative = verify(combine("not", property), **options)
        checks["all_completions_violate"] = negative.to_dict()
        false_witness = positive.counterexample
        true_witness = negative.counterexample
        if negative.status == "inconsistent_assumptions":
            verdict = "inconsistent_evidence"
        elif negative.status == "verified_bounded":
            verdict = "definitely_false"
        elif true_witness is not None and false_witness is not None:
            verdict = "contingent"
        else:
            verdict = "unavailable" if "unavailable" in {status, negative.status} else "unknown"
    observations = options["observations"]
    kleene = None
    if observations is not None:
        padded = [*observations, *({} for _ in range(options["horizon"] - len(observations)))]
        kleene = evaluate_formula(property, padded)
    # Differences between two witnesses are useful questions, not a minimal query
    # plan or a guarantee that observing one cell resolves the uncertainty.
    differences = []
    if true_witness is not None and false_witness is not None:
        differences = [{"step": t, "atom": name, "satisfying": state[name], "violating": false_witness[t][name]}
                       for t, state in enumerate(true_witness) for name in sorted(state)
                       if state[name] != false_witness[t][name]]
    return {"status": verdict, "checks": checks, "kleene_value_without_assumptions": kleene,
            "satisfying_completion": true_witness, "violating_completion": false_witness,
            "witness_differences": differences, "differences_are_minimal": False}


def _amendment(before: Formula, after: Formula, options: dict) -> dict:
    # Removed behaviors satisfy before but violate after, and vice versa for added.
    preservation = verify(_implication(before, after), **options)
    containment = verify(_implication(after, before), **options)
    statuses = {preservation.status, containment.status}
    if "inconsistent_assumptions" in statuses:
        status = "inconsistent_evidence"
    elif statuses == {"verified_bounded"}:
        status = "equivalent_bounded"
    elif preservation.status == "counterexample" and containment.status == "verified_bounded":
        status = "strict_strengthening"
    elif preservation.status == "verified_bounded" and containment.status == "counterexample":
        status = "strict_weakening"
    elif statuses == {"counterexample"}:
        status = "incomparable"
    else:
        status = "unavailable" if "unavailable" in statuses else "unknown"
    return {"status": status,
            "checks": {"before_implies_after": preservation.to_dict(), "after_implies_before": containment.to_dict()},
            "removed_behavior": preservation.counterexample, "added_behavior": containment.counterexample,
            "interpretation": "inclusion_of_satisfying_complete_traces_under_shared_evidence"}


def run_experiment_suite(corpus: dict, *, backend: str = "exhaustive") -> dict:
    """Reproduce a small, explicitly synthetic regression corpus with fixed labels."""
    if (not isinstance(corpus, dict) or set(corpus) != {"schema", "cases"}
            or corpus["schema"] != "vle-experiment-corpus/v1"):
        raise ValueError("invalid experiment corpus")
    cases = corpus["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 128:
        raise ValueError("corpus requires 1-128 cases")
    if backend not in {"exhaustive", "z3"}:
        raise ValueError("backend must be exhaustive or z3")
    seen = set()
    rows = []
    for case in cases:
        if (not isinstance(case, dict) or set(case) != {"id", "request", "expected_status"}
                or not isinstance(case["id"], str) or not case["id"].strip() or case["id"] in seen
                or not isinstance(case["expected_status"], str) or not isinstance(case["request"], dict)):
            raise ValueError("invalid or duplicate experiment case")
        seen.add(case["id"])
        result = run_experiment({**case["request"], "backend": backend})
        rows.append({"id": case["id"], "expected": case["expected_status"], "actual": result["status"],
                     "passed": result["status"] == case["expected_status"], "result": result})
    return {"schema": "vle-experiment-benchmark/v1", "corpus_sha256": digest(canonical_json(corpus)),
            "backend": backend, "case_count": len(rows), "passed": sum(row["passed"] for row in rows),
            "all_passed": all(row["passed"] for row in rows), "cases": rows,
            "scope": "synthetic_regression_not_held_out_legal_accuracy", "legal_certification": False}
