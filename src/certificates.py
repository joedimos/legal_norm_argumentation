"""Evidence production using the main evaluator; offline checker is independent."""

from __future__ import annotations

from itertools import product

from .audit_bundle import canonical_json, digest
from .checker import check_certificate, merkle, validate_query
from .model_checking import verify_request
from .phase4 import BoundHit, Budget, envelope
from .temporal import Formula, evaluate_formula


def query_from_request(request):
    allowed = {"property", "horizon", "assumptions", "observations", "binding", "max_states", "timeout_ms", "backend"}
    if not isinstance(request, dict) or set(request) - allowed or not {"property", "horizon"} <= set(request):
        raise ValueError("invalid certificate query")
    q = {
        k: request.get(k, default)
        for k, default in [
            ("property", None),
            ("horizon", None),
            ("assumptions", []),
            ("observations", []),
            ("binding", {}),
        ]
    }
    validate_query(q)
    return q


def enumeration_evidence(q, budget):
    names = validate_query(q)
    h = q["horizon"]
    fixed = {(t, k): v for t, s in enumerate(q["observations"]) for k, v in s.items() if v is not None}
    free = [(t, k) for t in range(h) for k in names if (t, k) not in fixed]
    if len(free) > budget.max_states.bit_length() or 2 ** len(free) > budget.max_states:
        raise BoundHit("state_limit")
    if 2 ** len(free) * (len(names) * h + 1) > 262144:
        raise BoundHit("work_limit")
    property = Formula.from_dict(q["property"])
    assumptions = [Formula.from_dict(a) for a in q["assumptions"]]
    leaves = []
    witness = None
    feasible = 0
    for i, values in enumerate(product((False, True), repeat=len(free))):
        budget.tick()
        assignment = {**fixed, **dict(zip(free, values, strict=True))}
        trace = [{k: assignment[t, k] for k in names} for t in range(h)]
        allowed = all(evaluate_formula(a, trace) is True for a in assumptions)
        value = evaluate_formula(property, trace)
        leaves.append({"index": i, "trace": trace, "admissible": allowed, "value": value})
        if allowed:
            feasible += 1
            if value is False and witness is None:
                witness = trace
    budget.tick()
    return {"state_count": len(leaves), "admissible_count": feasible, "merkle_root": merkle(leaves)}, witness


def make_certificate(q, kind, evidence):
    return {
        "schema": "vle-certificate/v1",
        "type": kind,
        "query": q,
        "query_sha256": digest(canonical_json(q)),
        "evidence": evidence,
        "review_status": "unreviewed",
        "decision_authorized": False,
        "legal_certification": False,
    }


def certify(request, *, expected_query_sha256=None):
    q = query_from_request(request)
    budget = Budget(max_states=request.get("max_states", 65536), timeout_ms=request.get("timeout_ms", 5000))
    backend = request.get("backend", "exhaustive")
    if backend not in {"exhaustive", "z3"}:
        raise ValueError("backend must be exhaustive or z3")
    try:
        budget.tick()
        names = validate_query(q)
        fixed_count = sum(v is not None for state in q["observations"] for v in state.values())
        free_count = len(names) * q["horizon"] - fixed_count
        if free_count > budget.max_states.bit_length() or 2 ** free_count > budget.max_states:
            raise BoundHit("state_limit")
        if 2 ** free_count * (len(names) * q["horizon"] + 1) > 262144:
            raise BoundHit("work_limit")
        checked = verify_request(
            {
                "property": q["property"],
                "horizon": q["horizon"],
                "assumptions": q["assumptions"],
                "observations": q["observations"] or None,
                "backend": backend,
                "max_traces": budget.max_states,
                "timeout_ms": budget.timeout_ms,
            }
        )
        budget.tick()
        if checked["status"] == "counterexample":
            certificate = make_certificate(q, "violation", {"trace": list(checked["counterexample"])})
        elif checked["status"] == "verified_bounded":
            evidence, witness = enumeration_evidence(q, budget)
            if witness is not None:
                raise BoundHit("unsupported_fragment")
            certificate = make_certificate(q, "compliant_within_bound", evidence)
        else:
            raise BoundHit(
                "unsupported_fragment"
                if checked["status"] in {"unavailable", "inconsistent_assumptions"}
                else "timeout"
                if "timeout" in checked["reason"]
                else "state_limit"
                if checked["reason"] == "state_space_limit"
                else "encoding_limit"
                if checked["reason"] == "encoding_budget"
                else "unsupported_fragment"
            )
    except BoundHit as exc:
        bound = str(exc)
        certificate = make_certificate(
            q,
            "unknown",
            {
                "bound": bound,
                "limit": budget.timeout_ms
                if bound == "timeout"
                else 262144
                if bound == "work_limit"
                else 250000
                if bound == "encoding_limit"
                else 0
                if bound == "unsupported_fragment"
                else budget.max_states,
            },
        )
    validation = check_certificate(
        certificate,
        expected_query_sha256=expected_query_sha256,
        max_states=budget.max_states,
        timeout_ms=budget.timeout_ms,
    )
    try:
        budget.tick()
    except BoundHit as exc:
        bound = str(exc)
        certificate = make_certificate(
            q,
            "unknown",
            {
                "bound": bound,
                "limit": budget.timeout_ms
                if bound == "timeout"
                else 262144
                if bound == "work_limit"
                else 250000
                if bound == "encoding_limit"
                else 0
                if bound == "unsupported_fragment"
                else budget.max_states,
            },
        )
    if not validation["valid"]:
        certificate = make_certificate(q, "unknown", {"bound": "unsupported_fragment", "limit": 0})
    if certificate["type"] == "unknown":
        validation = check_certificate(certificate, expected_query_sha256=expected_query_sha256)
    return envelope(
        "vle-certified-result/v1",
        request,
        status=certificate["type"],
        certificate=certificate,
        checker=validation,
        bounds=budget.to_dict(),
        proof_scope="explicit_formula_not_legal_or_priority_faithfulness",
    )
