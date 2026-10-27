"""Bounded research revision sequences with separately reviewed behavior contracts.

Candidates are supplied, not generated here. Draft constraints are hypothetical
behavior mappings, never trusted observations. Every candidate is rechecked.
"""
from __future__ import annotations

from copy import deepcopy

from .audit_bundle import canonical_json, digest
from .interpretation_family import analyze_family


def bind_candidate(family, candidate):
    if (not isinstance(candidate, dict) or not {"document", "behavior"} <= set(candidate)
            or set(candidate) - {"document", "behavior", "reviews"}):
        raise ValueError("candidate requires document, behavior, and optional mapping/grounding reviews")
    if "behavior" in family:
        raise ValueError("base family must not already bind a candidate")
    document = candidate["document"]
    if not isinstance(document, dict) or document.get("kind") != "draft":
        raise ValueError("candidate document must be a draft snapshot")
    reviews = candidate.get("reviews", {})
    if not isinstance(reviews, dict) or set(reviews) - {"grounding", "behavior"}:
        raise ValueError("candidate receipts may not overwrite normative-family approval")
    result = deepcopy(family)
    result["documents"].append(deepcopy(document))
    result["behavior"] = deepcopy(candidate["behavior"])
    # Prior behavior receipts cannot transfer to a new candidate.
    result.setdefault("reviews", {}).pop("behavior", None)
    result["reviews"].update(deepcopy(reviews))
    return result


def run_revision_sequence(request):
    if (not isinstance(request, dict) or request.get("schema") != "vle-revision/v1"
            or not {"family", "candidates"} <= set(request)
            or set(request) - {"schema", "family", "candidates", "max_cycles"}):
        raise ValueError("invalid vle-revision/v1 request")
    if not isinstance(request["family"], dict):
        raise ValueError("family must be an object")
    candidates = request["candidates"]
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 5:
        raise ValueError("provide 1-5 candidate revisions")
    cycles = request.get("max_cycles", 2)
    if type(cycles) is not int or not 1 <= cycles <= 5:
        raise ValueError("max_cycles must be 1-5 including the initial candidate")
    steps, seen = [], set()
    response = {"schema": "vle-revision-result/v1", "request_sha256": digest(canonical_json(request)),
                "mode": "provided_candidates_with_reviewed_behavior_contracts", "deployment_authorized": False,
                "legal_certification": False, "steps": steps, "selected_candidate": None, "selected_candidate_sha256": None,
                "max_cycles": cycles, "generation_calls": 0}
    for index, candidate in enumerate(candidates[:cycles]):
        bound = bind_candidate(request["family"], candidate)
        fingerprint = candidate["document"].get("sha256")
        if fingerprint in seen:
            return {**response, "status": "stalled", "reason": "candidate_text_repeated"}
        seen.add(fingerprint)
        result = analyze_family(bound)
        status = result["status"]
        feedback = {"status": status, "analysis_status": result["analysis"]["status"],
                    "review_findings": result["review_findings"],
                    "counterexample": result["analysis"].get("violating_witness"),
                    "disagreement": result["analysis"].get("indistinguishable_witnesses")}
        steps.append({"cycle": index, "candidate_id": candidate["document"]["id"],
                      "candidate_sha256": fingerprint, "feedback": feedback, "result": result})
        if status == "robust_satisfied":
            return {**response, "status": "model_satisfied", "selected_candidate": candidate["document"]["id"],
                    "selected_candidate_sha256": fingerprint}
        if status != "robust_violated":
            return {**response, "status": status, "reason": "clarification_or_review_required_before_revision"}
    return {**response, "status": "revision_limit", "reason": "no_model_satisfied_candidate_within_supplied_budget"}
