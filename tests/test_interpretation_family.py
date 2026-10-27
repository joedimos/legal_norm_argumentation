from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path

import pytest

from src.audit_bundle import digest, verify_bundle
from src.family_contract import family_fingerprint, record_family_review
from src.interpretation_family import analyze_family, run_family_benchmark
from src.temporal import Formula, atom, combine, evaluate_formula

P, Q = atom("notify"), atom("authorization")


def document(id="law", kind="normative", text="Synthetic rule: notify when required."):
    return {"id": id, "version": "1", "kind": kind, "language": "en", "text": text,
            "sha256": digest(text.encode()), **({"jurisdiction": "FR", "effective_from": "2020-01-01", "effective_to": None} if kind == "normative" else {})}


def reference(doc):
    return {"document_id": doc["id"], "start": 0, "end": len(doc["text"]), "quote": doc["text"]}


def request(*formulas, **changes):
    source = document()
    return {"schema": "vle-family/v1", "context": {"jurisdiction": "FR", "as_of": "2026-10-27",
            "actor": "synthetic operator", "capacity": "research fixture"}, "documents": [source],
            "interpretations": [{"id": f"i{i}", "property": f.to_dict(), "source_refs": [reference(source)],
                                 "rationale": "Synthetic interpretation for semantic regression only."}
                                for i, f in enumerate(formulas or (P,))], "horizon": 1, **changes}


def reviewed(payload):
    """Self-reported fixture receipts; these are not independent expert annotations."""
    return record_family_review(payload, scope="family", reviewer="synthetic test reviewer",
                                notes="Test receipt only; not independent legal review.")


def test_family_with_same_facts_opposite_interpretations_cannot_be_factually_resolved():
    payload = request(P, combine("not", P))
    result = analyze_family(payload)
    assert result["status"] == "review_required"
    analysis = result["analysis"]
    assert analysis["status"] == "interpretation_disagreement"
    first, second = analysis["indistinguishable_witnesses"]
    assert first["interpretation_id"] != second["interpretation_id"]
    assert first["trace"] == second["trace"]
    assert first["satisfies"] is not second["satisfies"]
    assert not result["deployment_authorized"]
    assert not result["legal_certification"]
    assert not result["reviews_authenticated"]


def test_agreeing_interpretations_have_factual_query_plan():
    result = analyze_family(reviewed(request(P, P)))
    analysis = result["analysis"]
    assert result["status"] == "clarification_needed"
    assert analysis["worst_case_cost"] == 1
    assert analysis["plan"]["nodes"][0]["atom"] == "notify"
    assert analysis["world_count"] == 4
    assert analysis["optimal_in_scope"]


def test_empty_member_blocks_family_instead_of_disappearing():
    payload = request(Formula("true"), Formula("true"))
    payload["interpretations"][1]["assumptions"] = [Formula("false").to_dict()]
    result = analyze_family(reviewed(payload))
    assert result["status"] == "inconsistent_family"
    assert result["analysis"]["infeasible_interpretations"] == ["i1"]
    assert result["analysis"]["plan"] is None


def test_uniform_satisfaction_and_violation_are_nonvacuous():
    assert analyze_family(reviewed(request(Formula("true"), Formula("true"))))["status"] == "robust_satisfied"
    result = analyze_family(reviewed(request(Formula("false"), Formula("false"))))
    assert result["status"] == "robust_violated"
    assert all(c["feasible"] > 0 for c in result["analysis"]["members"].values())


def test_disjoint_assumptions_can_be_distinguished_by_facts():
    payload = request(Formula("true"), Formula("false"))
    payload["interpretations"][0]["assumptions"] = [Q.to_dict()]
    payload["interpretations"][1]["assumptions"] = [combine("not", Q).to_dict()]
    result = analyze_family(reviewed(payload))
    assert result["analysis"]["status"] == "clarification_needed"
    assert result["analysis"]["worst_case_cost"] == 1


def test_restricted_queries_are_not_interpretation_disagreement():
    result = analyze_family(request(P, P, queries=[]))
    assert result["analysis"]["status"] == "unresolvable_queries"
    assert len(result["analysis"]["indistinguishable_witnesses"]) == 2


def test_assumption_trace_and_source_fingerprints_are_bound():
    payload = reviewed(request(Formula("true")))
    assert analyze_family(payload)["status"] == "robust_satisfied"
    payload["interpretations"][0]["rationale"] += " Changed mapping."
    result = analyze_family(payload)
    assert result["status"] == "review_required"
    assert any(f["code"] == "stale_or_unapproved_review" for f in result["review_findings"])
    changed = deepcopy(payload)
    changed["context"]["capacity"] = "another role"
    assert family_fingerprint(payload) != family_fingerprint(changed)


def grounded_request(kind="observation"):
    payload = request(P, observations=[{"notify": True}])
    evidence = document("event", kind, "Sensor record: notification occurred.")
    payload["documents"].append(evidence)
    payload["grounding"] = [{"step": 0, "atom": "notify", "value": True,
                              "evidence_refs": [reference(evidence)]}]
    return payload


def test_grounding_requires_own_current_review_and_draft_cannot_self_certify():
    payload = reviewed(grounded_request())
    assert analyze_family(payload)["status"] == "review_required"
    payload = record_family_review(payload, scope="grounding", reviewer="fixture grounding reviewer",
                                   notes="Synthetic grounding test only.")
    assert analyze_family(payload)["status"] == "robust_satisfied"
    payload["observations"][0]["notify"] = False
    payload["grounding"][0]["value"] = False
    assert analyze_family(payload)["status"] == "review_required"
    draft = reviewed(grounded_request("draft"))
    draft = record_family_review(draft, scope="grounding", reviewer="fixture", notes="Test receipt cannot erase source-kind gate.")
    assert any(f["code"] == "untrusted_grounding" for f in analyze_family(draft)["review_findings"])
    assert analyze_family(draft)["status"] == "review_required"


@pytest.mark.parametrize("change,code", [("jurisdiction", "jurisdiction_review_required"),
                                        ("expired", "ineffective_source"), ("reference", "unresolved_reference")])
def test_context_gates_cannot_be_overridden_by_a_receipt(change, code):
    payload = request(Formula("true"))
    if change == "jurisdiction":
        payload["documents"][0]["jurisdiction"] = "other"
    elif change == "expired":
        payload["documents"][0]["effective_to"] = "2025-01-01"
    else:
        payload["documents"][0]["references"] = ["missing-definition"]
    result = analyze_family(reviewed(payload))
    assert result["status"] == "review_required"
    assert any(f["code"] == code for f in result["review_findings"])


@pytest.mark.parametrize("damage", ["hash", "quote", "span", "duplicate", "grounding", "horizon", "queries"])
def test_corrupt_inputs_fail_explicitly(damage):
    payload = request(P)
    if damage == "hash":
        payload["documents"][0]["sha256"] = "bad"
    elif damage == "quote":
        payload["interpretations"][0]["source_refs"][0]["quote"] = "bad"
    elif damage == "span":
        payload["interpretations"][0]["source_refs"][0]["start"] = True
    elif damage == "duplicate":
        payload["interpretations"] *= 2
    elif damage == "grounding":
        payload["grounding"] = [{"step": 0, "atom": "notify", "value": True, "evidence_refs": []}]
    elif damage == "horizon":
        payload["horizon"] = True
    else:
        payload["queries"] = [{"interpretation_id": "i0", "cost": 1}]
    with pytest.raises(ValueError):
        analyze_family(payload)


@pytest.mark.parametrize("changes,reason", [({"horizon": 11}, "completion_space_limit"),
                                           ({"max_worlds": 1}, "world_space_limit"),
                                           ({"max_states": 1}, "planning_state_limit")])
def test_limits_release_neither_decision_nor_partial_plan(changes, reason):
    result = analyze_family(reviewed(request(P, P, **changes)))
    assert result["status"] == "unknown"
    assert result["analysis"]["reason"] == reason
    assert result["analysis"]["plan"] is None


def test_single_member_reduces_to_existing_planner():
    from src.evidence_planning import plan_evidence
    for formula in (P, combine("and", P, Q), combine("or", P, Q), combine("eventually", P)):
        for horizon in (1, 2):
            actual = analyze_family(request(formula, horizon=horizon))["analysis"]
            old = plan_evidence({"schema": "vle-evidence-plan/v1", "property": formula.to_dict(), "horizon": horizon})
            assert actual["worst_case_cost"] == old["worst_case_cost"]


def test_all_two_atom_family_truths_against_direct_oracle():
    formulas = [P, Q, combine("not", P), combine("and", P, Q), combine("or", P, Q), Formula("true")]
    traces = [{"notify": p, "authorization": q} for p, q in product((False, True), repeat=2)]
    for first, second in product(formulas, repeat=2):
        result = analyze_family(request(first, second))["analysis"]
        disagrees = any(evaluate_formula(first, [t]) != evaluate_formula(second, [t]) for t in traces)
        if disagrees:
            assert result["status"] == "interpretation_disagreement"
        elif all(evaluate_formula(first, [t]) for t in traces):
            assert result["status"] == "robust_satisfied"
        else:
            assert result["status"] == "clarification_needed"


def test_cli_family_audit_and_explicit_review_receipt(tmp_path):
    source, output = tmp_path / "family.json", tmp_path / "family.zip"
    source.write_text(json.dumps(request(P, combine("not", P))))
    cmd = [sys.executable, "-m", "src.cli", "family", "--input", str(source), "--bundle", str(output)]
    result = subprocess.run(cmd, cwd=Path(__file__).parents[1], text=True, capture_output=True)
    assert result.returncode == 1, result.stderr
    data = json.loads(result.stdout)
    assert data["status"] == "review_required"
    assert verify_bundle(output.read_bytes(), expected_manifest_digest=data["bundle_manifest_digest"])["valid"]
    review = subprocess.run([sys.executable, "-m", "src.cli", "family-review", "--input", str(source),
                             "--scope", "family", "--reviewer", "fixture", "--notes", "Synthetic test only."],
                            cwd=Path(__file__).parents[1], text=True, capture_output=True)
    assert review.returncode == 0, review.stderr
    assert json.loads(review.stdout)["reviews"]["family"]["decision"] == "approved"


def test_synthetic_family_corpus_is_labelled_unreviewed_and_never_authorizes_deployment():
    corpus = json.loads((Path(__file__).parents[1] / "benchmarks/families-v1.json").read_text())
    report = run_family_benchmark(corpus)
    assert report["all_passed"], [c for c in report["cases"] if not c["passed"]]
    assert report["case_count"] == 15
    assert report["annotation_status"] == "synthetic_unreviewed"
    assert all(c["result"]["status"] == "review_required" for c in report["cases"])
    assert all(not c["result"]["deployment_authorized"] for c in report["cases"])
    corpus["cases"][0]["request"] = reviewed(corpus["cases"][0]["request"])
    with pytest.raises(ValueError, match="independent approvals"):
        run_family_benchmark(corpus)


def test_effective_dates_must_be_explicit_and_source_change_invalidates_review():
    payload = request(Formula("true"))
    del payload["documents"][0]["effective_to"]
    result = analyze_family(reviewed(payload))
    assert result["status"] == "review_required"
    assert any(f["code"] == "unknown_effective_period" for f in result["review_findings"])
    payload = reviewed(request(Formula("true")))
    payload["documents"][0]["version"] = "2"
    assert analyze_family(payload)["status"] == "review_required"


def test_family_timeout_never_returns_partial_success(monkeypatch):
    ticks = iter((0.0, 1.0))
    monkeypatch.setattr("src.interpretation_family.monotonic", lambda: next(ticks))
    result = analyze_family(reviewed(request(P, timeout_ms=1)))
    assert result["analysis"]["status"] == "unknown"
    assert result["analysis"]["reason"] == "timeout"
    assert result["analysis"]["plan"] is None


def test_interpretation_order_does_not_change_semantic_findings():
    payload = request(P, combine("not", P))
    first = analyze_family(payload)
    payload["interpretations"].reverse()
    second = analyze_family(payload)
    assert first["analysis"] == second["analysis"]
    assert first["family_fingerprint"] != second["family_fingerprint"]


def test_family_uniform_verdicts_agree_with_independent_smt_member_checks():
    pytest.importorskip("z3")
    from src.model_checking import verify
    formulas = [P, combine("not", P), combine("eventually", P), combine("always", P), Formula("true")]
    for first, second in product(formulas, repeat=2):
        payload = request(first, second, horizon=2)
        analysis = analyze_family(payload)["analysis"]
        positive = [verify(f, horizon=2, backend="z3").status for f in (first, second)]
        negative = [verify(combine("not", f), horizon=2, backend="z3").status for f in (first, second)]
        assert (analysis["status"] == "robust_satisfied") == all(s == "verified_bounded" for s in positive)
        assert (analysis["status"] == "robust_violated") == all(s == "verified_bounded" for s in negative)
