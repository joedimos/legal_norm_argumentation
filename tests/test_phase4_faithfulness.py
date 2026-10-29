import json
from copy import deepcopy
from pathlib import Path

import jsonschema
import pytest

from src.assumptions import assumption_report
from src.certification import assumption_review_subject, phase4_advisory_gates, readiness_report
from src.faithfulness import faithfulness_report, live_recordings, symbolic_verdict
from src.redundancy.generate import FakeProvider
from src.review import ReviewLedger

CORPUS = json.loads(Path("benchmarks/phase4-v1.json").read_text())


def test_three_way_disagreements_grouped_and_hash_reproducible():
    r = faithfulness_report(CORPUS)
    jsonschema.validate(r, json.loads(Path("schemas/faithfulness-report.schema.json").read_text()))
    assert r == faithfulness_report(CORPUS)
    expected = json.loads(Path("tests/fixtures/phase4/faithfulness-expected.json").read_text())
    assert {row["id"]: row["symbolic"] for row in r["rows"]} == expected
    assert r["case_count"] == r["evaluated_count"] == 12
    assert len(r["by_family"]) == 6 and "negated_permission" in r["by_modality"]
    assert set(r["by_has_exception"]) == {"True", "False"}
    assert r["disagreement_rates"]["direct_vs_symbolic"]["disagreements"] > 0
    assert r["disagreement_rates"]["symbolic_vs_gold"]["disagreements"] > 0
    assert not r["decision_authorized"] and r["accuracy_is_not_faithfulness"]


def test_hitting_budget_cannot_report_partial_positive_results():
    r = faithfulness_report({**CORPUS, "max_states": 1})
    assert r["status"] == "unknown" and r["evaluated_count"] == 0 and r["rows"] == []
    assert r["disagreement_rates"]["direct_vs_gold"]["rate"] is None


def test_live_recording_uses_only_sources_and_context_never_gold():
    corpus = {**deepcopy(CORPUS), "cases": deepcopy(CORPUS["cases"][:1])}

    class SpyProvider(FakeProvider):
        def generate(self, prompt, **kw):
            assert "gold_verdict" not in prompt and "direct_solver" not in prompt
            return super().generate(prompt, **kw)

    solver = SpyProvider([{"verdict": "empty"}])
    formal = SpyProvider([corpus["cases"][0]["formalization"]])
    records = live_recordings(corpus, solver, formal, timestamp="2026-01-01T00:00:00+00:00")
    assert solver.calls == formal.calls == 1 and records["review_status"] == "unreviewed"
    assert faithfulness_report(records)["status"] == "reported"
    failed = live_recordings(corpus, FakeProvider([{}]), FakeProvider([{}]))
    r = faithfulness_report(failed)
    assert r["rows"][0]["direct"] == r["rows"][0]["symbolic"] == "unknown"
    assert r["disagreement_rates"]["direct_vs_symbolic"]["determinate_total"] == 0
    with pytest.raises(ValueError):
        live_recordings(CORPUS, solver, formal)


def test_advisory_gates_do_not_change_existing_readiness_or_accept_reviews():
    options = {
        "extraction_audit": {"summary": {"error_count": 0, "possible_omission_count": 0}},
        "review_statuses": ["approved"],
        "unresolved_interpretations": 0,
        "unresolved_conflicts": 0,
    }
    baseline = readiness_report(**options)
    evidence = {"faithfulness": faithfulness_report(CORPUS)}
    r = readiness_report(**options, phase4_evidence=evidence)
    assert r["status"] == baseline["status"] and r["failed_gates"] == baseline["failed_gates"]
    assert all(g["advisory"] for g in r["advisory_gates"])
    assert next(g for g in r["advisory_gates"] if g["gate"] == "faithfulness-gap-reported")["passed"]
    assumptions = assumption_report(
        {"schema": "vle-assumptions/v1", "expected": {"op": "true"}, "horizon": 1, "assumptions": []}
    )
    evidence["assumptions"] = assumptions
    ledger = ReviewLedger()
    assert not phase4_advisory_gates(evidence, review_ledger=ledger)[2]["passed"]
    ledger.record(assumption_review_subject(assumptions), "approved", "synthetic-test-reviewer")
    assert phase4_advisory_gates(evidence, review_ledger=ledger)[2]["passed"]
    changed = deepcopy(assumptions)
    changed["reason"] = "changed"
    assert not phase4_advisory_gates({**evidence, "assumptions": changed}, review_ledger=ledger)[2]["passed"]


def test_invalid_or_unsupported_recordings_never_trusted():
    with pytest.raises(ValueError):
        faithfulness_report({**CORPUS, "synthetic": False})
    for field, value in [
        ("gold_verdict", "approved"),
        ("synthetic", False),
        ("direct_solver", {"verdict": "compliant"}),
        ("formalization_provenance", {}),
    ]:
        corpus = deepcopy(CORPUS)
        corpus["cases"][0][field] = value
        with pytest.raises(ValueError):
            faithfulness_report(corpus)
    assert symbolic_verdict({"kind": "unsupported", "request": {}})[0] == "unknown"


def test_live_cannot_generate_structured_rule_priorities_or_bypass_preflight():
    corpus = {**deepcopy(CORPUS), "cases": deepcopy(CORPUS["cases"][:1])}
    theory = {"kind": "aspic", "query": "p", "request": {"schema": "vle-aspic/v1", "premises": [], "rules": []}}
    records = live_recordings(corpus, FakeProvider([{"verdict": "empty"}]), FakeProvider([theory]))
    assert records["cases"][0]["formalization_provenance"]["generation_failed"]
    assert faithfulness_report(records)["rows"][0]["symbolic"] == "unknown"
    provider = FakeProvider([])
    with pytest.raises(ValueError):
        live_recordings({**corpus, "max_states": 1}, provider, provider)
    assert provider.calls == 0
