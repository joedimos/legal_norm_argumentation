from __future__ import annotations

import json
import subprocess
import sys
from itertools import product
from pathlib import Path

import pytest

from src.audit_bundle import verify_bundle
from src.evidence_planning import plan_evidence
from src.temporal import Formula, atom, combine, evaluate_formula

P, Q, S = atom("p"), atom("q"), atom("selector")


def request(property=P, **changes):
    return {"schema": "vle-evidence-plan/v1", "property": property.to_dict(), "horizon": 1, **changes}


def replay(report, trace):
    nodes = {n["id"]: n for n in report["plan"]["nodes"]}
    current, cost, visited = report["plan"]["root"], 0, set()
    while "verdict" not in nodes[current]:
        node = nodes[current]
        cell = (node["step"], node["atom"])
        assert cell not in visited
        visited.add(cell)
        cost += node["cost"]
        current = node["on_true"] if trace[cell[0]][cell[1]] else node["on_false"]
    assert (nodes[current]["verdict"] == "definitely_true") == evaluate_formula(
        Formula.from_dict(report["test_property"]), trace)
    return cost


def test_leaf_zero_cost_and_impossible_context():
    for observed in (True, False):
        report = plan_evidence(request(observations=[{"p": observed}]))
        assert report["status"] == "already_decided"
        assert report["worst_case_cost"] == 0
        assert report["optimal_in_scope"]
    assert plan_evidence(request(assumptions=[Formula("false").to_dict()]))["status"] == "inconsistent_evidence"


def test_weighted_adaptive_multiplexer_plan_replays_every_completion():
    formula = combine("or", combine("and", S, P), combine("and", combine("not", S), Q))
    report = plan_evidence(request(formula, queries=[
        {"step": 0, "atom": "p", "cost": 10}, {"step": 0, "atom": "q", "cost": 10},
        {"step": 0, "atom": "selector", "cost": 1}]))
    assert report["status"] == "planned"
    assert report["worst_case_cost"] == 11
    assert report["plan"]["nodes"][0]["atom"] == "selector"
    report["test_property"] = formula.to_dict()
    costs = [replay(report, [dict(zip(("p", "q", "selector"), bits, strict=True))])
             for bits in product((False, True), repeat=3)]
    assert max(costs) == report["worst_case_cost"]


def test_unavailable_queries_return_indistinguishability_proof():
    formula = combine("and", P, Q)
    report = plan_evidence(request(formula, queries=[{"step": 0, "atom": "p", "cost": 1}]))
    assert report["status"] == "unresolvable"
    first, second = report["indistinguishable_witnesses"]
    assert first[0]["p"] == second[0]["p"]
    assert evaluate_formula(formula, first) != evaluate_formula(formula, second)
    assert report["plan"] is None
    assert report["optimal_in_scope"] is False


def test_assumptions_can_remove_need_for_unavailable_query():
    formula = combine("and", P, Q)
    report = plan_evidence(request(formula, assumptions=[Q.to_dict()], queries=[{"step": 0, "atom": "p", "cost": 3}]))
    assert report["status"] == "planned"
    assert report["worst_case_cost"] == 3
    assert report["feasible_completions"] == 2


def test_temporal_query_cells_are_distinct_and_fixed_cells_are_preserved():
    formula = combine("eventually", P)
    report = plan_evidence(request(formula, horizon=3, observations=[{"p": False}]))
    assert report["worst_case_cost"] == 2
    assert all(n.get("step") != 0 for n in report["plan"]["nodes"] if "atom" in n)
    report["test_property"] = formula.to_dict()
    for first, second in product((False, True), repeat=2):
        assert replay(report, [{"p": False}, {"p": first}, {"p": second}]) <= 2


def test_all_sixteen_two_atom_truth_tables_against_known_decision_complexities():
    assignments = list(product((False, True), repeat=2))
    for outputs in product((False, True), repeat=4):
        clauses = []
        for (p, q), truth in zip(assignments, outputs, strict=True):
            if truth:
                clauses.append(combine("and", P if p else combine("not", P), Q if q else combine("not", Q)))
        formula = Formula("false")
        for clause in clauses:
            formula = combine("or", formula, clause)
        report = plan_evidence(request(formula, queries=[
            {"step": 0, "atom": "p", "cost": 2}, {"step": 0, "atom": "q", "cost": 5}]))
        p_matters = any(outputs[i] != outputs[i + 2] for i in range(2))
        q_matters = any(outputs[i] != outputs[i + 1] for i in (0, 2))
        expected = (2 if p_matters else 0) + (5 if q_matters else 0)
        assert report["worst_case_cost"] == expected
        report["test_property"] = formula.to_dict()
        assert max(replay(report, [{"p": p, "q": q}]) for p, q in assignments) == expected


@pytest.mark.parametrize("changes, reason", [({"horizon": 11}, "completion_space_limit"),
                                             ({"max_states": 1}, "planning_state_limit"),
                                             ({"max_completions": 1}, "completion_space_limit")])
def test_limits_never_release_partial_or_optimal_plans(changes, reason):
    report = plan_evidence(request(**changes))
    assert report["status"] == "unknown"
    assert report["reason"] == reason
    assert report["optimal_in_scope"] is False
    assert report["plan"] is None
    assert report["worst_case_cost"] is None


def test_timeout_discards_partial_plan(monkeypatch):
    clock = iter((0.0, 1.0))
    monkeypatch.setattr("src.evidence_planning.monotonic", lambda: next(clock))
    report = plan_evidence(request(timeout_ms=1))
    assert report["status"] == "unknown"
    assert report["reason"] == "timeout"
    assert report["plan"] is None


@pytest.mark.parametrize("changes", [
    {"schema": "v2"}, {"horizon": True}, {"assumptions": {}}, {"observations": []},
    {"observations": [{"p": 1}]}, {"queries": [{"step": 0, "atom": "p", "cost": 0}]},
    {"queries": [{"step": True, "atom": "p", "cost": 1}]}, {"queries": [{"step": 0, "atom": "p", "cost": True}]},
    {"queries": [{"step": 0, "atom": "p", "cost": 1}] * 2},
    {"observations": [{"p": True}], "queries": [{"step": 0, "atom": "p", "cost": 1}]},
    {"queries": {}, "max_completions": 4097}, {"unknown_field": 1},
])
def test_invalid_requests_are_rejected(changes):
    with pytest.raises(ValueError):
        plan_evidence(request(**changes))


def test_query_order_invariance_and_cost_bound_request_hash():
    queries = [{"step": 0, "atom": "p", "cost": 2}, {"step": 0, "atom": "q", "cost": 1}]
    first = plan_evidence(request(combine("and", P, Q), queries=queries))
    second = plan_evidence(request(combine("and", P, Q), queries=queries[::-1]))
    assert first["plan"] == second["plan"]
    assert first["request_sha256"] != second["request_sha256"]


def test_cli_plan_audit_bundle(tmp_path):
    source, output = tmp_path / "request.json", tmp_path / "evidence.zip"
    source.write_text(json.dumps(request()))
    result = subprocess.run([sys.executable, "-m", "src.cli", "plan-evidence", "--input", str(source),
                             "--bundle", str(output)], cwd=Path(__file__).parents[1], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "planned"
    assert verify_bundle(output.read_bytes(), expected_manifest_digest=report["bundle_manifest_digest"])["valid"]
