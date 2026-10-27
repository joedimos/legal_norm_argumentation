from __future__ import annotations

import json
import subprocess
import sys
from itertools import product
from pathlib import Path

import pytest

from src.audit_bundle import verify_bundle
from src.experiments import run_experiment
from src.model_checking import VerificationResult, verify, verify_request
from src.temporal import Formula, atom, combine, evaluate_formula

P, Q = atom("p"), atom("q")


def request(kind="uncertainty", **kwargs):
    return {"schema": "vle-experiment/v1", "kind": kind, "horizon": 2,
            **({"property": P.to_dict()} if kind == "uncertainty" else
               {"before": P.to_dict(), "after": combine("and", P, Q).to_dict()}), **kwargs}


@pytest.mark.parametrize("backend", ["exhaustive", "z3"])
def test_completion_tautology_and_dual_witnesses(backend):
    if backend == "z3":
        pytest.importorskip("z3")
    tautology = combine("or", P, combine("not", P))
    result = run_experiment(request(property=tautology.to_dict(), observations=[{}], backend=backend))
    assert result["status"] == "definitely_true"
    assert result["kleene_value_without_assumptions"] is None
    result = run_experiment(request(observations=[{}, {"p": False}], backend=backend))
    assert result["status"] == "contingent"
    assert evaluate_formula(P, result["satisfying_completion"]) is True
    assert evaluate_formula(P, result["violating_completion"]) is False
    assert result["witness_differences"] == [{"step": 0, "atom": "p", "satisfying": True, "violating": False}]
    assert all(t[1]["p"] is False for t in (result["satisfying_completion"], result["violating_completion"]))
    assert result["differences_are_minimal"] is False


@pytest.mark.parametrize("backend", ["exhaustive", "z3"])
def test_evidence_conflicts_and_observations_reduce_enumeration(backend):
    if backend == "z3":
        pytest.importorskip("z3")
    result = verify(combine("always", P), horizon=40, observations=[{"p": True}] * 40,
                    max_traces=1, backend=backend)
    assert result.status == "verified_bounded"
    assert result.unknown_cells == 0
    assert len(result.observation_constraints) == 40
    incompatible = verify(P, horizon=1, assumptions=[P], observations=[{"p": False}], backend=backend)
    assert incompatible.status == "inconsistent_assumptions"
    assert run_experiment(request(observations=[{"p": False}], assumptions=[P.to_dict()], backend=backend))["status"] == "inconsistent_evidence"


@pytest.mark.parametrize("observations", [[], [{}] * 3, {"p": True}, [{"p": 1}], [{"": True}], [{"p": "false"}]])
def test_invalid_evidence_is_rejected(observations):
    with pytest.raises(ValueError):
        run_experiment(request(observations=observations))


@pytest.mark.parametrize("backend", ["exhaustive", "z3"])
@pytest.mark.parametrize("before,after,status", [
    (P, P, "equivalent_bounded"),
    (P, combine("and", P, Q), "strict_strengthening"),
    (combine("and", P, Q), P, "strict_weakening"),
    (P, Q, "incomparable"),
    (Formula("false"), Formula("true"), "strict_weakening"),
    (Formula("true"), Formula("false"), "strict_strengthening"),
])
def test_amendment_relations_and_witness_replay(backend, before, after, status):
    if backend == "z3":
        pytest.importorskip("z3")
    result = run_experiment(request("amendment", before=before.to_dict(), after=after.to_dict(), backend=backend))
    assert result["status"] == status
    if result["removed_behavior"] is not None:
        assert evaluate_formula(before, result["removed_behavior"]) is True
        assert evaluate_formula(after, result["removed_behavior"]) is False
    if result["added_behavior"] is not None:
        assert evaluate_formula(before, result["added_behavior"]) is False
        assert evaluate_formula(after, result["added_behavior"]) is True
    assert not result["legal_certification"]


def test_shared_assumptions_can_make_amendment_equivalent_only_in_scope():
    result = run_experiment(request("amendment", assumptions=[Q.to_dict()]))
    assert result["status"] == "equivalent_bounded"
    assert result["scope"]["assumptions"] == [Q.to_dict()]
    observed = run_experiment(request("amendment", observations=[{"q": True}]))
    assert observed["status"] == "equivalent_bounded"
    assert observed["request_sha256"] != result["request_sha256"]


def test_budget_failure_and_unavailable_solver_are_not_conclusions(monkeypatch):
    assert run_experiment(request(max_traces=1))["status"] == "unknown"
    monkeypatch.setattr("src.experiments.verify", lambda *args, **kwargs:
                        VerificationResult("unavailable", "z3", 2, ("p",), "missing"))
    assert run_experiment(request(backend="z3"))["status"] == "unavailable"
    assert run_experiment(request("amendment", backend="z3"))["status"] == "unavailable"


def test_verify_request_persists_conditional_evidence():
    result = verify_request({"property": P.to_dict(), "horizon": 1, "observations": [{"p": True}]})
    assert result["status"] == "verified_bounded"
    assert result["observation_constraints"] == ((0, "p", True),)


def test_supervaluation_against_independent_completion_oracle():
    """Enumerate partial evidence; use direct complete-trace truth as an oracle."""
    formulas = [P, combine("not", P), combine("next", P), combine("weak_next", P),
                combine("always", P), combine("eventually", P), combine("until", P, Q),
                combine("or", P, combine("not", P)), combine("and", P, Q)]
    for horizon in (1, 2):
        cells = [(t, n) for t in range(horizon) for n in ("p", "q")]
        for partial in product((None, False, True), repeat=len(cells)):
            observations = [{n: partial[cells.index((t, n))] for n in ("p", "q")} for t in range(horizon)]
            unknowns = [i for i, value in enumerate(partial) if value is None]
            completions = []
            for values in product((False, True), repeat=len(unknowns)):
                assignment = list(partial)
                for i, value in zip(unknowns, values, strict=True):
                    assignment[i] = value
                completions.append([{n: assignment[cells.index((t, n))] for n in ("p", "q")} for t in range(horizon)])
            for formula in formulas:
                truths = {evaluate_formula(formula, trace) for trace in completions}
                expected = "contingent" if len(truths) == 2 else "definitely_true" if True in truths else "definitely_false"
                payload = request(horizon=horizon, property=formula.to_dict(), observations=observations)
                assert run_experiment(payload)["status"] == expected


def test_backends_agree_under_partial_temporal_evidence():
    pytest.importorskip("z3")
    formulas = [combine("next", P), combine("always", P), combine("until", P, Q),
                combine("or", P, combine("not", P))]
    for values in product((None, False, True), repeat=2):
        observations = [{"p": values[0]}, {"q": values[1]}]
        for formula in formulas:
            payload = request(property=formula.to_dict(), observations=observations)
            assert run_experiment(payload)["status"] == run_experiment({**payload, "backend": "z3"})["status"]


def test_cli_bundle_contains_replayable_request(tmp_path):
    source, output = tmp_path / "experiment.json", tmp_path / "experiment.zip"
    source.write_text(json.dumps(request()))
    command = [sys.executable, "-m", "src.cli", "experiment", "--input", str(source), "--bundle", str(output)]
    result = subprocess.run(command, cwd=Path(__file__).parents[1], capture_output=True, text=True)
    assert result.returncode == 1, result.stderr  # contingent is not a passing conclusion
    payload = json.loads(result.stdout)
    assert payload["status"] == "contingent"
    assert verify_bundle(output.read_bytes(), expected_manifest_digest=payload["bundle_manifest_digest"])["valid"]
    duplicate = subprocess.run(command, cwd=Path(__file__).parents[1], capture_output=True, text=True)
    assert duplicate.returncode == 2  # evidence cannot be silently replaced


@pytest.mark.parametrize("backend", ["exhaustive", "z3"])
def test_versioned_experiment_corpus(backend):
    if backend == "z3":
        pytest.importorskip("z3")
    from src.experiments import run_experiment_suite
    corpus = json.loads((Path(__file__).parents[1] / "benchmarks/experiments-v1.json").read_text())
    report = run_experiment_suite(corpus, backend=backend)
    assert report["case_count"] == 19
    assert report["all_passed"], [c for c in report["cases"] if not c["passed"]]
    assert len(report["corpus_sha256"]) == 64


def test_duplicate_request_keys_fail_before_creating_evidence(tmp_path):
    source = tmp_path / "ambiguous.json"
    source.write_text('{"schema":"vle-experiment/v1","schema":"other"}')
    result = subprocess.run([sys.executable, "-m", "src.cli", "experiment", "--input", str(source)],
                            capture_output=True, text=True, cwd=Path(__file__).parents[1])
    assert result.returncode == 2
    assert "duplicate JSON" in json.loads(result.stdout)["error"]


@pytest.mark.parametrize("changes", [{"schema": "v2"}, {"kind": "unknown"}, {"extra": 1},
                                     {"horizon": True}, {"assumptions": {}}, {"horizon": 129}])
def test_experiment_request_rejects_unsupported_fields_and_bounds(changes):
    with pytest.raises(ValueError):
        run_experiment(request(**changes))
