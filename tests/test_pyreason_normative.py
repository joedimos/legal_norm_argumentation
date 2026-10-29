"""Synthetic, unreviewed fixtures for the required normative backend. Offline."""

import json
import subprocess
from dataclasses import replace
from itertools import product
from pathlib import Path

import pytest

from src.compliance import evaluate
from src.defeasible import activation
from src.normative_model import FactContext, Modality, NormInterpretation, Predicate, SourceAuthority
from src.pyreason_connector import BACKEND_VERSION, digest
from src.pyreason_normative import compile_activations, run_activation_backend
from src.pyreason_program import execute_program
from tools.fixtures import pyreason as fake


@pytest.fixture
def worker(monkeypatch):
    def run(cmd, **kwargs):
        program = json.loads(Path(cmd[-2]).read_text())
        result = {
            "status": "observed",
            "backend_version": BACKEND_VERSION,
            "input_sha256": digest(program),
            "annotations": execute_program(program, fake),
        }
        Path(cmd[-1]).write_text(json.dumps(result))

    monkeypatch.setattr(subprocess, "run", run)
    return run


def norm(**kwargs):
    return NormInterpretation(
        id="n",
        source_norm_id="synthetic-unreviewed",
        subject="operator",
        action="reproduce",
        modality=Modality.PROHIBITION,
        **kwargs,
    )


def test_exhaustive_three_valued_activation_matches_reference(worker):
    for values in product((False, True, None), repeat=4):
        context = FactContext(
            {key: value for key, value in zip(("c0", "c1", "e0", "e1"), values, strict=True) if value is not None}
        )
        for negate in (False, True):
            n = norm(
                conditions=(Predicate("c0", negate), Predicate("c1")),
                exceptions=(Predicate("e0"), Predicate("e1", negate)),
            )
            report = run_activation_backend([n], context)
            expected = activation(n, context)
            assert report["status"] == "computed", report
            assert report["activations"]["n"]["status"] == expected.status
            assert report["activations"]["n"]["reasons"] == list(expected.reasons)


def test_scope_order_actor_and_semantic_conflicts(worker):
    for actor, date in product((None, "operator", "other"), (None, "2025-01-01", "2026-06-01", "2027-01-01")):
        n = norm(
            conditions=(Predicate("missing"),),
            exceptions=(Predicate("exception"),),
            authority=SourceAuthority(effective_from="2026-01-01", effective_to="2026-12-31"),
        )
        context = FactContext({"exception": True}, actor=actor, at_date=date)
        assert run_activation_backend([n], context)["activations"]["n"]["status"] == activation(n, context).status
    obligation = replace(norm(), id="obligation", modality=Modality.OBLIGATION)
    prohibition = norm()
    for rules in (
        [obligation],
        [prohibition],
        [obligation, prohibition],
        [replace(obligation, modality=Modality.PERMISSION)],
        [replace(obligation, modality=Modality.RECOMMENDATION)],
        [replace(obligation, deadline="unresolved natural-language deadline")],
        [replace(obligation, object="structured work")],
        [obligation, replace(prohibition, authority=SourceAuthority(authority_rank=10))],
    ):
        for actions in ((), ("reproduce",)):
            context = FactContext({}, observed_actions=actions)
            result = evaluate(rules, context, backend="pyreason")
            backend = result["reasoning"].pop("backend")
            assert backend["status"] == "computed"
            assert result == evaluate(rules, context)


def test_no_fallback_when_engine_missing_or_inconsistent(monkeypatch):
    def missing(*args, **kwargs):
        raise OSError("private worker failure")

    monkeypatch.setattr(subprocess, "run", missing)
    assert evaluate([norm()], FactContext({}), backend="pyreason")["status"] == "indeterminate"
    assert evaluate([norm()], FactContext({}))["status"] == "compliant"

    def wrong(cmd, **kwargs):
        program = json.loads(Path(cmd[-2]).read_text())
        annotations = execute_program(program, fake)
        annotations["active"]["n0"] = [0, 1]
        annotations["exception_applies"]["n0"] = [1, 1]
        Path(cmd[-1]).write_text(
            json.dumps(
                {
                    "status": "observed",
                    "backend_version": BACKEND_VERSION,
                    "input_sha256": digest(program),
                    "annotations": annotations,
                }
            )
        )

    monkeypatch.setattr(subprocess, "run", wrong)
    result = evaluate([norm()], FactContext({}), backend="pyreason")
    assert result["status"] == "indeterminate"
    assert result["reasoning"]["backend"]["reason"] == "backend_disagreement"
    assert result["reasoning"]["backend"]["activations"] == {}


def test_compilation_is_deterministic_and_bounded():
    from src.pyreason_connector import ResourceBound

    a, b = norm(), replace(norm(), id="second", conditions=(Predicate("arbitrary):injection"),))
    p, m = compile_activations([a, b], FactContext({}))
    assert (p, m) == compile_activations([b, a], FactContext({}))
    assert "injection" not in json.dumps(p)
    with pytest.raises(ResourceBound):
        compile_activations([replace(norm(), id=str(i)) for i in range(65)], FactContext({}))
    report = run_activation_backend([norm(conditions=tuple(Predicate(str(i)) for i in range(33)))], FactContext({}))
    assert report["status"] == "unknown" and report["bound_hit"]


def test_lifecycle_requires_reviews_and_records_backend_lineage(worker, monkeypatch):
    from src.lifecycle import run_lifecycle
    from src.norm_extraction import extract_norms
    from src.verification import audit_extraction

    text = "The operator must retain logs."
    candidates = extract_norms(text, source_id="synthetic-unreviewed")
    kwargs = {
        "text": text,
        "source_id": "synthetic-unreviewed",
        "candidates": candidates,
        "extraction_audit": audit_extraction(text, candidates),
        "context": FactContext({}, observed_actions=("retain logs",)),
        "reasoning_backend": "pyreason",
    }
    missing = run_lifecycle(**kwargs)
    assert missing["compliance"]["status"] == "indeterminate"
    assert "required_pyreason_computation" in missing["readiness"]["failed_gates"]
    result = run_lifecycle(
        **kwargs, review_statuses=["approved"] * len(candidates)
    )  # Synthetic test gate, not expert data.
    assert result["compliance"]["status"] == "compliant"
    assert result["compliance"]["reasoning"]["backend"]["status"] == "computed"
    assert result["decision_authorized"] is False
    assert any(e["relation"] == "computes_activation_for" for e in result["evidence"]["edges"])


def test_cli_request_schema_and_offline_audit_bundle(worker, tmp_path, monkeypatch, capsys):
    import sys

    from jsonschema import validate

    from src.audit_bundle import verify_bundle
    from src.cli import main

    root = Path(__file__).resolve().parents[1]
    path = root / "examples/pyreason-normative.json"
    bundle = tmp_path / "normative.zip"
    monkeypatch.setattr(sys, "argv", ["vle-legato", "norm-evaluate", "--input", str(path), "--bundle", str(bundle)])
    with pytest.raises(SystemExit) as code:
        main()
    assert code.value.code == 1  # Completed computation reports the planted violation.
    result = json.loads(capsys.readouterr().out)
    validate(result, json.loads((root / "schemas/normative-evaluation-result.schema.json").read_text()))
    validate(
        result["compliance"]["reasoning"]["backend"],
        json.loads((root / "schemas/pyreason-normative.schema.json").read_text()),
    )
    assert result["status"] == "non_compliant"
    assert verify_bundle(bundle.read_bytes())["valid"]


def test_api_selected_backend_is_required_persisted_and_never_auto_authorizes(worker, tmp_path):
    from fastapi.testclient import TestClient

    from src.service.app import create_app
    from tests.test_service import OPERATOR_TOKEN, REVIEWER_TOKEN, auth, ingest, settings

    with TestClient(create_app(settings(tmp_path, pyreason_enabled=True, release_enforcement="reviewed"))) as client:
        source = ingest(client, "The operator must retain logs.")
        url = f"/v1/sources/{source['id']}/evaluate"
        body = {"reasoning_backend": "pyreason", "observed_actions": ["retain logs"], "high_stakes": False}
        first = client.post(url, json=body, headers={**auth(OPERATOR_TOKEN), "Idempotency-Key": "synthetic-first"})
        assert first.status_code == 200
        assert first.json()["compliance"]["status"] == "indeterminate"
        for i, candidate in enumerate(source["candidates"]):
            r = client.post(
                f"/v1/sources/{source['id']}/reviews",
                json={"candidate_id": candidate["id"], "decision": "approved"},
                headers={**auth(REVIEWER_TOKEN), "Idempotency-Key": f"synthetic-review-{i}"},
            )
            assert r.status_code == 201
        result = client.post(
            url, json=body, headers={**auth(OPERATOR_TOKEN), "Idempotency-Key": "synthetic-approved"}
        ).json()
        assert result["compliance"]["status"] == "compliant"
        assert result["compliance"]["reasoning"]["backend"]["status"] == "computed"
        assert result["decision_authorized"] is False
        high = client.post(
            url,
            json={**body, "high_stakes": True},
            headers={**auth(OPERATOR_TOKEN), "Idempotency-Key": "synthetic-high"},
        )
        assert high.status_code == 409
        assert not high.json()["decision_authorized"]
        assert "reasoning_backend_release_qualification" in high.json()["readiness"]["failed_gates"]
        # Both the selected backend and its result are carried by the normal run bundle.
        audit = client.get(f"/v1/runs/{result['run_id']}/audit-bundle", headers=auth(REVIEWER_TOKEN))
        assert audit.status_code == 200
        from src.audit_bundle import verify_bundle

        assert verify_bundle(audit.content)["valid"]


def test_service_opt_in_is_explicit(tmp_path):
    from fastapi.testclient import TestClient

    from src.service.app import create_app
    from tests.test_service import OPERATOR_TOKEN, auth, ingest, settings

    with TestClient(create_app(settings(tmp_path))) as client:
        source = ingest(client, "The operator must retain logs.")
        response = client.post(
            f"/v1/sources/{source['id']}/evaluate",
            json={"reasoning_backend": "pyreason", "high_stakes": False},
            headers={**auth(OPERATOR_TOKEN), "Idempotency-Key": "synthetic-disabled"},
        )
        assert response.status_code == 503


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "wrong"},
        {"backend": "invalid"},
        {"timeout_seconds": True},
        {"norms": []},
        {"context": []},
        {"context": {"facts": {"p": 1}}},
        {"context": {"observed_actions": [False]}},
        {"context": {"actor": False}},
        {"unknown": "field"},
    ],
)
def test_raw_request_rejects_unsupported_or_ambiguous_inputs(change):
    from src.normative_evaluation import evaluate_request

    request = {"schema": "vle-normative-evaluation/v1", "norms": [norm().to_dict()]}
    with pytest.raises(ValueError):
        evaluate_request({**request, **change})


@pytest.mark.parametrize(
    "change",
    [
        {"timesteps": 2},
        {"nodes": {}},
        {"nodes": {"unsafe):": {}}},
        {"nodes": {"n0": {"scope_t": 1}}},
        {"rules": [{"node": "n0", "head": "injected", "body": []}]},
        {"rules": [{"node": "n0", "head": "active", "body": [["missing", True]]}]},
    ],
)
def test_program_grammar_never_accepts_unsafe_statements(change):
    from src.pyreason_program import validate_program

    program, _ = compile_activations([norm()], FactContext({}))
    with pytest.raises(ValueError):
        validate_program({**program, **change})


def test_timeout_limits_and_partial_results_are_unknown(monkeypatch, worker):
    from src.pyreason_program import LIMITS

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("worker", 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    report = run_activation_backend([norm()], FactContext({}), timeout_seconds=1)
    assert report["status"] == "unknown" and report["bound_hit"]
    assert report["reason"] == "backend_timeout" and report["activations"] == {}
    monkeypatch.setitem(LIMITS, "max_rules", 1)
    report = run_activation_backend([norm()], FactContext({}))
    assert report["status"] == "unknown" and report["bound_hit"]
    assert report["bounds"]["max_rules"] == 1


@pytest.mark.parametrize("fault", ["missing", "soft", "multiple", "unavailable"])
def test_malformed_or_partial_outcomes_block_every_norm(monkeypatch, worker, fault):
    def corrupt(cmd, **kwargs):
        program = json.loads(Path(cmd[-2]).read_text())
        annotations = execute_program(program, fake)
        if fault == "missing":
            annotations["active"].pop("n0")
        elif fault == "soft":
            annotations["active"]["n0"] = [0.5, 1]
        elif fault == "multiple":
            annotations["inactive_condition"]["n0"] = [1, 1]
        report = (
            {
                "status": "observed",
                "backend_version": BACKEND_VERSION,
                "input_sha256": digest(program),
                "annotations": annotations,
            }
            if fault != "unavailable"
            else {"status": "unknown"}
        )
        Path(cmd[-1]).write_text(json.dumps(report))

    monkeypatch.setattr(subprocess, "run", corrupt)
    report = evaluate([norm()], FactContext({}), backend="pyreason")
    assert report["status"] == "indeterminate"
    assert report["reasoning"]["backend"]["activations"] == {}


def test_backend_reports_schemas_and_unknown_review_gate(worker):
    from jsonschema import validate

    from src.pyreason_normative import unknown_report

    schema = json.loads((Path(__file__).resolve().parents[1] / "schemas/pyreason-normative.schema.json").read_text())
    for report in [
        run_activation_backend([norm()], FactContext({})),
        unknown_report("reviews_missing"),
        run_activation_backend([], FactContext({})),
        run_activation_backend([norm()], FactContext({}), timeout_seconds=0),
    ]:
        validate(report, schema)
    # A completed computation cannot be relabelled unknown while retaining positive activations.
    report = run_activation_backend([norm()], FactContext({}))
    report.update(status="unknown", reason="mutated")
    from jsonschema import ValidationError

    with pytest.raises(ValidationError):
        validate(report, schema)


def test_worker_capacity_is_bounded_and_released(worker):
    from src.pyreason_connector import _WORKER_SLOT

    assert _WORKER_SLOT.acquire(blocking=False)
    try:
        report = evaluate([norm()], FactContext({}), backend="pyreason")
        backend = report["reasoning"]["backend"]
        assert report["status"] == "indeterminate" and backend["bound_hit"]
        assert backend["reason"] == "worker_concurrency_bound"
    finally:
        _WORKER_SLOT.release()
    assert run_activation_backend([norm()], FactContext({}))["status"] == "computed"
