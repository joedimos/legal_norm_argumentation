"""Synthetic, unreviewed regression evidence. No real experts or live providers."""

import json
import sys
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import func, select

from src.audit_bundle import canonical_json, digest
from src.courtlistener import IP_QUERIES, CaseOpinion
from src.legal_annotation import prepare_annotation
from src.resources import resource_path
from src.service.app import create_app
from src.service.automatic_imports import configuration_hash, jobs
from src.service.db_models import AuditEventRecord, ImportCursorRecord
from src.service.ip_release import DIMENSIONS, REVIEW_CHECKS, SCOPE, probe_ip_sources, source_scope_gate
from src.service.release_policy import deployment_release_gate
from src.service.release_readiness import release_readiness
from src.signed_annotation import sign_judgment
from tests.test_production_workflows import (
    BUILD,
    enrollment,
    post,
    release_manifest,
    signed_packet,
    source_review,
    write_json,
)
from tests.test_service import OPERATOR_TOKEN, auth, ingest, settings


def ip_settings(tmp_path, **kwargs):
    values = dict(build_revision=BUILD, release_profile="ip-supervised-v1", model_name="test-model",
                  model_revision="synthetic-test-revision", courtlistener_enabled=True,
                  legifrance_piste_client_id="synthetic-client", legifrance_piste_client_secret="synthetic-value",
                  courtlistener_api_token="synthetic-token", inference_enabled=True)
    values.update(kwargs)
    return settings(tmp_path, **values)


def quality_vector():
    return {name: {label: {"case_count": 6 // len(labels), "correct": 6 // len(labels), "abstentions": 0,
                           "false_compliance": 0, "extraction_f1": 1.0} for label in labels}
            for name, labels in DIMENSIONS.items()}


def fixture_graph(tmp_path):
    registry = enrollment(tmp_path)
    cfg = ip_settings(tmp_path, release_min_cases=6, reviewer_registry_path=str(tmp_path / "registry.json"),
                      reviewer_registry_sha256=write_json(tmp_path / "registry.json", registry))
    signed = signed_packet(tmp_path, "The operator must retain logs.")
    original = signed["packet"]["units"][0]
    packet = prepare_annotation({"schema": "vle-annotation-study/v1", "study_id": "SYNTHETIC-UNREVIEWED-TEST",
                                 "units": [{k: (f"u-{i}" if k == "id" else deepcopy(v)) for k, v in original.items()
                                            if k != "unit_sha256"} for i in range(6)]})
    for i, submission in enumerate(signed["submissions"]):
        payload = deepcopy(submission["payload"])
        judgment = payload["annotations"][0]["judgment"]
        payload.update(packet_sha256=packet["packet_sha256"], annotations=[
            {"unit_id": u["id"], "unit_sha256": u["unit_sha256"], "judgment": deepcopy(judgment)} for u in packet["units"]])
        signed["submissions"][i] = sign_judgment(payload, packet_sha256=packet["packet_sha256"], role="annotator",
                                               private_key_file=tmp_path / f"person-{i}.reviewer.key")
    signed["packet"] = packet
    manifest = release_manifest(tmp_path, cfg, signed)
    manifest.update(schema="vle-release-evidence/v2", scope=SCOPE, model_revision=cfg.model_revision)
    manifest["evidence"].pop("legifrance")
    protocol = {"schema": "vle-ip-review-protocol/v1", "scope": SCOPE,
                "checks": dict.fromkeys(REVIEW_CHECKS, True), "instructions_sha256": "a" * 64}
    benchmark = json.loads((tmp_path / "benchmark.json").read_text())
    benchmark.update(schema="vle-production-benchmark/v2", case_count=6, predictions_sha256="a" * 64,
                     protocol_sha256=digest(canonical_json(protocol)), breakdowns=quality_vector(),
                     faithfulness={p: {"total": 6, "disagreements": 0} for p in
                                   ("direct_vs_symbolic", "direct_vs_gold", "symbolic_vs_gold")})
    receipt = {"source_key": "legifrance:ip:L111-1", "content_sha256": "a" * 64, "raw_payload_sha256": "b" * 64, "full_text": True}
    sources = {"schema": "vle-operational-evidence/v2", "kind": "ip_sources", "scope": SCOPE, "passed": True,
               "evidence_origin": "live_provider_probe", "build_revision": BUILD, "deployment_id": "test-only",
               "observed_at": datetime.now(UTC).isoformat(), "providers": {"legifrance_ip": {"ip": receipt},
               "courtlistener_ip": {t: {**receipt, "source_key": "courtlistener:opinion:1"} for t in IP_QUERIES}}}
    for kind, value in {"review_protocol": protocol, "benchmark": benchmark, "ip_sources": sources}.items():
        manifest["evidence"][kind] = {"filename": kind + ".json", "sha256": write_json(tmp_path / (kind + ".json"), value)}
    cfg.release_evidence_sha256 = write_json(Path(cfg.release_evidence_path), manifest)
    return cfg, manifest


def rewrite_artifact(tmp_path, cfg, manifest, kind, mutate):
    path = tmp_path / manifest["evidence"][kind]["filename"]
    value = json.loads(path.read_text())
    mutate(value)
    manifest["evidence"][kind]["sha256"] = write_json(path, value)
    cfg.release_evidence_sha256 = write_json(Path(cfg.release_evidence_path), manifest)


def test_ip_graph_accepts_complete_structure_and_preserves_legacy(tmp_path):
    cfg, _ = fixture_graph(tmp_path)
    assert deployment_release_gate(cfg)["passed"]
    cfg.release_profile = "legacy"
    assert deployment_release_gate(cfg)["passed"] is False


@pytest.mark.parametrize("kind", ["benchmark", "ip_sources", "review_protocol", "model"])
def test_malformed_pinned_artifacts_block_without_crashing(tmp_path, kind):
    cfg, manifest = fixture_graph(tmp_path)
    manifest["evidence"][kind]["sha256"] = write_json(tmp_path / manifest["evidence"][kind]["filename"], [])
    cfg.release_evidence_sha256 = write_json(Path(cfg.release_evidence_path), manifest)
    assert deployment_release_gate(cfg)["passed"] is False


@pytest.mark.parametrize("kind,mutate", [
    ("ip_sources", lambda x: x["providers"].pop("courtlistener_ip")),
    ("ip_sources", lambda x: x["providers"]["courtlistener_ip"].pop("patent")),
    ("ip_sources", lambda x: x.update(observed_at="2020-01-01T00:00:00+00:00")),
    ("ip_sources", lambda x: x.update(observed_at="2099-01-01T00:00:00+00:00")),
    ("ip_sources", lambda x: x.update(evidence_origin="synthetic_fixture")),
    ("ip_sources", lambda x: x["providers"]["legifrance_ip"]["ip"].update(full_text=False)),
    ("benchmark", lambda x: x["breakdowns"]["language"]["fr"].update(correct=2, abstentions=1)),
    ("benchmark", lambda x: x["breakdowns"]["modality"]["permission"].update(extraction_f1=0.0)),
    ("benchmark", lambda x: x["faithfulness"]["direct_vs_symbolic"].update(total=5)),
    ("benchmark", lambda x: x.update(protocol_sha256="0" * 64)),
    ("review_protocol", lambda x: x["checks"].update(holdings_vs_dicta=1)),
])
def test_ip_gate_rejects_incomplete_stale_or_miscounted_evidence(tmp_path, kind, mutate):
    cfg, manifest = fixture_graph(tmp_path)
    rewrite_artifact(tmp_path, cfg, manifest, kind, mutate)
    assert deployment_release_gate(cfg)["passed"] is False


@pytest.mark.parametrize("kwargs", [{"release_enforcement": "reviewed"}, {"model_revision": None},
                                     {"courtlistener_enabled": False}, {"legifrance_piste_environment": "sandbox"}])
def test_profile_cannot_downgrade_its_policy(tmp_path, kwargs):
    with pytest.raises(ValidationError):
        ip_settings(tmp_path, **kwargs)


def test_source_scope_and_case_review_are_independent_requirements(tmp_path):
    cfg = ip_settings(tmp_path)
    source = SimpleNamespace(source_key="courtlistener:opinion:1", extraction_audit={"provenance": {
        "jurisdiction": "US", "document_type": "judicial_opinion", "raw_payload_sha256": "a" * 64}})
    assert source_scope_gate(cfg, source, {"jurisdiction": "US"})["passed"]
    assert not source_scope_gate(cfg, source, {"jurisdiction": "FR"})["passed"]
    with TestClient(create_app(cfg)) as client:
        raw = ingest(client, "The operator must retain logs.")
        source_review(client, raw)
        result = post(client, f"/v1/sources/{raw['id']}/evaluate", {"high_stakes": True, "observed_actions": ["retain logs"]})
        assert result.status_code == 409
        assert result.json()["decision_authorized"] is False
        assert result.json()["production_release"]["source_scope"]["passed"] is False


def test_readiness_is_authenticated_read_only_and_non_authorizing(tmp_path):
    cfg = ip_settings(tmp_path)
    with TestClient(create_app(cfg)) as client:
        with client.app.state.db.session() as session:
            before = session.scalar(select(func.count()).select_from(AuditEventRecord))
        assert client.get("/v1/release-readiness").status_code == 401
        value = client.get("/v1/release-readiness", headers=auth(OPERATOR_TOKEN)).json()
        assert value["status"] == "blocked" and value["decision_authorized"] is False
        assert "pinned_external_evidence" in value["missing"]
        assert "synthetic-token" not in json.dumps(value)
        with client.app.state.db.session() as session:
            assert session.scalar(select(func.count()).select_from(AuditEventRecord)) == before
        jsonschema.validate(value, json.loads(resource_path("schemas", "release-readiness.schema.json").read_text()))


def test_freshness_checks_scope_changes_future_stamps_and_failures(tmp_path, monkeypatch):
    cfg = ip_settings(tmp_path)
    now = datetime.now(UTC)
    with TestClient(create_app(cfg)) as client:
        db = client.app.state.db
        monkeypatch.setattr(db, "ready", lambda **kw: True)
        active = jobs(cfg)
        with db.session() as session:
            for key, job in active.items():
                session.add(ImportCursorRecord(key=key, config_hash=configuration_hash(cfg, job), last_success_at=now,
                                               due_at=now, failure_count=0, position={}, report={}))
        report = release_readiness(cfg, db, now=now)
        assert "recent_import_batches" not in report["missing"]
        assert report["decision_authorized"] is False and report["status"] == "blocked"
        with db.session() as session:
            session.get(ImportCursorRecord, "legifrance:ip").last_success_at = now + timedelta(seconds=1)
        assert "recent_import_batches" in release_readiness(cfg, db, now=now)["missing"]
        with db.session() as session:
            record = session.get(ImportCursorRecord, "legifrance:ip")
            record.last_success_at, record.config_hash = now, "0" * 64
        assert "recent_import_batches" in release_readiness(cfg, db, now=now)["missing"]


def test_bounded_source_probe_with_recorded_provider(tmp_path):
    calls = []
    opinion = CaseOpinion(1, 2, "SYNTHETIC UNREVIEWED", "The operator must retain logs.", {"raw_payload_sha256": "a" * 64})
    provision = SimpleNamespace(source_key="legifrance:ip:L111-1", source_text="L'auteur doit notifier.",
                                provenance={"raw_payload_sha256": "b" * 64})

    class RecordedProvider:
        def manifest(self, day):
            return {"articles": [{"id": "LEGIARTI1"}], "manifest_sha256": "c" * 64}

        def article(self, item, **kwargs):
            return provision

        def search_page(self, **kwargs):
            calls.append(kwargs)
            return {"pending": [{"id": 1}]}

        def opinion(self, item):
            return opinion

    value = probe_ip_sources(ip_settings(tmp_path), client_factory=lambda *args: RecordedProvider())
    assert len(calls) == 4 and set(value["providers"]["courtlistener_ip"]) == set(IP_QUERIES)
    assert value["bounds"]["max_texts"] == 5
    for group, name in (("benchmarks", "research-v1.json"), ("schemas", "certificate.schema.json")):
        assert resource_path(group, name).is_file()
    with pytest.raises(ValueError):
        resource_path("schemas", "../private.json")


def test_v2_artifact_schemas(tmp_path):
    cfg, manifest = fixture_graph(tmp_path)
    pairs = {"benchmark": "production-benchmark-v2.schema.json", "ip_sources": "ip-source-evidence.schema.json",
             "review_protocol": "ip-review-protocol.schema.json"}
    for kind, schema in pairs.items():
        jsonschema.validate(json.loads((tmp_path / manifest["evidence"][kind]["filename"]).read_text()),
                            json.loads(resource_path("schemas", schema).read_text()))
    jsonschema.validate(manifest, json.loads(resource_path("schemas", "release-evidence-v2.schema.json").read_text()))
    assert deployment_release_gate(cfg)["passed"]


def test_forward_only_installed_migration_entry_point(tmp_path, monkeypatch, capsys):
    from src.service import migrate

    monkeypatch.setenv("VLE_DATABASE_URL", "sqlite+pysqlite:///" + str(tmp_path / "migration.db"))
    monkeypatch.setattr(sys, "argv", ["vle-legato-migrate", "--check"])
    assert migrate.main() == 1
    assert json.loads(capsys.readouterr().out)["passed"] is False
    monkeypatch.setattr(sys, "argv", ["vle-legato-migrate"])
    assert migrate.main() == 0
    assert json.loads(capsys.readouterr().out)["current_heads"] == ["0004_automatic_imports"]
    monkeypatch.setattr(sys, "argv", ["vle-legato-migrate", "--check"])
    assert migrate.main() == 0
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["vle-legato-migrate"])

    def failed(*args):
        raise ValueError("sensitive connection details")

    monkeypatch.setattr(migrate.command, "upgrade", failed)
    assert migrate.main() == 1
    assert "sensitive" not in capsys.readouterr().out


def test_readiness_cli_is_fail_closed_and_sanitizes_config_errors(tmp_path, monkeypatch, capsys):
    from src.service import config
    from src.service import release_readiness as cli

    monkeypatch.setattr(sys, "argv", ["vle-legato-release-check"])
    monkeypatch.setattr(config, "get_settings", lambda: settings(tmp_path))
    assert cli.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert not report["decision_authorized"]

    def failed():
        raise ValueError("sensitive validation input")

    monkeypatch.setattr(config, "get_settings", failed)
    assert cli.main() == 1
    result = capsys.readouterr().out
    assert "sensitive" not in result
    jsonschema.validate(json.loads(result), json.loads(resource_path("schemas", "release-readiness.schema.json").read_text()))
