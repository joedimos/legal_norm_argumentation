"""Synthetic/unreviewed software fixtures; no provider or expert evaluation."""

import json
import shutil
import sys
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import jsonschema
import pytest
from sqlalchemy import select

from src import api_contract
from src.audit_bundle import canonical_json, digest
from src.courtlistener import CaseOpinion
from src.resources import resource_path
from src.service import migrate, operations
from src.service.config import Settings
from src.service.database import Database
from src.service.db_models import AuditEventRecord, IdempotencyRecord, ImportCursorRecord, RateLimitBucket
from src.service.ip_release import ip_source_evidence
from src.service.operations import SnapshotBound, database_snapshot, extended_database_snapshot
from src.service.repository import append_audit, verify_audit_chain
from tests.test_release_candidate import ip_settings


def schema(name, value):
    jsonschema.validate(value, json.loads(resource_path("schemas", name).read_text()))


def baseline():
    return json.loads(resource_path("schemas", "api-v1-openapi.json").read_text())


def test_current_api_contract_and_pin_without_database_writes(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    current = api_contract.current_contract()
    assert len(current["paths"]) == 27
    assert current["paths"]["/v1/workflows/families"]["post"]["x-vle-role-guards"] == [["operator"]]
    assert api_contract.compare_contracts(baseline(), current)["status"] == "compatible_with_baseline"
    assert digest(canonical_json(baseline())) == api_contract.BASELINE_SHA256
    monkeypatch.setattr(sys, "argv", ["contract"])
    assert api_contract.main() == 0
    schema("api-contract-report.schema.json", json.loads(capsys.readouterr().out))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mutate", [
    lambda d: d["paths"].pop("/v1/sources"),
    lambda d: d["components"]["schemas"]["SourceCreateRequest"]["properties"]["text"].update(maxLength=20),
    lambda d: d["components"]["schemas"]["EvaluateRequest"]["properties"]["high_stakes"].update(default=False),
    lambda d: d["paths"]["/v1/sources"]["post"].update(**{"x-vle-role-guards": [["reader"]]}),
    lambda d: d["components"]["schemas"]["SourceCreateRequest"]["required"].append("extra"),
    lambda d: d["paths"]["/v1/sources"]["post"]["responses"].pop("201"),
])
def test_api_contract_detects_route_default_role_constraint_and_response_drift(mutate):
    old = baseline()
    current = deepcopy(old)
    mutate(current)
    report = api_contract.compare_contracts(old, current)
    assert report["status"] == "review_required" and report["changes"]
    assert report["decision_authorized"] is False


def test_additive_extensions_and_invalid_references_are_distinct():
    old = baseline()
    current = deepcopy(old)
    current["paths"]["/v1/new-test-route"] = deepcopy(current["paths"]["/v1/sources"])
    current["components"]["schemas"]["SourceCreateRequest"]["properties"]["extra"] = {"type": "string"}
    assert api_contract.compare_contracts(old, current)["status"] == "compatible_with_baseline"
    current["paths"]["/v1/new-test-route"] = "malformed"
    assert api_contract.compare_contracts(old, current)["status"] == "unknown"
    current = deepcopy(old)
    current["components"]["schemas"]["SourceCreateRequest"]["$ref"] = "https://example.invalid/schema"
    assert api_contract.compare_contracts(old, current)["status"] == "unknown"
    for bounds in ({"max_nodes": 1}, {"max_depth": 1}, {"max_nodes": True}):
        report = api_contract.compare_contracts(old, old, **bounds)
        assert report["status"] == "unknown" and report["truncated"]


def test_contract_cli_invalid_input_and_baseline_change(tmp_path, monkeypatch, capsys):
    p = tmp_path / "invalid.json"
    p.write_text("{")
    monkeypatch.setattr(sys, "argv", ["contract", "--current", str(p)])
    assert api_contract.main() == 1
    assert json.loads(capsys.readouterr().out)["status"] == "unknown"
    monkeypatch.setattr(api_contract, "BASELINE_SHA256", "0" * 64)
    with pytest.raises(SystemExit) as error:
        api_contract.main()
    assert error.value.code == 2


class Clock:
    value = 0

    def __call__(self):
        return self.value


class RecordedProvider:
    def __init__(self, clock, *, slow=False, empty=False, excluded=False):
        self.clock, self.slow, self.empty, self.excluded = clock, slow, empty, excluded

    def manifest(self, day):
        if self.slow:
            self.clock.value += 1000
        return {"articles": [{"id": "LEGIARTI1"}], "manifest_sha256": "c" * 64}

    def article(self, item, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(source_key="legifrance:ip:L111-1", source_text="L'auteur doit notifier.",
                               provenance={"raw_payload_sha256": "b" * 64})

    def search_page(self, **kwargs):
        return {"pending": [] if self.empty else [{"id": 1}]}

    def opinion(self, item):
        return None if self.excluded else CaseOpinion(1, 2, "SYNTHETIC UNREVIEWED", "The operator must retain logs.",
                                                      {"raw_payload_sha256": "a" * 64})


@pytest.mark.parametrize("options", [{}, {"slow": True}, {"empty": True}, {"excluded": True}])
def test_source_evidence_always_schema_valid_and_bound_hits_unknown(tmp_path, options):
    clock = Clock()
    result = ip_source_evidence(ip_settings(tmp_path), deployment_id="synthetic", build_revision="a" * 40,
                                monotonic=clock, client_factory=lambda *args: RecordedProvider(clock, **options))
    schema("ip-source-evidence.schema.json", result)
    assert result["passed"] is (not options)
    assert result["decision_authorized"] is False
    if options:
        assert result["status"] == "unknown" and "providers" not in result
        assert result["bounds"]["max_texts"] == 5
    if options.get("slow"):
        assert result["bound_hit"] and result["reason"] == "source_probe_bound"


def test_failed_probe_cli_hides_credentials_and_retains_format(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ops", "--mode", "ip-sources", "--deployment-id", "synthetic", "--build-revision", "a" * 40])

    def bad_settings():
        raise ValueError("sensitive credentials")

    monkeypatch.setattr(operations, "Settings", bad_settings)
    with pytest.raises(SystemExit) as error:
        operations.main()
    assert error.value.code == 1
    raw = capsys.readouterr().out
    assert "sensitive" not in raw
    result = json.loads(raw)
    schema("ip-source-evidence.schema.json", result)
    assert result["status"] == "unknown" and result["bounds"]["max_total_seconds"] is None


@pytest.fixture
def migrated(tmp_path, monkeypatch, capsys):
    url = "sqlite+pysqlite:///" + str(tmp_path / "source.db")
    monkeypatch.setenv("VLE_DATABASE_URL", url)
    monkeypatch.setattr(sys, "argv", ["migrate"])
    assert migrate.main() == 0
    capsys.readouterr()
    db = Database(Settings(environment="test", database_url=url, auto_create_schema=False, _env_file=None))
    stamp = datetime(2026, 10, 29, tzinfo=UTC)
    with db.session() as session:
        append_audit(session, actor="synthetic", action="fixture.recovery", resource_type="fixture", resource_id="synthetic-unreviewed",
                     request_id="synthetic", payload={"synthetic": True, "review_status": "unreviewed"})
        session.add(ImportCursorRecord(key="legifrance:ip", config_hash="a" * 64, position={"offset": 3},
                                       due_at=stamp, status="pending", report={"synthetic": True}, last_success_at=stamp,
                                       lease_token="synthetic-lease", lease_expires_at=stamp + timedelta(seconds=180)))
        session.add(RateLimitBucket(key_hash="b" * 64, window_start=stamp, count=2))
        session.add(IdempotencyRecord(key_hash="c" * 64, actor="synthetic", endpoint="synthetic", request_hash="d" * 64,
                                      response_payload={"synthetic": True}, status_code=200, expires_at=stamp + timedelta(days=1)))
    yield db
    db.dispose()


def test_v2_recovery_preserves_operational_state_and_v1_remains_compatible(migrated, tmp_path):
    old = database_snapshot(migrated)
    snapshot = extended_database_snapshot(migrated)
    schema("database-snapshot-v2.schema.json", snapshot)
    assert all(v["count"] == 1 for v in snapshot["operational_tables"].values())
    migrated.dispose()
    shutil.copyfile(tmp_path / "source.db", tmp_path / "restored.db")
    restored = Database(Settings(database_url="sqlite+pysqlite:///" + str(tmp_path / "restored.db"), _env_file=None))
    try:
        assert extended_database_snapshot(restored) == snapshot
        assert database_snapshot(restored) == old
        with restored.session() as session:
            cursor = session.get(ImportCursorRecord, "legifrance:ip")
            assert cursor.position == {"offset": 3} and cursor.lease_token == "synthetic-lease"
            cursor.position = {"offset": 4}
        assert extended_database_snapshot(restored) != snapshot
        assert database_snapshot(restored) == old  # Demonstrates the prior verification gap.
    finally:
        restored.dispose()


def test_snapshot_row_byte_and_time_bounds_and_chain_tamper(migrated):
    for kwargs in ({"max_rows": 1}, {"max_bytes": 1}):
        with pytest.raises(SnapshotBound):
            extended_database_snapshot(migrated, **kwargs)
    clock = iter([0, 31])
    with pytest.raises(SnapshotBound):
        extended_database_snapshot(migrated, monotonic=lambda: next(clock))
    with pytest.raises(ValueError):
        extended_database_snapshot(migrated, max_rows=True)
    with migrated.session() as session:
        event = session.scalar(select(AuditEventRecord))
        event.event_hash = "0" * 64
    with pytest.raises(ValueError, match="audit chain"):
        extended_database_snapshot(migrated)


def test_snapshot_bound_cli_has_no_partial_success(migrated, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ops", "--mode", "snapshot", "--include-operational", "--snapshot-max-rows", "1",
                                     "--deployment-id", "synthetic", "--build-revision", "a" * 40])
    with pytest.raises(SystemExit) as error:
        operations.main()
    result = json.loads(capsys.readouterr().out)
    assert error.value.code == 1 and result["status"] == "unknown" and result["passed"] is False
    assert result["bounds"]["max_rows"] == 1 and "tables" not in result


def test_bounded_audit_stream_and_legacy_results_agree(migrated):
    with migrated.session() as session:
        assert verify_audit_chain(session, max_events=10) == verify_audit_chain(session)
        append_audit(session, actor="synthetic", action="fixture.recovery", resource_type="fixture", resource_id="synthetic-unreviewed",
                     request_id="second", payload={"synthetic": True})
        assert verify_audit_chain(session, max_events=1)["status"] == "unknown"
