from __future__ import annotations

import json
import os

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.service.app import create_app
from src.service.config import Settings
from src.service.database import Database
from src.service.rate_limit import DatabaseRateLimiter

OPERATOR_TOKEN = "operator-token-abcdefghijklmnopqrstuvwxyz-123456"
REVIEWER_TOKEN = "reviewer-token-abcdefghijklmnopqrstuvwxyz-123456"
METRICS_TOKEN = "metrics-token-abcdefghijklmnopqrstuvwxyz-1234567"


def settings(tmp_path, **overrides) -> Settings:
    payload = dict(
        environment="test",
        database_url=f"sqlite+pysqlite:///{tmp_path / 'service.db'}",
        api_keys_json=json.dumps(
            [
                {"token": OPERATOR_TOKEN, "subject": "operator-a", "roles": ["operator", "reader"]},
                {"token": REVIEWER_TOKEN, "subject": "reviewer-b", "roles": ["reviewer", "reader", "auditor"]},
            ]
        ),
        metrics_bearer_token=METRICS_TOKEN,
        auto_create_schema=True,
        require_separation_of_duties=True,
        rate_limit_per_minute=1000,
    )
    payload.update(overrides)
    return Settings(**payload)


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def ingest(client: TestClient, text: str, key: str = "source-idem-001") -> dict:
    response = client.post(
        "/v1/sources",
        json={"source_id": "policy-1", "text": text},
        headers={**auth(OPERATOR_TOKEN), "Idempotency-Key": key},
    )
    assert response.status_code == 201
    return response.json()


def test_health_authentication_and_idempotency(tmp_path):
    with TestClient(create_app(settings(tmp_path))) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
        assert client.post("/v1/sources", json={"source_id": "x", "text": "A must act."}).status_code == 401

        body = {"source_id": "policy-1", "text": "The operator must retain logs."}
        headers = {**auth(OPERATOR_TOKEN), "Idempotency-Key": "same-request-001"}
        first = client.post("/v1/sources", json=body, headers=headers)
        second = client.post("/v1/sources", json=body, headers=headers)
        assert first.status_code == second.status_code == 201
        assert first.json()["id"] == second.json()["id"]

        changed = client.post(
            "/v1/sources",
            json={"source_id": "policy-1", "text": "The operator must retain different logs."},
            headers=headers,
        )
        assert changed.status_code == 409

        assert client.get("/metrics").status_code == 401
        assert client.get("/metrics", headers=auth(METRICS_TOKEN)).status_code == 200


def test_high_stakes_decision_is_blocked_until_review(tmp_path):
    with TestClient(create_app(settings(tmp_path))) as client:
        source = ingest(client, "The operator must retain logs.")
        source_id = source["id"]
        blocked = client.post(
            f"/v1/sources/{source_id}/evaluate",
            json={"observed_actions": ["retain logs"], "high_stakes": True},
            headers={**auth(OPERATOR_TOKEN), "Idempotency-Key": "eval-blocked-001"},
        )
        assert blocked.status_code == 409
        assert blocked.json()["decision_authorized"] is False
        assert "human_review" in blocked.json()["readiness"]["failed_gates"]

        candidate_id = source["candidates"][0]["id"]
        review = client.post(
            f"/v1/sources/{source_id}/reviews",
            json={"candidate_id": candidate_id, "decision": "approved"},
            headers={**auth(REVIEWER_TOKEN), "Idempotency-Key": "review-001"},
        )
        assert review.status_code == 201

        approved = client.post(
            f"/v1/sources/{source_id}/evaluate",
            json={"observed_actions": ["retain logs"], "high_stakes": True},
            headers={**auth(OPERATOR_TOKEN), "Idempotency-Key": "eval-approved-001"},
        )
        assert approved.status_code == 200
        assert approved.json()["decision_authorized"] is True
        assert approved.json()["compliance"]["status"] == "compliant"


def test_ambiguous_formalization_requires_explicit_selection(tmp_path):
    with TestClient(create_app(settings(tmp_path))) as client:
        source = ingest(client, "The operator must retain it.", key="source-ambiguous-001")
        source_id = source["id"]
        candidate_id = source["candidates"][0]["id"]
        formal = source["formalizations"][0]
        assert formal["unresolved"] is True
        interpretation_id = formal["interpretations"][0]["id"]

        review = client.post(
            f"/v1/sources/{source_id}/reviews",
            json={"candidate_id": candidate_id, "decision": "approved", "note": "Source checked."},
            headers={**auth(REVIEWER_TOKEN), "Idempotency-Key": "review-ambiguous-001"},
        )
        assert review.status_code == 201

        still_blocked = client.post(
            f"/v1/sources/{source_id}/evaluate",
            json={"observed_actions": ["retain it"], "high_stakes": True},
            headers={**auth(OPERATOR_TOKEN), "Idempotency-Key": "eval-ambiguous-001"},
        )
        assert still_blocked.status_code == 409
        assert "interpretation_resolution" in still_blocked.json()["readiness"]["failed_gates"]

        selected = client.post(
            f"/v1/sources/{source_id}/interpretations",
            json={
                "candidate_id": candidate_id,
                "interpretation_id": interpretation_id,
                "rationale": "Human reviewer resolved the anaphoric scope against the source context.",
            },
            headers={**auth(REVIEWER_TOKEN), "Idempotency-Key": "selection-001"},
        )
        assert selected.status_code == 201

        resolved = client.post(
            f"/v1/sources/{source_id}/evaluate",
            json={"observed_actions": ["retain it"], "high_stakes": True},
            headers={**auth(OPERATOR_TOKEN), "Idempotency-Key": "eval-resolved-001"},
        )
        assert resolved.status_code == 200
        assert resolved.json()["decision_authorized"] is True


def test_audit_chain_and_rbac(tmp_path):
    with TestClient(create_app(settings(tmp_path))) as client:
        source = ingest(client, "The operator must retain logs.", key="source-audit-001")
        source_id = source["id"]
        candidate_id = source["candidates"][0]["id"]

        assert client.post(
            f"/v1/sources/{source_id}/reviews",
            json={"candidate_id": candidate_id, "decision": "approved"},
            headers={**auth(OPERATOR_TOKEN), "Idempotency-Key": "wrong-role-001"},
        ).status_code == 403

        assert client.post(
            f"/v1/sources/{source_id}/reviews",
            json={"candidate_id": candidate_id, "decision": "approved"},
            headers={**auth(REVIEWER_TOKEN), "Idempotency-Key": "audit-review-001"},
        ).status_code == 201

        audit = client.get(f"/v1/audit/{source_id}", headers=auth(REVIEWER_TOKEN))
        assert audit.status_code == 200
        events = audit.json()["events"]
        assert len(events) >= 2
        assert events[1]["previous_hash"] == events[0]["event_hash"]
        chain = client.get("/v1/audit-chain/verify", headers=auth(REVIEWER_TOKEN))
        assert chain.status_code == 200
        assert chain.json()["valid"] is True


def test_request_body_limit(tmp_path):
    with TestClient(create_app(settings(tmp_path, max_request_bytes=1024))) as client:
        response = client.post(
            "/v1/sources",
            content=b"x" * 1200,
            headers={
                **auth(OPERATOR_TOKEN),
                "Content-Type": "application/json",
                "Idempotency-Key": "body-limit-001",
            },
        )
        assert response.status_code == 413


def test_postgres_migration_readiness_when_configured():
    url = os.environ.get("VLE_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("VLE_TEST_POSTGRES_URL not configured")
    cfg = Settings(
        environment="test",
        database_url=url,
        api_keys_json=json.dumps(
            [
                {"token": OPERATOR_TOKEN, "subject": "operator-a", "roles": ["operator", "reader"]},
                {"token": REVIEWER_TOKEN, "subject": "reviewer-b", "roles": ["reviewer", "auditor"]},
            ]
        ),
        metrics_bearer_token=METRICS_TOKEN,
        auto_create_schema=False,
        rate_limit_per_minute=1000,
    )
    app = create_app(cfg)
    assert app.state.db.ready(require_migrations=True) is True
    app.state.db.dispose()


def test_production_settings_fail_closed():
    with pytest.raises(ValidationError):
        Settings(environment="production", database_url="sqlite+pysqlite:///unsafe.db", api_keys_json="[]")


def test_database_rate_limit_is_shared_state(tmp_path):
    cfg = settings(tmp_path)
    db = Database(cfg)
    db.create_schema_for_tests()
    limiter = DatabaseRateLimiter(db, 1)
    limiter.consume("key-hash")
    with pytest.raises(HTTPException) as exc:
        limiter.consume("key-hash")
    assert exc.value.status_code == 429
    db.dispose()


def test_audit_bundle_requires_auditor_and_freezes_review_evidence(tmp_path):
    import io
    import zipfile

    from src.audit_bundle import verify_bundle

    with TestClient(create_app(settings(tmp_path, build_revision="a" * 40))) as client:
        source = ingest(client, "The operator must retain logs.")
        sid = source["id"]
        run = client.post(
            f"/v1/sources/{sid}/evaluate", json={"high_stakes": True},
            headers={**auth(OPERATOR_TOKEN), "Idempotency-Key": "bundle-evaluation-001"},
        )
        assert run.status_code == 409
        rid = run.json()["run_id"]
        url = f"/v1/runs/{rid}/audit-bundle"
        assert client.get(url).status_code == 401
        assert client.get(url, headers=auth(OPERATOR_TOKEN)).status_code == 403
        first = client.get(url, headers=auth(REVIEWER_TOKEN))
        assert first.status_code == 200
        assert first.headers["cache-control"] == "no-store"
        assert verify_bundle(first.content, expected_manifest_digest=first.headers["x-audit-manifest-sha256"])["valid"]
        with zipfile.ZipFile(io.BytesIO(first.content)) as archive:
            review = json.loads(archive.read("reviews.json"))
            assert review["statuses"] == ["unreviewed"]
            assert json.loads(archive.read("run.json"))["decision_authorized"] is False
            assert json.loads(archive.read("runtime.json"))["code_revision"] == "a" * 40
        approved = client.post(
            f"/v1/sources/{sid}/reviews",
            json={"candidate_id": source["candidates"][0]["id"], "decision": "approved"},
            headers={**auth(REVIEWER_TOKEN), "Idempotency-Key": "bundle-review-001"},
        )
        assert approved.status_code == 201
        assert client.get(url, headers=auth(REVIEWER_TOKEN)).content == first.content
        assert client.get("/v1/runs/missing/audit-bundle", headers=auth(REVIEWER_TOKEN)).status_code == 404
