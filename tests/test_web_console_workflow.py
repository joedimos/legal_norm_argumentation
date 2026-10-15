from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src.service.app_with_ui import create_app
from src.service.config import Settings

OPERATOR_TOKEN = "web-operator-token-abcdefghijklmnopqrstuvwxyz-12345678"
REVIEWER_TOKEN = "web-reviewer-token-abcdefghijklmnopqrstuvwxyz-12345678"
AUDITOR_TOKEN = "web-auditor-token-abcdefghijklmnopqrstuvwxyz-123456789"
METRICS_TOKEN = "web-metrics-token-abcdefghijklmnopqrstuvwxyz-123456789"


def _settings(tmp_path) -> Settings:
    return Settings(
        environment="test",
        database_url=f"sqlite+pysqlite:///{tmp_path / 'web-workflow.db'}",
        api_keys_json=json.dumps(
            [
                {"token": OPERATOR_TOKEN, "subject": "operator-a", "roles": ["operator", "reader"]},
                {"token": REVIEWER_TOKEN, "subject": "reviewer-b", "roles": ["reviewer", "reader"]},
                {"token": AUDITOR_TOKEN, "subject": "auditor-c", "roles": ["auditor", "reader"]},
            ]
        ),
        metrics_bearer_token=METRICS_TOKEN,
        auto_create_schema=True,
        require_separation_of_duties=True,
        rate_limit_per_minute=1000,
        legifrance_request_delay_seconds=0,
    )


def _auth(token: str, idem: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    if idem:
        headers["Idempotency-Key"] = idem
    return headers


def _ingest_ambiguous_source(client: TestClient) -> dict:
    response = client.post(
        "/v1/sources",
        json={"source_id": "console-policy", "text": "The operator must retain it."},
        headers=_auth(OPERATOR_TOKEN, "console-source-001"),
    )
    assert response.status_code == 201
    return response.json()


def test_console_page_exposes_end_to_end_workflow_and_keeps_security_headers(tmp_path):
    with TestClient(create_app(_settings(tmp_path))) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Decision lab" in response.text
        assert "Human review & interpretation" in response.text
        assert "/v1/session" in response.text
        assert "/v1/console/runs" in response.text
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
        assert response.headers["cache-control"] == "no-store"


def test_console_session_reports_subject_and_roles_without_secret_material(tmp_path):
    with TestClient(create_app(_settings(tmp_path))) as client:
        assert client.get("/v1/session").status_code == 401
        response = client.get("/v1/session", headers=_auth(REVIEWER_TOKEN))
        assert response.status_code == 200
        assert response.json() == {"subject": "reviewer-b", "roles": ["reader", "reviewer"]}
        payload = json.dumps(response.json())
        assert REVIEWER_TOKEN not in payload
        assert "key_hash" not in payload


def test_console_state_persists_human_interpretation_selection(tmp_path):
    with TestClient(create_app(_settings(tmp_path))) as client:
        source = _ingest_ambiguous_source(client)
        source_id = source["id"]
        candidate_id = source["candidates"][0]["id"]
        formalization = source["formalizations"][0]
        assert formalization["unresolved"] is True
        interpretation_id = formalization["interpretations"][0]["id"]

        before = client.get(f"/v1/console/sources/{source_id}/state", headers=_auth(REVIEWER_TOKEN))
        assert before.status_code == 200
        assert before.json()["created_by"] == "operator-a"
        assert before.json()["interpretation_selections"] == {}

        review = client.post(
            f"/v1/sources/{source_id}/reviews",
            json={"candidate_id": candidate_id, "decision": "approved", "note": "Verified against source."},
            headers=_auth(REVIEWER_TOKEN, "console-review-001"),
        )
        assert review.status_code == 201

        selection = client.post(
            f"/v1/sources/{source_id}/interpretations",
            json={
                "candidate_id": candidate_id,
                "interpretation_id": interpretation_id,
                "rationale": "Resolved the anaphoric action against the provision context.",
            },
            headers=_auth(REVIEWER_TOKEN, "console-selection-001"),
        )
        assert selection.status_code == 201

        after = client.get(f"/v1/console/sources/{source_id}/state", headers=_auth(REVIEWER_TOKEN))
        assert after.status_code == 200
        assert after.json()["interpretation_selections"][candidate_id]["selected_id"] == interpretation_id
        assert "provision context" in after.json()["interpretation_selections"][candidate_id]["rationale"]


def test_console_run_history_summarizes_governed_evaluation_without_context_leakage(tmp_path):
    with TestClient(create_app(_settings(tmp_path))) as client:
        source = _ingest_ambiguous_source(client)
        source_id = source["id"]
        candidate_id = source["candidates"][0]["id"]
        interpretation_id = source["formalizations"][0]["interpretations"][0]["id"]

        assert client.post(
            f"/v1/sources/{source_id}/reviews",
            json={"candidate_id": candidate_id, "decision": "approved"},
            headers=_auth(REVIEWER_TOKEN, "run-review-001"),
        ).status_code == 201
        assert client.post(
            f"/v1/sources/{source_id}/interpretations",
            json={
                "candidate_id": candidate_id,
                "interpretation_id": interpretation_id,
                "rationale": "Human reviewer selected the source-consistent interpretation.",
            },
            headers=_auth(REVIEWER_TOKEN, "run-selection-001"),
        ).status_code == 201

        evaluated = client.post(
            f"/v1/sources/{source_id}/evaluate",
            json={"observed_actions": ["retain it"], "facts": {"private-sensitive-fact": True}, "high_stakes": True},
            headers=_auth(OPERATOR_TOKEN, "run-evaluate-001"),
        )
        assert evaluated.status_code == 200
        run_id = evaluated.json()["run_id"]

        history = client.get(
            f"/v1/console/runs?source_id={source_id}&limit=10",
            headers=_auth(AUDITOR_TOKEN),
        )
        assert history.status_code == 200
        assert history.json()["count"] == 1
        run = history.json()["runs"][0]
        assert run["id"] == run_id
        assert run["source_id"] == source_id
        assert run["decision_authorized"] is True
        assert run["readiness_status"] == "research_ready"
        assert "context_payload" not in run
        assert "result_payload" not in run
        assert "private-sensitive-fact" not in json.dumps(run)

        assert client.get(f"/v1/runs/{run_id}/audit-bundle", headers=_auth(OPERATOR_TOKEN)).status_code == 403
        assert client.get(f"/v1/runs/{run_id}/audit-bundle", headers=_auth(AUDITOR_TOKEN)).status_code == 200
