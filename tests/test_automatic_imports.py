"""Synthetic/unreviewed transport fixtures; no legal citations or expert judgements."""

from __future__ import annotations

import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import jsonschema
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from src.courtlistener import API, IP_QUERIES, CaseOpinion, CourtListenerClient
from src.legal_transport import ImportTransportError, LegalJSONTransport, retry_delay
from src.legifrance import LegalProvision
from src.legifrance_ip import LegifranceIPClient
from src.service import automatic_imports as imports
from src.service.app_with_ui import create_app
from src.service.config import Settings
from src.service.database import Database
from src.service.db_models import ImportCursorRecord, SourceDocumentRecord
from src.service.repository import review_statuses, verify_audit_chain

TOKEN = "synthetic-import-test-token-0123456789-abcdefghijklmnopqrstuvwxyz"


def settings(tmp_path, **overrides):
    return Settings(
        environment="test",
        database_url=f"sqlite+pysqlite:///{tmp_path / 'imports.db'}",
        auto_create_schema=True,
        api_keys_json=json.dumps([{"token": TOKEN, "subject": "synthetic-operator", "roles": ["operator", "reader"]}]),
        require_separation_of_duties=False,
        import_batch_size=1,
        **overrides,
    )


class Clock:
    def __init__(self):
        self.value = datetime(2026, 10, 29, tzinfo=UTC)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += timedelta(seconds=seconds)


class FrenchFixture:
    version = ""

    def manifest(self, effective_date):
        return {
            "articles": [{"id": "LEGIARTI1", "num": "L111-1"}, {"id": "LEGIARTI2", "num": "L111-2"}],
            "manifest_sha256": "1" * 64,
            "excluded_noncurrent": 1,
        }

    def article(self, item, **kwargs):
        return LegalProvision(
            "ip",
            "Synthetic IP code",
            item["num"],
            "https://www.legifrance.gouv.fr/codes/article_lc/" + item["id"] + "/",
            "L'auteur doit notifier le titulaire. " + self.version,
        )


class CLFixture:
    def __init__(self):
        self.cursors = []

    def search_page(self, **kwargs):
        self.cursors.append(kwargs["cursor"])
        if kwargs["cursor"] is None:
            return {"pending": [{"id": 1}, {"id": 2}], "cursor": "next", "search_sha256": "2" * 64}
        return {"pending": [{"id": 3}], "cursor": None, "search_sha256": "3" * 64}

    def opinion(self, item):
        return CaseOpinion(
            item["id"], 10, "Synthetic IP case", "The operator must notify the copyright owner.", {"synthetic": True}
        )


def test_french_acquisition_resumes_then_versions_changed_sources_unreviewed(tmp_path):
    s, clock, fixture = settings(tmp_path), Clock(), FrenchFixture()
    db = Database(s)
    db.create_schema_for_tests()

    def factory(*_):
        return fixture

    first = imports.process_one(db, s, client_factory=factory, clock=clock)
    assert first["status"] == "in_progress" and first["summary"]["created"] == 1
    jsonschema.validate(first, json.loads(Path("schemas/import-result.schema.json").read_text()))
    assert imports.process_one(db, s, client_factory=factory, clock=clock)["status"] == "idle"
    clock.advance(16)
    second = imports.process_one(db, s, client_factory=factory, clock=clock)
    assert second["status"] == "query_cycle_completed" and second["cycle_items"] == 2
    # A later complete scan refreshes existing source versions, without timestamps in content hashes.
    clock.advance(s.import_refresh_seconds)
    unchanged = imports.process_one(db, s, client_factory=factory, clock=clock)
    assert unchanged["summary"]["unchanged"] == 1
    clock.advance(16)
    fixture.version = "Le titulaire peut autoriser une copie."
    changed = imports.process_one(db, s, client_factory=factory, clock=clock)
    assert changed["summary"]["updated"] == 1
    with db.session() as session:
        rows = session.scalars(select(SourceDocumentRecord)).all()
        assert len(rows) == 3
        assert all(all(v == "unreviewed" for v in review_statuses(session, r)) for r in rows)
        assert all(r.extraction_audit["provenance"]["import_job"] == "legifrance:ip" for r in rows)
        assert verify_audit_chain(session)["valid"]
    db.dispose()


def test_us_search_page_pending_and_next_cursor_survive_worker_restarts(tmp_path):
    s, clock, fixture = settings(tmp_path, legifrance_enabled=False, courtlistener_enabled=True), Clock(), CLFixture()
    db = Database(s)
    db.create_schema_for_tests()
    # Isolate one of the fixed jobs to demonstrate page continuation.
    original = imports.jobs
    from unittest.mock import patch

    job = {"courtlistener:copyright:backfill": original(s)["courtlistener:copyright:backfill"]}
    with patch.object(imports, "jobs", return_value=job):
        for expected in ("in_progress", "in_progress", "query_cycle_completed"):
            r = imports.process_one(db, s, client_factory=lambda *_: fixture, clock=clock)
            assert r["status"] == expected
            db.dispose()
            db = Database(s)  # Database state, not client/worker memory, carries the continuation.
            clock.advance(16)
    assert fixture.cursors == [None, "next"]
    with db.session() as session:
        assert session.scalar(select(func.count()).select_from(SourceDocumentRecord)) == 3
    db.dispose()


def test_failed_batch_rolls_back_sources_and_cursor_then_retries(tmp_path, monkeypatch):
    s, clock = settings(tmp_path), Clock()
    db = Database(s)
    db.create_schema_for_tests()
    original = imports.ingest_legal_items

    def fail_after_insert(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("synthetic secret must not escape")

    monkeypatch.setattr(imports, "ingest_legal_items", fail_after_insert)
    r = imports.process_one(db, s, client_factory=lambda *_: FrenchFixture(), clock=clock)
    assert r["status"] == "retry_scheduled" and "secret" not in json.dumps(r)
    with db.session() as session:
        assert session.scalar(select(func.count()).select_from(SourceDocumentRecord)) == 0
        assert session.get(ImportCursorRecord, "legifrance:ip").position == {}
    monkeypatch.setattr(imports, "ingest_legal_items", original)
    clock.advance(61)
    assert imports.process_one(db, s, client_factory=lambda *_: FrenchFixture(), clock=clock)["summary"]["created"] == 1
    db.dispose()


def test_429_retry_after_is_persisted_and_never_skips_a_page(tmp_path):
    s, clock = settings(tmp_path), Clock()
    db = Database(s)
    db.create_schema_for_tests()

    def limited(*_):
        raise ImportTransportError("upstream_http_429", retry_after=7200)

    r = imports.process_one(db, s, client_factory=limited, clock=clock)
    assert r["retry_seconds"] == 7200
    jsonschema.validate(r, json.loads(Path("schemas/import-result.schema.json").read_text()))
    clock.advance(7199)
    assert imports.process_one(db, s, client_factory=limited, clock=clock)["status"] == "idle"
    db.dispose()


def test_stale_worker_is_fenced_after_another_worker_takes_lease(tmp_path):
    s, clock = settings(tmp_path), Clock()
    db = Database(s)
    db.create_schema_for_tests()
    claims = []

    class Stale(FrenchFixture):
        def article(self, item, **kwargs):
            clock.advance(s.import_lease_seconds + 1)
            claims.append(imports.claim(db, s, now=clock()))
            return super().article(item, **kwargs)

    assert imports.process_one(db, s, client_factory=lambda *_: Stale(), clock=clock)["status"] == "lease_lost"
    assert claims[0] is not None
    with db.session() as session:
        assert session.scalar(select(func.count()).select_from(SourceDocumentRecord)) == 0
    db.dispose()


def test_scope_change_resets_progress_but_credential_rotation_does_not(tmp_path):
    s, clock = settings(tmp_path), Clock()
    db = Database(s)
    db.create_schema_for_tests()
    imports.process_one(db, s, client_factory=lambda *_: FrenchFixture(), clock=clock)
    clock.advance(16)
    rotated = s.model_copy(update={"courtlistener_api_token": "different-unused-token"})
    claimed = imports.claim(db, rotated, now=clock())
    assert claimed["position"]["offset"] == 1
    clock.advance(s.import_lease_seconds + 1)
    changed = s.model_copy(update={"import_start_date": "1800-01-01"})
    assert imports.claim(db, changed, now=clock())["position"] == {}
    db.dispose()


class RecordedJSON:
    def __init__(self, responses):
        self.responses, self.calls = iter(responses), []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return next(self.responses)

    def post_json(self, url, body, headers):
        return self.request("POST", url, body=body, headers=headers)


def test_courtlistener_uses_nested_opinion_ids_full_text_and_cluster_urls():
    page = {
        "results": [
            {
                "cluster_id": 55,
                "caseName": "Synthetic IP case",
                "dateFiled": "2026-01-01",
                "court_id": "synthetic",
                "citation": [],
                "opinions": [{"id": 77}],
            }
        ],
        "next": None,
    }
    body = {
        "id": 77,
        "cluster": API + "/clusters/55/",
        "type": "010combined",
        "html_with_citations": "<p>The operator must notify the copyright owner.</p>",
        "plain_text": "Wrong fallback text",
        "download_url": "https://example.invalid/never-fetch",
    }
    transport = RecordedJSON([page, {"id": "synthetic", "jurisdiction": "FD"}, body])
    client = CourtListenerClient(token="synthetic-token", transport=transport)
    result = client.search_page(topic="copyright", since="1700-01-01", until="2026-10-29")
    assert result["pending"][0]["id"] == 77
    opinion = client.opinion(result["pending"][0])
    assert opinion.source_key == "courtlistener:opinion:77"
    assert opinion.provenance["url"] == "https://www.courtlistener.com/opinion/55/"
    assert opinion.provenance["text_field"] == "html_with_citations" and "Wrong fallback" not in opinion.source_text
    assert transport.calls[0][2]["headers"]["Authorization"] == "Token synthetic-token"
    assert all("example.invalid" not in call[1] for call in transport.calls)


@pytest.mark.parametrize(
    "target",
    [
        "https://evil.invalid/api/rest/v4/search/?cursor=x",
        "http://www.courtlistener.com/api/rest/v4/search/?cursor=x",
        "https://www.courtlistener.com:444/api/rest/v4/search/?cursor=x",
    ],
)
def test_pagination_cannot_redirect_tokens_or_change_query(target):
    client = CourtListenerClient(token="synthetic-token", transport=RecordedJSON([{"results": [], "next": target}]))
    with pytest.raises(ImportTransportError):
        client.search_page(topic="patent", since="1700-01-01", until="2026-10-29")


def test_cursor_pagination_and_empty_page_are_not_a_false_completion():
    q = IP_QUERIES["patent"] + " AND dateFiled:[1700-01-01 TO 2026-10-29]"
    next_url = API + "/search/?" + urlencode({"q": q, "cursor": "opaque"})
    client = CourtListenerClient(token="synthetic-token", transport=RecordedJSON([{"results": [], "next": next_url}]))
    page = client.search_page(topic="patent", since="1700-01-01", until="2026-10-29")
    assert page["cursor"] == "opaque"
    assert parse_qs(urlparse(client.transport.calls[0][1]).query)["type"] == ["o"]


@pytest.mark.parametrize(
    "body",
    [
        {"id": 2, "plain_text": "x" * 30},
        {"id": True, "plain_text": "x" * 30},
        {"id": 1},
        {"id": 1, "plain_text": "x" * 200001},
    ],
)
def test_missing_wrong_or_oversized_opinion_never_imports_a_snippet(body):
    client = CourtListenerClient(
        token="synthetic-token",
        transport=RecordedJSON([{"id": "synthetic", "jurisdiction": "FD"}, {"cluster": API + "/clusters/2/", **body}]),
    )
    with pytest.raises(ImportTransportError):
        client.opinion({"id": 1, "cluster_id": 2, "case_name": "synthetic", "court": "synthetic"})


def test_piste_discovers_manifest_then_hydrates_article_and_preserves_provenance():
    fixture = RecordedJSON(
        [
            {"access_token": "synthetic-oauth"},
            {
                "articles": [
                    {"id": "LEGIARTI1", "num": "L111-1", "etat": "VIGUEUR"},
                    {"id": "LEGIARTI2", "num": "L111-2", "etat": "ABROGE"},
                ]
            },
            {
                "article": {
                    "id": "LEGIARTI1",
                    "texte": "<p>L'auteur doit notifier le titulaire des droits.</p>",
                    "etat": "VIGUEUR",
                }
            },
        ]
    )
    client = LegifranceIPClient(client_id="synthetic-id", client_secret="synthetic-secret", transport=fixture)
    manifest = client.manifest("2026-10-29")
    assert len(manifest["articles"]) == 1 and manifest["excluded_noncurrent"] == 1
    article = client.article(
        manifest["articles"][0], effective_date="2026-10-29", manifest_sha256=manifest["manifest_sha256"]
    )
    assert article.source_key == "legifrance:ip:l111-1" and article.provenance["consultation_date"] == "2026-10-29"
    assert fixture.calls[2][2]["body"] == {"id": "LEGIARTI1"}
    assert "synthetic-secret" not in json.dumps(article.provenance)


def test_transport_bounds_streamed_bytes_redirects_and_retry_after(monkeypatch):
    class Response(io.BytesIO):
        status, headers = 200, {}

        def release_conn(self):
            pass

    transport = LegalJSONTransport(max_bytes=5)
    monkeypatch.setattr(transport.http, "request", lambda *_args, **_kwargs: Response(b'{"too":"large"}'))
    with pytest.raises(ImportTransportError, match="response_size_bound"):
        transport.request("GET", API + "/search/")
    response = Response(b"{}")
    response.status, response.headers = 429, {"Retry-After": "7200"}
    monkeypatch.setattr(transport.http, "request", lambda *_args, **_kwargs: response)
    with pytest.raises(ImportTransportError) as exc:
        transport.request("GET", API + "/search/")
    assert exc.value.retry_after == 7200
    assert retry_delay("Thu, 29 Oct 2026 01:00:00 GMT", now=datetime(2026, 10, 29, tzinfo=UTC)) == 3600


def test_authenticated_status_reports_missing_setup_without_exposing_tokens(tmp_path):
    s = settings(tmp_path, courtlistener_enabled=True)
    with TestClient(create_app(s)) as client:
        assert client.get("/v1/imports/status").status_code == 401
        r = client.get("/v1/imports/status", headers={"Authorization": "Bearer " + TOKEN})
        assert r.status_code == 200 and len(r.json()["configured_jobs"]) == 9
        assert "CourtListener API token" in r.json()["missing_credentials"]
        assert not r.json()["decision_authorized"]
        jsonschema.validate(r.json(), json.loads(Path("schemas/import-status.schema.json").read_text()))
        assert "Automatic IP imports" in client.get("/").text


def test_disabled_and_bad_lease_configuration(tmp_path):
    s = settings(tmp_path, auto_import_enabled=False)
    assert imports.process_one(None, s)["status"] == "disabled"
    with pytest.raises(ValueError):
        settings(tmp_path, import_lease_seconds=60)


def test_provider_wide_rate_limit_blocks_every_topic_job(tmp_path):
    s, clock = settings(tmp_path, legifrance_enabled=False, courtlistener_enabled=True), Clock()
    db = Database(s)
    db.create_schema_for_tests()

    def limited(*_):
        raise ImportTransportError("upstream_http_429", retry_after=7200)

    r = imports.process_one(db, s, client_factory=limited, clock=clock)
    assert r["provider_wide_cooldown"]
    assert imports.process_one(db, s, client_factory=limited, clock=clock)["status"] == "idle"
    with db.session() as session:
        assert all(
            imports.aware(r.due_at) >= clock() + timedelta(seconds=7200)
            for r in session.scalars(select(ImportCursorRecord))
        )
    db.dispose()


@pytest.mark.parametrize("noncourt", ["I", "T", "C", "SAG"])
def test_international_court_is_explicitly_excluded_and_unknown_type_blocks(noncourt):
    item = {"id": 1, "cluster_id": 2, "case_name": "Synthetic foreign case", "court": "synthetic"}
    client = CourtListenerClient(
        token="synthetic-token", transport=RecordedJSON([{"id": "synthetic", "jurisdiction": noncourt}])
    )
    assert client.opinion(item) is None and client.exclusions[0]["reason"] == "outside_US_court_scope"
    assert len(client.transport.calls) == 1  # No opinion/PDF request for the excluded court.
    client = CourtListenerClient(
        token="synthetic-token", transport=RecordedJSON([{"id": "synthetic", "jurisdiction": "NEW"}])
    )
    with pytest.raises(ImportTransportError, match="unmapped_court_jurisdiction"):
        client.opinion(item)


def test_configured_connectors_require_credentials_and_importer_uses_piste_only(tmp_path):
    s = settings(tmp_path)
    with pytest.raises(ImportTransportError, match="piste_credentials_missing"):
        imports.build_client(s, "legifrance")
    with pytest.raises(ImportTransportError, match="courtlistener_token_missing"):
        imports.build_client(s, "courtlistener")
    with pytest.raises(ImportTransportError, match="production_piste"):
        imports.build_client(s.model_copy(update={"legifrance_piste_environment": "sandbox"}), "legifrance")


def test_transport_valid_json_duplicate_keys_redirect_and_time_bound(monkeypatch):
    class Response(io.BytesIO):
        status, headers = 200, {}

        def release_conn(self):
            pass

    transport = LegalJSONTransport()
    monkeypatch.setattr(transport.http, "request", lambda *_a, **_k: Response(b'{"results":[]}'))
    assert transport.request("GET", API + "/search/") == {"results": []}
    monkeypatch.setattr(transport.http, "request", lambda *_a, **_k: Response(b'{"x":1,"x":2}'))
    with pytest.raises(ImportTransportError, match="upstream_transport_or_json_error"):
        transport.request("GET", API + "/search/")
    response = Response(b"{}")
    response.status, response.headers = 302, {"Location": "https://example.invalid/"}
    monkeypatch.setattr(transport.http, "request", lambda *_a, **_k: response)
    with pytest.raises(ImportTransportError, match="upstream_http_302"):
        transport.request("GET", API + "/search/")
    transport.deadline = 0
    with pytest.raises(ImportTransportError, match="request_time_bound"):
        transport.request("GET", API + "/search/")
    assert retry_delay("invalid") == 60


def test_worker_cli_status_and_once_use_same_durable_scheduler(tmp_path, monkeypatch, capsys):
    import sys

    from src.service import config

    s = settings(tmp_path)
    db = Database(s)
    db.create_schema_for_tests()
    db.dispose()
    monkeypatch.setattr(config, "get_settings", lambda: s)
    monkeypatch.setattr(imports, "build_client", lambda *_: FrenchFixture())
    monkeypatch.setattr(sys, "argv", ["importer", "--status"])
    imports.main()
    assert json.loads(capsys.readouterr().out)["configured_jobs"] == ["legifrance:ip"]
    monkeypatch.setattr(sys, "argv", ["importer", "--once"])
    with pytest.raises(SystemExit) as code:
        imports.main()
    assert code.value.code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "in_progress"


def test_parallel_claim_postgres(tmp_path, monkeypatch):
    url = os.environ.get("VLE_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("PostgreSQL integration is configured in CI")
    s = settings(tmp_path).model_copy(update={"database_url": url})
    db = Database(s)
    import uuid

    from sqlalchemy import delete

    key = "synthetic-claim-test:" + str(uuid.uuid4())
    monkeypatch.setattr(imports, "jobs", lambda _: {key: {"provider": "legifrance", "corpus": "ip"}})
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: imports.claim(db, s, now=datetime.now(UTC)), range(2)))
        assert sum(r is not None for r in results) == 1
    finally:
        with db.session() as session:
            session.execute(delete(ImportCursorRecord).where(ImportCursorRecord.key == key))
        db.dispose()


def test_french_manifest_never_silently_ignores_an_unsupported_branch():
    transport = RecordedJSON(
        [
            {"access_token": "synthetic"},
            {"articles": [{"id": "LEGIARTI1", "num": "L111-1", "etat": "VIGUEUR"}], "sections": ["unsupported"]},
        ]
    )
    client = LegifranceIPClient(client_id="synthetic", client_secret="synthetic", transport=transport)
    with pytest.raises(ImportTransportError, match="invalid_manifest_structure"):
        client.manifest("2026-10-29")


def test_provider_deadline_discards_entire_hydration_batch_and_resumes_unreviewed(tmp_path, monkeypatch):
    from src.http_transport import HTTPTransportError

    s, clock = settings(tmp_path).model_copy(update={"import_batch_size": 2}), Clock()
    db = Database(s)
    db.create_schema_for_tests()
    bounded = LegalJSONTransport(timeout=1)

    def timeout(*_a, **_kw):
        raise HTTPTransportError("timeout")

    monkeypatch.setattr(bounded.http, "request", timeout)

    class DeadlineFixture(FrenchFixture):
        def article(self, item, **kwargs):
            if item["id"] == "LEGIARTI2":
                bounded.request("GET", API + "/search/")
            return super().article(item, **kwargs)

    result = imports.process_one(db, s, client_factory=lambda *_: DeadlineFixture(), clock=clock)
    assert result["status"] == "retry_scheduled" and result["reason"] == "request_time_bound"
    assert not result["decision_authorized"]
    with db.session() as session:
        record = session.get(ImportCursorRecord, "legifrance:ip")
        assert record.position == {} and record.last_success_at is None
        assert session.scalar(select(func.count()).select_from(SourceDocumentRecord)) == 0
    clock.advance(61)
    resumed = imports.process_one(db, s, client_factory=lambda *_: FrenchFixture(), clock=clock)
    assert resumed["status"] == "query_cycle_completed" and resumed["summary"]["created"] == 2
    with db.session() as session:
        assert all(all(v == "unreviewed" for v in review_statuses(session, r))
                   for r in session.scalars(select(SourceDocumentRecord)))
        assert verify_audit_chain(session)["valid"]
    db.dispose()
