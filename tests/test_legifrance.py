from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src.formalization import candidate_to_interpretations
from src.legifrance import LegalProvision, LegifranceClient, ScrapeBatch, parse_article_links, parse_article_text
from src.legifrance_piste import PisteLegifranceClient, provisions_from_piste_payload
from src.service.app_with_ui import create_app
from src.service.config import Settings
from src.service.legifrance_provider import build_legifrance_client

OPERATOR_TOKEN = "legifrance-operator-token-abcdefghijklmnopqrstuvwxyz-123456"
READER_TOKEN = "legifrance-reader-token-abcdefghijklmnopqrstuvwxyz-12345678"
METRICS_TOKEN = "legifrance-metrics-token-abcdefghijklmnopqrstuvwxyz-1234567"


def _settings(tmp_path, **overrides) -> Settings:
    payload = dict(
        environment="test",
        database_url=f"sqlite+pysqlite:///{tmp_path / 'legifrance.db'}",
        api_keys_json=json.dumps(
            [
                {"token": OPERATOR_TOKEN, "subject": "collector", "roles": ["operator", "reader"]},
                {"token": READER_TOKEN, "subject": "reader", "roles": ["reader"]},
            ]
        ),
        metrics_bearer_token=METRICS_TOKEN,
        auto_create_schema=True,
        require_separation_of_duties=False,
        rate_limit_per_minute=1000,
        legifrance_request_delay_seconds=0,
        legifrance_max_articles_per_run=25,
    )
    payload.update(overrides)
    return Settings(**payload)


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_legifrance_html_parsers_are_bounded_to_article_links_and_legal_text():
    index_html = """
    <main>
      <a href="/codes/article_lc/LEGIARTI000000000001/">Article&nbsp;L420-1</a>
      <a href="https://example.org/codes/article_lc/EVIL">Article L420-2</a>
      <a href="/codes/section_lc/LEGITEXT/SECTION">Article L420-3</a>
    </main>
    """
    links = parse_article_links(
        index_html,
        base_url="https://www.legifrance.gouv.fr/codes/section_lc/LEGITEXT/SECTION/",
    )
    assert [(item.article, item.url) for item in links] == [
        ("L420-1", "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000000000001/")
    ]

    article_html = """
    <main>
      <h2>Article L420-1</h2>
      <p>Version en vigueur depuis le 01/01/2026</p>
      <p>Modifié par LOI n°2025-1</p>
      <p>Sont prohibées les pratiques qui empêchent le libre jeu de la concurrence.</p>
      <div>Code de commerce</div>
      <p>Navigation qui ne doit pas être collectée.</p>
    </main>
    """
    text = parse_article_text(article_html, article="L420-1")
    assert text == "Sont prohibées les pratiques qui empêchent le libre jeu de la concurrence."


def test_piste_payload_uses_structured_competition_section_and_in_force_articles():
    payload = {
        "articles": [{"id": "LEGIARTI000ROOT", "num": "R1", "etat": "VIGUEUR", "content": "<p>Root ignored.</p>"}],
        "sections": [
            {
                "cid": "LEGISCTA000000OTHER",
                "title": "Livre III",
                "articles": [
                    {"id": "LEGIARTI000OTHER", "num": "L300-1", "etat": "VIGUEUR", "content": "<p>Other law.</p>"}
                ],
                "sections": [],
            },
            {
                "cid": "LEGISCTA000006113741",
                "title": "Livre IV : De la liberté des prix et de la concurrence",
                "articles": [
                    {
                        "id": "LEGIARTI000COMP1",
                        "num": "L420-1",
                        "etat": "VIGUEUR",
                        "content": "<p>Sont prohibées les pratiques qui empêchent la concurrence.</p>",
                    },
                    {
                        "id": "LEGIARTI000OLD",
                        "num": "L420-0",
                        "etat": "ABROGE",
                        "content": "<p>Ancien texte sans vigueur actuelle.</p>",
                    },
                ],
                "sections": [
                    {
                        "cid": "LEGISCTA000CHILD",
                        "title": "Chapitre II",
                        "articles": [
                            {
                                "id": "LEGIARTI000COMP2",
                                "num": "L430-1",
                                "etat": "VIGUEUR",
                                "content": "<p>L'Autorité peut examiner une concentration.</p>",
                            }
                        ],
                        "sections": [],
                    }
                ],
            },
        ],
    }
    provisions = provisions_from_piste_payload(payload, corpus="competition")
    assert [item.article for item in provisions] == ["L420-1", "L430-1"]
    assert all(item.provenance["retrieval_mode"] == "piste_api" for item in provisions)
    assert all("PISTE" in item.provenance["provider"] for item in provisions)
    assert provisions[0].url.endswith("/LEGIARTI000COMP1/")


def test_auto_provider_prefers_piste_only_when_credentials_exist(tmp_path):
    html_client = build_legifrance_client(_settings(tmp_path, legifrance_source="auto"))
    assert isinstance(html_client, LegifranceClient)

    piste_client = build_legifrance_client(
        _settings(
            tmp_path,
            legifrance_source="auto",
            legifrance_piste_client_id="client-id",
            legifrance_piste_client_secret="client-secret",
        )
    )
    assert isinstance(piste_client, PisteLegifranceClient)
    assert piste_client.client_id == "client-id"
    assert piste_client.client_secret == "client-secret"


def test_french_conditions_and_exceptions_compile_to_clean_predicates():
    result = candidate_to_interpretations(
        {
            "id": "fr-norm-1",
            "source_id": "legifrance:competition:l-test",
            "source_span": [0, 100],
            "sentence": "Lorsqu’elle franchit le seuil, l'entreprise doit notifier sauf si une exemption s'applique.",
            "modality": "obligation",
            "actor": "l'entreprise",
            "action": "notifier",
            "condition": "Lorsqu’elle franchit le seuil",
            "exception": "sauf si une exemption s'applique",
            "confidence": 0.9,
            "ambiguity_flags": [],
        }
    )
    norm = result.selected()
    assert norm is not None
    assert norm.conditions[0].name == "elle franchit le seuil"
    assert norm.exceptions[0].name == "une exemption s'applique"


def test_web_console_and_corpus_catalog_are_public_but_sources_require_auth(tmp_path):
    with TestClient(create_app(_settings(tmp_path))) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "VLE–LEGATO" in page.text
        assert "frame-ancestors 'none'" in page.headers["content-security-policy"]

        corpora = client.get("/v1/legifrance/corpora")
        assert corpora.status_code == 200
        assert corpora.json()["provider"] == "html"
        assert {item["key"] for item in corpora.json()["corpora"]} == {"ip", "competition"}

        assert client.get("/v1/sources").status_code == 401


def test_scrape_ingests_french_norms_versions_changes_and_deduplicates(tmp_path, monkeypatch):
    class FakeClient:
        calls = 0

        def fetch_corpus(self, corpus: str, *, limit: int, offset: int) -> ScrapeBatch:
            assert corpus == "competition"
            assert limit == 1
            assert offset == 0
            type(self).calls += 1
            suffix = " Elle doit être signalée à l'autorité." if type(self).calls >= 3 else ""
            provision = LegalProvision(
                corpus="competition",
                corpus_label="Code de commerce — Livre IV",
                article="L420-1",
                url="https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000000000001/",
                text=(
                    "Sont prohibées les pratiques qui ont pour objet d'empêcher le libre jeu de la concurrence."
                    + suffix
                ),
            )
            return ScrapeBatch(
                corpus="competition",
                seed_url="https://www.legifrance.gouv.fr/codes/section_lc/example/",
                discovered=1,
                selected=1,
                provisions=(provision,),
                errors=(),
            )

    fake = FakeClient()
    monkeypatch.setattr("src.service.legifrance_routes.build_legifrance_client", lambda _settings: fake)

    with TestClient(create_app(_settings(tmp_path))) as client:
        payload = {"corpus": "competition", "limit": 1, "offset": 0}
        first = client.post("/v1/legifrance/scrape", json=payload, headers=_auth(OPERATOR_TOKEN))
        assert first.status_code == 200
        assert first.json()["created"] == 1
        assert first.json()["items"][0]["candidate_count"] >= 1
        assert first.json()["items"][0]["provenance"]["article"] == "L420-1"

        second = client.post("/v1/legifrance/scrape", json=payload, headers=_auth(OPERATOR_TOKEN))
        assert second.status_code == 200
        assert second.json()["unchanged"] == 1

        third = client.post("/v1/legifrance/scrape", json=payload, headers=_auth(OPERATOR_TOKEN))
        assert third.status_code == 200
        assert third.json()["updated"] == 1

        sources = client.get("/v1/sources?prefix=legifrance:competition:", headers=_auth(READER_TOKEN))
        assert sources.status_code == 200
        assert sources.json()["count"] == 2
        assert all(item["source_id"] == "legifrance:competition:l420-1" for item in sources.json()["sources"])


def test_scrape_enforces_server_side_article_limit(tmp_path):
    with TestClient(create_app(_settings(tmp_path))) as client:
        response = client.post(
            "/v1/legifrance/scrape",
            json={"corpus": "ip", "limit": 26, "offset": 0},
            headers=_auth(OPERATOR_TOKEN),
        )
        assert response.status_code == 422
        assert "configured maximum" in response.json()["detail"]
