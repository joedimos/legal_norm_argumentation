from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any, Final, Literal

import urllib3
from urllib3.util import Retry, Timeout

from .legifrance import CORPORA, LEGIFRANCE_ORIGIN, LegifranceError, ScrapeBatch

_PRODUCTION_OAUTH: Final = "https://oauth.piste.gouv.fr/api/oauth/token"
_PRODUCTION_API: Final = "https://api.piste.gouv.fr/dila/legifrance/lf-engine-app"
_SANDBOX_OAUTH: Final = "https://sandbox-oauth.piste.gouv.fr/api/oauth/token"
_SANDBOX_API: Final = "https://sandbox-api.piste.gouv.fr/dila/legifrance/lf-engine-app"

_TEXT_IDS: Final = {
    "ip": "LEGITEXT000006069414",
    "competition": "LEGITEXT000005634379",
}
_COMPETITION_SECTION_CID: Final = "LEGISCTA000006113741"
_COMPETITION_TITLE_MARKER: Final = "liberté des prix et de la concurrence"


class _HTMLText(HTMLParser):
    _BREAKS = {"br", "div", "li", "p", "section", "tr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag in {"script", "style", "svg", "noscript"}:
            self._ignored += 1
        elif tag in self._BREAKS and not self._ignored:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "svg", "noscript"}:
            self._ignored = max(0, self._ignored - 1)
        elif tag in self._BREAKS and not self._ignored:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored and data.strip():
            self.parts.append(data)


def html_to_text(value: str) -> str:
    parser = _HTMLText()
    parser.feed(value)
    lines = [re.sub(r"\s+", " ", line).strip() for line in "".join(parser.parts).splitlines()]
    return "\n".join(line for line in lines if line).strip()


@dataclass(frozen=True)
class PisteLegalProvision:
    corpus: str
    corpus_label: str
    article: str
    url: str
    text: str

    @property
    def source_key(self) -> str:
        compact = re.sub(r"\s+", "", self.article).lower()
        return f"legifrance:{self.corpus}:{compact}"

    @property
    def source_text(self) -> str:
        return (
            "Source officielle: Légifrance\n"
            f"Corpus: {self.corpus_label}\n"
            f"Article: {self.article}\n"
            f"URL: {self.url}\n\n"
            f"{self.text.strip()}"
        )

    @property
    def provenance(self) -> dict[str, str]:
        return {
            "provider": "Légifrance PISTE / DILA",
            "corpus": self.corpus,
            "corpus_label": self.corpus_label,
            "article": self.article,
            "url": self.url,
            "retrieval_mode": "piste_api",
        }


def _walk_articles(section: dict[str, Any]) -> list[dict[str, Any]]:
    articles = [item for item in section.get("articles", []) if isinstance(item, dict)]
    for child in section.get("sections", []):
        if isinstance(child, dict):
            articles.extend(_walk_articles(child))
    return articles


def _find_section(sections: list[Any], *, cid: str, title_marker: str) -> dict[str, Any] | None:
    folded_marker = title_marker.casefold()
    for raw in sections:
        if not isinstance(raw, dict):
            continue
        if raw.get("cid") == cid or folded_marker in str(raw.get("title", "")).casefold():
            return raw
        found = _find_section(raw.get("sections", []), cid=cid, title_marker=title_marker)
        if found is not None:
            return found
    return None


def _all_articles(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = [item for item in payload.get("articles", []) if isinstance(item, dict)]
    for section in payload.get("sections", []):
        if isinstance(section, dict):
            result.extend(_walk_articles(section))
    return result


def _competition_articles(payload: dict[str, Any]) -> list[dict[str, Any]]:
    target = _find_section(
        payload.get("sections", []),
        cid=_COMPETITION_SECTION_CID,
        title_marker=_COMPETITION_TITLE_MARKER,
    )
    if target is not None:
        return _walk_articles(target)

    selected: list[dict[str, Any]] = []
    for article in _all_articles(payload):
        path_titles = article.get("pathTitle") or []
        joined = " ".join(str(value) for value in path_titles).casefold()
        if "livre iv" in joined and _COMPETITION_TITLE_MARKER in joined:
            selected.append(article)
    return selected


def provisions_from_piste_payload(payload: dict[str, Any], *, corpus: str) -> tuple[PisteLegalProvision, ...]:
    if corpus not in CORPORA:
        raise ValueError(f"unknown Légifrance corpus: {corpus}")
    articles = _competition_articles(payload) if corpus == "competition" else _all_articles(payload)
    spec = CORPORA[corpus]
    provisions: list[PisteLegalProvision] = []
    seen: set[str] = set()

    for article in articles:
        article_id = str(article.get("id") or "").strip()
        number = str(article.get("num") or "").strip()
        content = str(article.get("content") or "")
        status = str(article.get("etat") or "").upper()
        if not article_id.startswith("LEGIARTI") or not number or not content:
            continue
        if status and not status.startswith("VIGUEUR"):
            continue
        text = html_to_text(content)
        if len(text) < 20:
            continue
        key = article_id.casefold()
        if key in seen:
            continue
        seen.add(key)
        provisions.append(
            PisteLegalProvision(
                corpus=corpus,
                corpus_label=spec.label,
                article=number,
                url=f"{LEGIFRANCE_ORIGIN}/codes/article_lc/{article_id}/",
                text=text,
            )
        )
    return tuple(provisions)


@dataclass(frozen=True)
class PisteEndpoints:
    oauth_url: str
    api_base: str


class PisteLegifranceClient:
    """Official OAuth2 PISTE provider for deterministic LEGI text consultation."""

    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        environment: Literal["production", "sandbox"] = "production",
        timeout_seconds: float = 12.0,
        max_response_bytes: int = 40_000_000,
    ) -> None:
        if not client_id or not client_secret:
            raise ValueError("PISTE client id and secret are required")
        self.client_id = client_id
        self.client_secret = client_secret
        self.max_response_bytes = max_response_bytes
        self.endpoints = PisteEndpoints(
            oauth_url=_PRODUCTION_OAUTH if environment == "production" else _SANDBOX_OAUTH,
            api_base=_PRODUCTION_API if environment == "production" else _SANDBOX_API,
        )
        self._timeout = Timeout(connect=timeout_seconds, read=timeout_seconds)
        self._retries = Retry(
            total=2,
            connect=2,
            read=2,
            status=2,
            backoff_factor=0.4,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"POST"}),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        self._http = urllib3.PoolManager(num_pools=2, cert_reqs="CERT_REQUIRED")

    def _json_response(self, response: Any, *, operation: str) -> dict[str, Any]:
        if response.status != 200:
            raise LegifranceError(f"PISTE {operation} returned HTTP {response.status}")
        if len(response.data) > self.max_response_bytes:
            raise LegifranceError(f"PISTE {operation} response exceeded configured size limit")
        try:
            payload = json.loads(response.data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LegifranceError(f"PISTE {operation} returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise LegifranceError(f"PISTE {operation} returned an unexpected JSON shape")
        return payload

    def _access_token(self) -> str:
        response = self._http.request(
            "POST",
            self.endpoints.oauth_url,
            fields={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": "openid",
            },
            encode_multipart=False,
            timeout=self._timeout,
            retries=self._retries,
            redirect=False,
            preload_content=True,
        )
        payload = self._json_response(response, operation="OAuth token")
        token = str(payload.get("access_token") or "")
        if not token:
            raise LegifranceError("PISTE OAuth response did not contain an access token")
        return token

    def _legi_part(self, *, text_id: str) -> dict[str, Any]:
        token = self._access_token()
        effective_date = datetime.now(UTC).strftime("%Y-%m-%dT00:00:00.000Z")
        body = json.dumps({"textId": text_id, "date": effective_date}, separators=(",", ":")).encode()
        response = self._http.request(
            "POST",
            f"{self.endpoints.api_base}/consult/legiPart",
            body=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=self._timeout,
            retries=self._retries,
            redirect=False,
            preload_content=True,
        )
        return self._json_response(response, operation="LEGI consultation")

    def fetch_corpus(self, corpus: str, *, limit: int = 20, offset: int = 0) -> ScrapeBatch:
        if corpus not in CORPORA:
            raise ValueError(f"unknown Légifrance corpus: {corpus}")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if not 0 <= offset <= 100_000:
            raise ValueError("offset must be between 0 and 100000")

        payload = self._legi_part(text_id=_TEXT_IDS[corpus])
        provisions = provisions_from_piste_payload(payload, corpus=corpus)
        if not provisions:
            raise LegifranceError(f"PISTE returned no in-force articles for {corpus}")
        selected = provisions[offset : offset + limit]
        spec = CORPORA[corpus]
        return ScrapeBatch(
            corpus=corpus,
            seed_url=spec.seed_url,
            discovered=len(provisions),
            selected=len(selected),
            provisions=selected,  # type: ignore[arg-type]
            errors=(),
        )
