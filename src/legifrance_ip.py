"""Pinned-date IP-code manifest plus explicit article hydration from official PISTE."""

from __future__ import annotations

import hashlib
import re
from datetime import date
from urllib.parse import urlencode

from .audit_bundle import canonical_json
from .legal_transport import ImportTransportError, LegalJSONTransport
from .legifrance import CORPORA
from .legifrance_piste import (
    PisteLegalProvision,
    _all_articles,
)


class IPCodeProvision(PisteLegalProvision):
    def __init__(self, *, metadata, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "metadata", metadata)

    @property
    def provenance(self):
        return {
            **super().provenance,
            "jurisdiction": "FR",
            "language": "fr",
            "document_type": "code_article",
            **self.metadata,
        }


class LegifranceIPClient:
    def __init__(
        self, *, client_id, client_secret, environment="production", transport=None, timeout=12, deadline_seconds=120
    ):
        if not client_id or not client_secret:
            raise ImportTransportError("piste_credentials_missing", retry_after=3600)
        if environment not in {"production", "sandbox"}:
            raise ValueError("invalid PISTE environment")
        prefix = "sandbox-" if environment == "sandbox" else ""
        self.base = f"https://{prefix}api.piste.gouv.fr/dila/legifrance/lf-engine-app"
        self.oauth = f"https://{prefix}oauth.piste.gouv.fr/api/oauth/token"
        self.transport = transport or LegalJSONTransport(
            timeout=timeout, max_bytes=40_000_000, deadline_seconds=deadline_seconds
        )
        self._id, self._secret, self._token = client_id, client_secret, None

    def _post(self, path, body):
        # New connector per bounded worker batch: short-lived tokens are never
        # persisted. One refresh on 401 is safe for these read-only POST methods.
        for attempt in range(2):
            if self._token is None:
                payload = self.transport.request(
                    "POST",
                    self.oauth,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    body=urlencode(
                        {
                            "grant_type": "client_credentials",
                            "scope": "openid",
                            "client_id": self._id,
                            "client_secret": self._secret,
                        }
                    ).encode(),
                )
                self._token = payload.get("access_token")
                if not isinstance(self._token, str) or not self._token:
                    raise ImportTransportError("piste_token_missing")
            try:
                return self.transport.post_json(self.base + path, body, {"Authorization": "Bearer " + self._token})
            except ImportTransportError as exc:
                if str(exc) != "upstream_http_401" or attempt:
                    raise
                self._token = None
        raise ImportTransportError("piste_authentication_failed")

    def manifest(self, effective_date):
        date.fromisoformat(effective_date)
        payload = self._post("/consult/legiPart", {"textId": "LEGITEXT000006069414", "date": effective_date})
        # Reject unsupported branches instead of silently importing a partial tree.
        stack, visited = [(payload, 0)], 0
        while stack:
            node, depth = stack.pop()
            visited += 1
            if visited > 20000 or depth > 32:
                raise ImportTransportError("manifest_structure_bound")
            if (
                not isinstance(node, dict)
                or not isinstance(node.get("articles", []), list)
                or not isinstance(node.get("sections", []), list)
                or any(not isinstance(a, dict) for a in node.get("articles", []))
            ):
                raise ImportTransportError("invalid_manifest_structure")
            stack.extend((child, depth + 1) for child in node.get("sections", []))
        raw = _all_articles(payload)
        if not raw or len(raw) > 10000:
            raise ImportTransportError("ip_manifest_missing_or_bound")
        selected, seen, excluded = [], set(), 0
        for a in raw:
            if a.get("etat") and not str(a["etat"]).upper().startswith("VIGUEUR"):
                excluded += 1
                continue
            id, num = a.get("id"), a.get("num")
            if (
                not isinstance(id, str)
                or not re.fullmatch(r"LEGIARTI[0-9A-Za-z]{1,32}", id)
                or not isinstance(num, str)
                or not num.strip()
            ):
                raise ImportTransportError("invalid_manifest_article")
            if id not in seen:
                selected.append({"id": id, "num": num})
                seen.add(id)
        if not selected:
            raise ImportTransportError("no_current_ip_articles")
        return {
            "articles": selected,
            "excluded_noncurrent": excluded,
            "manifest_sha256": hashlib.sha256(canonical_json(payload)).hexdigest(),
        }

    def article(self, item, *, effective_date, manifest_sha256):
        id = item["id"]
        if not isinstance(id, str) or not re.fullmatch(r"LEGIARTI[0-9A-Za-z]{1,32}", id):
            raise ImportTransportError("invalid_article_identifier")
        payload = self._post("/consult/getArticle", {"id": id})
        article = payload.get("article", payload)
        if not isinstance(article, dict) or article.get("id") != id:
            raise ImportTransportError("article_identity_mismatch")
        from .legifrance_piste import html_to_text

        text = article.get("texte") or article.get("content") or article.get("texteHtml")
        if not isinstance(text, str):
            raise ImportTransportError("article_text_missing")
        text = html_to_text(text)
        if not 20 <= len(text) <= 200000:
            raise ImportTransportError("article_text_size_bound")
        return IPCodeProvision(
            corpus="ip",
            corpus_label=CORPORA["ip"].label,
            article=item["num"],
            url=f"https://www.legifrance.gouv.fr/codes/article_lc/{id}/",
            text=text,
            metadata={
                "official_article_id": id,
                "consultation_date": effective_date,
                "manifest_sha256": manifest_sha256,
                "etat": article.get("etat"),
                "date_debut": article.get("dateDebut"),
                "date_fin": article.get("dateFin"),
                "raw_payload_sha256": hashlib.sha256(canonical_json(payload)).hexdigest(),
            },
        )
