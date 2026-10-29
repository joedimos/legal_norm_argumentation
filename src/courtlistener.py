"""Free Law Project CourtListener v4 opinions: targeted IP discovery and full text."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date
from urllib.parse import parse_qs, urlencode, urlparse

from .audit_bundle import canonical_json
from .legal_transport import ImportTransportError, LegalJSONTransport
from .legifrance_piste import html_to_text

ORIGIN = "https://www.courtlistener.com"
API = ORIGIN + "/api/rest/v4"
# A legal-topic expression; this string contains no authentication material.
TRADE_QUERY = '("trade secret" OR "Defend Trade Secrets Act")'
IP_QUERIES = {
    "copyright": '(copyright OR "17 U.S.C." OR "Digital Millennium Copyright Act")',
    "patent": '("patent infringement" OR "35 U.S.C." OR "patent validity")',
    "trademark": '(trademark OR "Lanham Act" OR "15 U.S.C. 1125")',
    "trade_secret": TRADE_QUERY,
}

# Verified against Free Law Project's Court.JURISDICTIONS definitions.
US_COURT_TYPES = {
    "F",
    "FD",
    "FB",
    "FBP",
    "FS",
    "S",
    "SA",
    "ST",
    "SS",
    "TRS",
    "TRA",
    "TRT",
    "TRX",
    "TS",
    "TA",
    "TT",
    "TSP",
    "MA",
    "MT",
}


def positive_id(value):
    if type(value) is not int or not 1 <= value <= 10**12:
        raise ImportTransportError("invalid_upstream_identifier")
    return value


@dataclass(frozen=True)
class CaseOpinion:
    opinion_id: int
    cluster_id: int
    case_name: str
    text: str
    metadata: dict

    @property
    def source_key(self):
        return f"courtlistener:opinion:{self.opinion_id}"

    @property
    def source_text(self):
        # Retrieval dates, changing search snippets and IP-topic matches do not
        # alter source identity. The full opinion is imported, never the snippet.
        return (
            f"Source: Free Law Project / CourtListener\nCase: {self.case_name}\n"
            f"Opinion ID: {self.opinion_id}\nCourt: {self.metadata.get('court')}\n"
            f"Filed: {self.metadata.get('date_filed')}\nOpinion type: {self.metadata.get('opinion_type')}\nURL: {ORIGIN}/opinion/{self.cluster_id}/\n\n{self.text}"
        )

    @property
    def provenance(self):
        return {
            "provider": "Free Law Project / CourtListener",
            "jurisdiction": "US",
            "language": "en",
            "document_type": "judicial_opinion",
            "opinion_id": self.opinion_id,
            "cluster_id": self.cluster_id,
            "url": f"{ORIGIN}/opinion/{self.cluster_id}/",
            "retrieval_mode": "courtlistener_api_v4",
            "scope_basis": "IP_keyword_search_unreviewed_relevance",
            **self.metadata,
        }


class CourtListenerClient:
    def __init__(self, *, token, transport=None, delay=13, timeout=12, deadline_seconds=120):
        if not token or not token.strip() or any(c.isspace() for c in token):
            raise ImportTransportError("courtlistener_token_missing_or_invalid", retry_after=3600)
        self.courts, self.exclusions = {}, []
        self._headers = {"Authorization": "Token " + token}
        self.transport = transport or LegalJSONTransport(
            timeout=timeout, delay=delay, deadline_seconds=deadline_seconds
        )

    def search_page(self, *, topic, since, until, cursor=None):
        if topic not in IP_QUERIES:
            raise ValueError("unknown IP topic")
        date.fromisoformat(since)
        date.fromisoformat(until)
        if since > until:
            raise ValueError("invalid case-law date window")
        query = IP_QUERIES[topic] + f" AND dateFiled:[{since} TO {until}]"
        params = {"q": query, "type": "o", "order_by": "dateFiled asc"}
        if cursor is not None:
            if not isinstance(cursor, str) or not 1 <= len(cursor) <= 4096:
                raise ImportTransportError("invalid_cursor")
            params["cursor"] = cursor
        url = API + "/search/?" + urlencode(params)
        payload = self.transport.request("GET", url, headers=self._headers)
        results = payload.get("results")
        if not isinstance(results, list) or len(results) > 100:
            raise ImportTransportError("invalid_search_results")
        pending = []
        seen = set()
        for item in results:
            if not isinstance(item, dict) or not isinstance(item.get("opinions"), list) or not item["opinions"]:
                raise ImportTransportError("search_opinion_ids_missing")
            cluster = positive_id(item.get("cluster_id"))
            if len(item["opinions"]) > 100:
                raise ImportTransportError("opinion_group_bound")
            for opinion in item["opinions"]:
                id = positive_id(opinion.get("id"))
                if id in seen:
                    continue
                seen.add(id)
                pending.append(
                    {
                        "id": id,
                        "cluster_id": cluster,
                        "case_name": item.get("caseNameFull") or item.get("caseName") or "Untitled opinion",
                        "court": item.get("court_id"),
                        "date_filed": item.get("dateFiled"),
                        "citations": item.get("citation", []),
                        "matched_topic": topic,
                    }
                )
        if len(pending) > 1000:
            raise ImportTransportError("pending_opinion_bound")
        next_cursor = None
        if payload.get("next") is not None:
            p = urlparse(payload["next"])
            values = parse_qs(p.query, strict_parsing=True)
            if (
                p.scheme != "https"
                or p.netloc != "www.courtlistener.com"
                or p.path != "/api/rest/v4/search/"
                or p.fragment
                or values.get("q") != [query]
                or len(values.get("cursor", [])) != 1
            ):
                raise ImportTransportError("unsafe_search_pagination")
            next_cursor = values["cursor"][0]
            if next_cursor == cursor or len(next_cursor) > 4096:
                raise ImportTransportError("repeated_or_invalid_cursor")
        return {
            "pending": pending,
            "cursor": next_cursor,
            "search_sha256": hashlib.sha256(canonical_json(payload)).hexdigest(),
        }

    def opinion(self, item):
        court_id = item.get("court")
        if not isinstance(court_id, str) or not re.fullmatch(r"[a-z0-9-]{1,32}", court_id):
            raise ImportTransportError("invalid_court_identifier")
        if court_id not in self.courts:
            court = self.transport.request("GET", f"{API}/courts/{court_id}/", headers=self._headers)
            if court.get("id") != court_id:
                raise ImportTransportError("court_identity_mismatch")
            self.courts[court_id] = court
        court = self.courts[court_id]
        jurisdiction = court.get("jurisdiction")
        if jurisdiction in {"I", "T", "C", "SAG"}:
            self.exclusions.append(
                {
                    "opinion_id": item["id"],
                    "court": court_id,
                    "reason": "outside_US_court_scope",
                    "court_jurisdiction": jurisdiction,
                    "court_payload_sha256": hashlib.sha256(canonical_json(court)).hexdigest(),
                }
            )
            return None
        if jurisdiction not in US_COURT_TYPES:
            raise ImportTransportError("unmapped_court_jurisdiction")
        id = positive_id(item["id"])
        payload = self.transport.request("GET", f"{API}/opinions/{id}/", headers=self._headers)
        if positive_id(payload.get("id")) != id:
            raise ImportTransportError("opinion_identity_mismatch")
        cluster = positive_id(item["cluster_id"])
        value = payload.get("cluster")
        if not (
            type(value) is int and value == cluster or isinstance(value, str) and value == f"{API}/clusters/{cluster}/"
        ):
            raise ImportTransportError("opinion_cluster_mismatch")
        field = next(
            (
                f
                for f in ("html_with_citations", "html", "html_lawbox", "html_columbia", "plain_text")
                if isinstance(payload.get(f), str) and payload[f].strip()
            ),
            None,
        )
        if field is None:
            raise ImportTransportError("opinion_full_text_missing", retry_after=3600)
        text = payload[field].strip() if field == "plain_text" else html_to_text(payload[field])
        if not 20 <= len(text) <= 200_000:
            raise ImportTransportError("opinion_text_size_bound", retry_after=3600)
        return CaseOpinion(
            id,
            positive_id(item["cluster_id"]),
            item["case_name"],
            text,
            {
                **{k: v for k, v in item.items() if k not in {"id", "cluster_id", "case_name"}},
                "court_jurisdiction": jurisdiction,
                "court_payload_sha256": hashlib.sha256(canonical_json(court)).hexdigest(),
                "opinion_type": payload.get("type"),
                "text_field": field,
                "date_modified": payload.get("date_modified"),
                "api_url": f"{API}/opinions/{id}/",
                "raw_payload_sha256": hashlib.sha256(canonical_json(payload)).hexdigest(),
            },
        )
