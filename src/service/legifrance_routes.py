from __future__ import annotations

import hashlib
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import desc, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.legifrance import CORPORA, LegifranceClient, LegifranceError
from src.norm_extraction import extract_norms
from src.verification import audit_extraction

from .config import Settings
from .db_models import SourceDocumentRecord
from .repository import append_audit, create_source, find_source_by_key_hash, review_statuses
from .schemas import LegifranceScrapeRequest
from .security import Principal, require_roles
from .web_ui import HTML

router = APIRouter()


def _db_session(request: Request) -> Iterator[Session]:
    database = request.app.state.db
    with database.session() as session:
        yield session


def _source_summary(session: Session, source: SourceDocumentRecord) -> dict:
    statuses = review_statuses(session, source)
    return {
        "id": source.id,
        "source_id": source.source_key,
        "content_hash": source.content_hash,
        "created_by": source.created_by,
        "created_at": source.created_at.isoformat(),
        "candidate_count": len(source.candidates),
        "review_statuses": statuses,
        "provenance": source.extraction_audit.get("provenance", {}),
    }


def _latest_source_for_key(session: Session, source_key: str) -> SourceDocumentRecord | None:
    return session.execute(
        select(SourceDocumentRecord)
        .where(SourceDocumentRecord.source_key == source_key)
        .order_by(desc(SourceDocumentRecord.created_at), desc(SourceDocumentRecord.id))
        .limit(1)
    ).scalar_one_or_none()


@router.get("/", include_in_schema=False, response_class=HTMLResponse)
def web_console() -> HTMLResponse:
    return HTMLResponse(
        HTML,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'self'; img-src 'self' data:; style-src 'unsafe-inline'; "
                "script-src 'unsafe-inline'; connect-src 'self'; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
            ),
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/v1/legifrance/corpora")
def list_legifrance_corpora(request: Request) -> dict:
    settings: Settings = request.app.state.settings
    return {
        "enabled": settings.legifrance_enabled,
        "max_articles_per_run": settings.legifrance_max_articles_per_run,
        "corpora": [
            {
                "key": spec.key,
                "label": spec.label,
                "description": spec.description,
                "legal_scope": spec.legal_scope,
                "seed_url": spec.seed_url,
            }
            for spec in CORPORA.values()
        ],
    }


@router.get("/v1/sources")
def list_sources(
    prefix: str = Query(default="", max_length=256),
    limit: int = Query(default=100, ge=1, le=250),
    session: Session = Depends(_db_session),
    principal: Principal = Depends(require_roles("reader", "operator", "reviewer", "auditor")),
) -> dict:
    del principal
    statement = select(SourceDocumentRecord)
    if prefix:
        statement = statement.where(SourceDocumentRecord.source_key.startswith(prefix))
    statement = statement.order_by(desc(SourceDocumentRecord.created_at), desc(SourceDocumentRecord.id)).limit(limit)
    sources = session.execute(statement).scalars().all()
    return {"sources": [_source_summary(session, source) for source in sources], "count": len(sources)}


@router.post("/v1/legifrance/scrape")
def scrape_legifrance(
    payload: LegifranceScrapeRequest,
    request: Request,
    session: Session = Depends(_db_session),
    principal: Principal = Depends(require_roles("operator")),
) -> dict:
    settings: Settings = request.app.state.settings
    if not settings.legifrance_enabled:
        raise HTTPException(status_code=503, detail="Légifrance ingestion is disabled")
    if payload.limit > settings.legifrance_max_articles_per_run:
        raise HTTPException(
            status_code=422,
            detail=f"limit exceeds configured maximum of {settings.legifrance_max_articles_per_run}",
        )

    client = LegifranceClient(
        timeout_seconds=settings.legifrance_timeout_seconds,
        request_delay_seconds=settings.legifrance_request_delay_seconds,
    )
    try:
        batch = client.fetch_corpus(payload.corpus, limit=payload.limit, offset=payload.offset)
    except (LegifranceError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"Légifrance collection failed: {exc}") from exc

    created = 0
    updated = 0
    unchanged = 0
    items: list[dict] = []

    for provision in batch.provisions:
        source_key = provision.source_key
        source_text = provision.source_text
        content_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        existing = find_source_by_key_hash(session, source_key, content_hash)
        if existing is not None:
            unchanged += 1
            append_audit(
                session,
                actor=principal.subject,
                action="source.refresh_checked",
                resource_type="source",
                resource_id=existing.id,
                request_id=request.state.request_id,
                payload={"source_id": source_key, "content_hash": content_hash, **provision.provenance},
            )
            items.append({"status": "unchanged", **_source_summary(session, existing)})
            continue

        previous = _latest_source_for_key(session, source_key)
        candidates = extract_norms(source_text, source_id=source_key)
        extraction_audit = dict(audit_extraction(source_text, candidates))
        extraction_audit["provenance"] = provision.provenance
        extraction_audit["language"] = "fr"
        try:
            with session.begin_nested():
                source = create_source(
                    session,
                    source_key=source_key,
                    content=source_text,
                    candidates=candidates,
                    extraction_audit=extraction_audit,
                    created_by=principal.subject,
                )
        except IntegrityError:
            source = find_source_by_key_hash(session, source_key, content_hash)
            if source is None:
                raise
            unchanged += 1
            items.append({"status": "unchanged", **_source_summary(session, source)})
            continue

        status = "updated" if previous is not None else "created"
        if status == "updated":
            updated += 1
        else:
            created += 1
        append_audit(
            session,
            actor=principal.subject,
            action="source.ingested.legifrance",
            resource_type="source",
            resource_id=source.id,
            request_id=request.state.request_id,
            payload={
                "source_id": source_key,
                "content_hash": source.content_hash,
                "candidate_ids": [candidate.get("id") for candidate in candidates],
                "previous_source_id": previous.id if previous else None,
                **provision.provenance,
            },
        )
        items.append({"status": status, **_source_summary(session, source)})

    return {
        "corpus": batch.corpus,
        "seed_url": batch.seed_url,
        "discovered": batch.discovered,
        "selected": batch.selected,
        "created": created,
        "updated": updated,
        "unchanged": unchanged,
        "errors": list(batch.errors),
        "items": items,
    }
