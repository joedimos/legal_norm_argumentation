from __future__ import annotations

from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from src.legifrance import CORPORA, LegifranceClient, LegifranceError

from .config import Settings
from .db_models import SourceDocumentRecord
from .legifrance_ingestion import ingest_legifrance_batch, source_summary
from .schemas import LegifranceScrapeRequest
from .security import Principal, require_roles
from .web_ui import HTML

router = APIRouter()


def _db_session(request: Request) -> Iterator[Session]:
    database = request.app.state.db
    with database.session() as session:
        yield session


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
    return {"sources": [source_summary(session, source) for source in sources], "count": len(sources)}


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

    return ingest_legifrance_batch(
        session,
        batch,
        actor=principal.subject,
        request_id=request.state.request_id,
    )
