from __future__ import annotations

from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from src.legifrance import CORPORA, LegifranceError

from .config import Settings
from .db_models import LifecycleRunRecord, SourceDocumentRecord
from .legifrance_ingestion import ingest_legifrance_batch, source_summary
from .legifrance_provider import build_legifrance_client, provider_name
from .repository import get_source, interpretation_selections
from .schemas import LegifranceScrapeRequest
from .security import Principal, require_roles
from .web_ui_v2 import HTML

router = APIRouter()


def _db_session(request: Request) -> Iterator[Session]:
    database = request.app.state.db
    with database.session() as session:
        yield session


def _run_summary(run: LifecycleRunRecord) -> dict:
    result = run.result_payload or {}
    readiness = result.get("readiness") or {}
    compliance = result.get("compliance") or {}
    return {
        "id": run.id,
        "source_id": run.source_id,
        "requested_by": run.requested_by,
        "high_stakes": run.high_stakes,
        "decision_authorized": run.decision_authorized,
        "created_at": run.created_at.isoformat(),
        "readiness_status": readiness.get("status"),
        "failed_gates": readiness.get("failed_gates", []),
        "compliance_status": compliance.get("status"),
    }


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


@router.get("/v1/session")
def session_info(
    principal: Principal = Depends(require_roles("reader", "operator", "reviewer", "auditor")),
) -> dict:
    """Expose only the authenticated subject and roles needed to drive the console UX."""
    return {"subject": principal.subject, "roles": sorted(principal.roles)}


@router.get("/v1/legifrance/corpora")
def list_legifrance_corpora(request: Request) -> dict:
    settings: Settings = request.app.state.settings
    return {
        "enabled": settings.legifrance_enabled,
        "provider": provider_name(settings),
        "configured_source": settings.legifrance_source,
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


@router.get("/v1/console/sources/{source_id}/state")
def source_console_state(
    source_id: str,
    session: Session = Depends(_db_session),
    principal: Principal = Depends(require_roles("reader", "operator", "reviewer", "auditor")),
) -> dict:
    del principal
    source = get_source(session, source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="source not found")
    return {
        "source_id": source.id,
        "created_by": source.created_by,
        "interpretation_selections": interpretation_selections(session, source),
    }


@router.get("/v1/console/runs")
def list_console_runs(
    source_id: str | None = Query(default=None, max_length=36),
    limit: int = Query(default=50, ge=1, le=100),
    session: Session = Depends(_db_session),
    principal: Principal = Depends(require_roles("reader", "operator", "reviewer", "auditor")),
) -> dict:
    del principal
    statement = select(LifecycleRunRecord)
    if source_id:
        statement = statement.where(LifecycleRunRecord.source_id == source_id)
    statement = statement.order_by(desc(LifecycleRunRecord.created_at), desc(LifecycleRunRecord.id)).limit(limit)
    runs = session.execute(statement).scalars().all()
    return {"runs": [_run_summary(run) for run in runs], "count": len(runs)}


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

    try:
        client = build_legifrance_client(settings)
        batch = client.fetch_corpus(payload.corpus, limit=payload.limit, offset=payload.offset)
    except (LegifranceError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"Légifrance collection failed: {exc}") from exc

    return ingest_legifrance_batch(
        session,
        batch,
        actor=principal.subject,
        request_id=request.state.request_id,
    )


@router.get("/v1/imports/status")
def automatic_import_status(
    request: Request,
    session: Session = Depends(_db_session),
    principal: Principal = Depends(require_roles("reader", "operator", "reviewer", "auditor")),
) -> dict:
    del principal
    from .automatic_imports import import_status
    return import_status(session, request.app.state.settings)
