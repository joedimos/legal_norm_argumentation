from __future__ import annotations

import hashlib
import logging
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy.orm import Session

from src.audit_bundle import create_bundle, digest, runtime_evidence
from src.formalization import formalize_candidates
from src.lifecycle import run_lifecycle
from src.norm_extraction import extract_norms
from src.normative_model import FactContext
from src.review import semantic_fingerprint
from src.verification import audit_extraction

from .config import Settings, get_settings
from .database import Database
from .logging import configure_logging
from .metrics import LIFECYCLE_RUNS
from .middleware import MaxBodySizeMiddleware, request_context_middleware
from .rate_limit import DatabaseRateLimiter
from .repository import (
    add_interpretation_selection,
    add_review,
    append_audit,
    audit_for_resource,
    canonical_hash,
    create_run,
    create_source,
    find_source_by_key_hash,
    get_candidate,
    get_idempotent,
    get_run,
    get_source,
    interpretation_selections,
    latest_review,
    review_evidence_snapshot,
    review_statuses,
    store_idempotent,
    verify_audit_chain,
)
from .schemas import EvaluateRequest, InterpretationSelectionRequest, ReviewCreateRequest, SourceCreateRequest
from .security import APIKeyStore, Principal, require_roles, verify_metrics_token
from .telemetry import configure_telemetry

logger = logging.getLogger("vle.api")


def _formalization_preview(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            item = formalize_candidates([candidate])[0]
            preview.append(
                {
                    "candidate_id": item.source_norm_id,
                    "selected_id": item.selected_id,
                    "unresolved": item.unresolved,
                    "interpretations": [value.to_dict() for value in item.interpretations],
                }
            )
        except Exception as exc:
            preview.append({"candidate_id": candidate.get("id"), "error": type(exc).__name__})
    return preview


def _source_payload(session: Session, source) -> dict[str, Any]:
    return {
        "id": source.id,
        "source_id": source.source_key,
        "content_hash": source.content_hash,
        "created_by": source.created_by,
        "created_at": source.created_at.isoformat(),
        "candidates": source.candidates,
        "extraction_audit": source.extraction_audit,
        "review_statuses": review_statuses(session, source),
        "formalizations": _formalization_preview(source.candidates),
    }


def _require_idempotency(value: str | None) -> str:
    if value is None or not (8 <= len(value) <= 128):
        raise HTTPException(status_code=400, detail="Idempotency-Key header must contain 8-128 characters")
    return value


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    configure_telemetry(settings)

    app = FastAPI(
        title="Legal Norm Argumentation Service",
        version=settings.service_version,
        docs_url="/docs" if settings.allow_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.allow_docs else None,
    )
    database = Database(settings)
    if settings.auto_create_schema:
        database.create_schema_for_tests()
    app.state.settings = settings
    app.state.db = database
    app.state.api_key_store = APIKeyStore(settings.api_keys)
    app.state.rate_limiter = DatabaseRateLimiter(database, settings.rate_limit_per_minute)

    app.add_middleware(MaxBodySizeMiddleware, max_bytes=settings.max_request_bytes)
    app.middleware("http")(request_context_middleware)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "X-Request-ID"],
            max_age=600,
        )

    def db_session():
        with database.session() as session:
            yield session

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        logger.exception("unhandled_error")
        return JSONResponse(
            status_code=500,
            content={"detail": "internal server error", "request_id": getattr(request.state, "request_id", None)},
        )

    @app.get("/health/live", include_in_schema=False)
    def live() -> dict[str, str]:
        return {"status": "ok", "version": settings.service_version}

    @app.get("/health/ready", include_in_schema=False)
    def ready():
        healthy = database.ready(require_migrations=settings.environment == "production")
        if not healthy:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return {"status": "ready"}

    @app.get("/metrics", include_in_schema=False, dependencies=[Depends(verify_metrics_token)])
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/sources", status_code=201)
    def ingest_source(
        payload: SourceCreateRequest,
        request: Request,
        session: Session = Depends(db_session),
        principal: Principal = Depends(require_roles("operator")),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        key = _require_idempotency(idempotency_key)
        request_hash = canonical_hash(payload.model_dump(mode="json"))
        endpoint = "POST:/v1/sources"
        try:
            cached = get_idempotent(session, actor=principal.subject, endpoint=endpoint, key=key, request_hash=request_hash)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if cached:
            return JSONResponse(status_code=cached.status_code, content=cached.response_payload)

        content_hash = hashlib.sha256(payload.text.encode("utf-8")).hexdigest()
        existing = find_source_by_key_hash(session, payload.source_id, content_hash)
        if existing:
            response_payload = _source_payload(session, existing)
            store_idempotent(
                session,
                actor=principal.subject,
                endpoint=endpoint,
                key=key,
                request_hash=request_hash,
                response_payload=response_payload,
                status_code=200,
                ttl_seconds=settings.idempotency_ttl_seconds,
            )
            return JSONResponse(status_code=200, content=response_payload)

        candidates = extract_norms(payload.text, source_id=payload.source_id)
        extraction_audit = audit_extraction(payload.text, candidates)
        source = create_source(
            session,
            source_key=payload.source_id,
            content=payload.text,
            candidates=candidates,
            extraction_audit=extraction_audit,
            created_by=principal.subject,
        )
        append_audit(
            session,
            actor=principal.subject,
            action="source.ingested",
            resource_type="source",
            resource_id=source.id,
            request_id=request.state.request_id,
            payload={"source_id": payload.source_id, "content_hash": source.content_hash, "candidate_ids": [c.get("id") for c in candidates]},
        )
        response_payload = _source_payload(session, source)
        store_idempotent(
            session,
            actor=principal.subject,
            endpoint=endpoint,
            key=key,
            request_hash=request_hash,
            response_payload=response_payload,
            status_code=201,
            ttl_seconds=settings.idempotency_ttl_seconds,
        )
        return JSONResponse(status_code=201, content=response_payload)

    @app.get("/v1/sources/{source_id}")
    def read_source(
        source_id: str,
        session: Session = Depends(db_session),
        principal: Principal = Depends(require_roles("reader", "operator", "reviewer", "auditor")),
    ):
        source = get_source(session, source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="source not found")
        return _source_payload(session, source)

    @app.post("/v1/sources/{source_id}/reviews", status_code=201)
    def review_candidate(
        source_id: str,
        payload: ReviewCreateRequest,
        request: Request,
        session: Session = Depends(db_session),
        principal: Principal = Depends(require_roles("reviewer")),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        key = _require_idempotency(idempotency_key)
        endpoint = f"POST:/v1/sources/{source_id}/reviews"
        request_hash = canonical_hash(payload.model_dump(mode="json"))
        try:
            cached = get_idempotent(session, actor=principal.subject, endpoint=endpoint, key=key, request_hash=request_hash)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if cached:
            return JSONResponse(status_code=cached.status_code, content=cached.response_payload)

        source = get_source(session, source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="source not found")
        if settings.require_separation_of_duties and source.created_by == principal.subject:
            raise HTTPException(status_code=409, detail="separation of duties forbids self-review")
        candidate = get_candidate(source, payload.candidate_id)
        if candidate is None:
            raise HTTPException(status_code=404, detail="candidate not found")
        record = add_review(
            session,
            source=source,
            candidate=candidate,
            decision=payload.decision,
            reviewer=principal.subject,
            note=payload.note,
        )
        append_audit(
            session,
            actor=principal.subject,
            action="candidate.reviewed",
            resource_type="source",
            resource_id=source.id,
            request_id=request.state.request_id,
            payload={"candidate_id": payload.candidate_id, "decision": payload.decision, "fingerprint": record.candidate_fingerprint},
        )
        response_payload = {
            "review_id": record.id,
            "candidate_id": record.candidate_id,
            "decision": record.decision,
            "fingerprint": record.candidate_fingerprint,
        }
        store_idempotent(
            session,
            actor=principal.subject,
            endpoint=endpoint,
            key=key,
            request_hash=request_hash,
            response_payload=response_payload,
            status_code=201,
            ttl_seconds=settings.idempotency_ttl_seconds,
        )
        return JSONResponse(status_code=201, content=response_payload)

    @app.post("/v1/sources/{source_id}/interpretations", status_code=201)
    def select_interpretation_endpoint(
        source_id: str,
        payload: InterpretationSelectionRequest,
        request: Request,
        session: Session = Depends(db_session),
        principal: Principal = Depends(require_roles("reviewer")),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        key = _require_idempotency(idempotency_key)
        endpoint = f"POST:/v1/sources/{source_id}/interpretations"
        request_hash = canonical_hash(payload.model_dump(mode="json"))
        try:
            cached = get_idempotent(session, actor=principal.subject, endpoint=endpoint, key=key, request_hash=request_hash)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if cached:
            return JSONResponse(status_code=cached.status_code, content=cached.response_payload)
        source = get_source(session, source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="source not found")
        if settings.require_separation_of_duties and source.created_by == principal.subject:
            raise HTTPException(status_code=409, detail="separation of duties forbids self-selection")
        candidate = get_candidate(source, payload.candidate_id)
        if candidate is None:
            raise HTTPException(status_code=404, detail="candidate not found")
        review = latest_review(session, source.id, payload.candidate_id)
        if review is None or review.decision != "approved" or review.candidate_fingerprint != semantic_fingerprint(candidate):
            raise HTTPException(status_code=409, detail="a current approved human review is required before interpretation selection")
        formal_set = formalize_candidates([candidate])[0]
        available = {item.id for item in formal_set.interpretations}
        if payload.interpretation_id not in available:
            raise HTTPException(status_code=422, detail="interpretation id is not valid for the current candidate")
        selection = add_interpretation_selection(
            session,
            source=source,
            candidate=candidate,
            interpretation_id=payload.interpretation_id,
            rationale=payload.rationale,
            reviewer=principal.subject,
        )
        append_audit(
            session,
            actor=principal.subject,
            action="interpretation.selected",
            resource_type="source",
            resource_id=source.id,
            request_id=request.state.request_id,
            payload={"candidate_id": payload.candidate_id, "interpretation_id": payload.interpretation_id, "fingerprint": selection.candidate_fingerprint},
        )
        response_payload = {
            "selection_id": selection.id,
            "candidate_id": selection.candidate_id,
            "interpretation_id": selection.interpretation_id,
        }
        store_idempotent(
            session,
            actor=principal.subject,
            endpoint=endpoint,
            key=key,
            request_hash=request_hash,
            response_payload=response_payload,
            status_code=201,
            ttl_seconds=settings.idempotency_ttl_seconds,
        )
        return JSONResponse(status_code=201, content=response_payload)

    @app.post("/v1/sources/{source_id}/evaluate")
    def evaluate_source(
        source_id: str,
        payload: EvaluateRequest,
        request: Request,
        session: Session = Depends(db_session),
        principal: Principal = Depends(require_roles("operator")),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        key = _require_idempotency(idempotency_key)
        endpoint = f"POST:/v1/sources/{source_id}/evaluate"
        request_payload = payload.model_dump(mode="json")
        request_hash = canonical_hash(request_payload)
        try:
            cached = get_idempotent(session, actor=principal.subject, endpoint=endpoint, key=key, request_hash=request_hash)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if cached:
            return JSONResponse(status_code=cached.status_code, content=cached.response_payload)
        source = get_source(session, source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="source not found")
        context = FactContext(
            facts=payload.facts,
            actor=payload.actor,
            observed_actions=tuple(payload.observed_actions),
            at_date=payload.at_date,
            metadata=payload.metadata,
        )
        statuses = review_statuses(session, source)
        selections = interpretation_selections(session, source)
        result = run_lifecycle(
            text=source.content,
            source_id=source.source_key,
            candidates=source.candidates,
            extraction_audit=source.extraction_audit,
            context=context,
            review_statuses=statuses,
            interpretation_selections=selections,
        )
        result["review_evidence"] = review_evidence_snapshot(session, source)
        result["runtime_evidence"] = runtime_evidence(revision=settings.build_revision)
        ready = result.get("readiness", {}).get("status") == "research_ready"
        authorized = (not payload.high_stakes) or ready
        run = create_run(
            session,
            source_id=source.id,
            requested_by=principal.subject,
            high_stakes=payload.high_stakes,
            decision_authorized=authorized,
            request_payload=request_payload,
            context_payload=request_payload,
            result_payload=result,
        )
        append_audit(
            session,
            actor=principal.subject,
            action="lifecycle.evaluated",
            resource_type="run",
            resource_id=run.id,
            request_id=request.state.request_id,
            payload={"source_id": source.id, "decision_authorized": authorized, "readiness": result.get("readiness")},
        )
        LIFECYCLE_RUNS.labels(str(authorized).lower(), str(payload.high_stakes).lower()).inc()
        status_code = 200 if authorized else 409
        if authorized:
            response_payload = {"run_id": run.id, "decision_authorized": True, **result}
        else:
            response_payload = {
                "run_id": run.id,
                "decision_authorized": False,
                "readiness": result.get("readiness"),
                "detail": "high-stakes decision blocked by evidence/readiness gates",
            }
        store_idempotent(
            session,
            actor=principal.subject,
            endpoint=endpoint,
            key=key,
            request_hash=request_hash,
            response_payload=response_payload,
            status_code=status_code,
            ttl_seconds=settings.idempotency_ttl_seconds,
        )
        return JSONResponse(status_code=status_code, content=response_payload)

    @app.get("/v1/runs/{run_id}")
    def read_run(
        run_id: str,
        session: Session = Depends(db_session),
        principal: Principal = Depends(require_roles("reader", "operator", "reviewer", "auditor")),
    ):
        run = get_run(session, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        result = dict(run.result_payload)
        if run.high_stakes and not run.decision_authorized and not principal.has_any({"reviewer", "auditor"}):
            result.pop("compliance", None)
            result["compliance"] = {"status": "withheld", "reason": "high-stakes readiness gate did not pass"}
        return {
            "run_id": run.id,
            "source_id": run.source_id,
            "high_stakes": run.high_stakes,
            "decision_authorized": run.decision_authorized,
            "created_at": run.created_at.isoformat(),
            "result": result,
        }

    @app.get("/v1/runs/{run_id}/audit-bundle")
    def export_audit_bundle(
        run_id: str,
        session: Session = Depends(db_session),
        principal: Principal = Depends(require_roles("auditor")),
    ):
        run = get_run(session, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        source = get_source(session, run.source_id)
        result = dict(run.result_payload)
        artifacts = {
            "source.json": {"id": source.id, "source_id": source.source_key, "text": source.content,
                            "content_hash": source.content_hash, "candidates": source.candidates,
                            "extraction_audit": source.extraction_audit},
            "run.json": {"id": run.id, "source_id": run.source_id, "requested_by": run.requested_by,
                         "request_hash": run.request_hash, "context": run.context_payload,
                         "high_stakes": run.high_stakes, "decision_authorized": run.decision_authorized,
                         "created_at": run.created_at.isoformat(), "result": result},
            "reviews.json": result.get("review_evidence", {"available": False, "reason": "historical_run"}),
            "runtime.json": result.get("runtime_evidence", {"available": False, "reason": "historical_run"}),
            "audit-events.json": [{"sequence": e.sequence, "actor": e.actor, "action": e.action,
                                   "resource_id": e.resource_id, "request_id": e.request_id,
                                   "payload_hash": e.payload_hash, "previous_hash": e.previous_hash,
                                   "event_hash": e.event_hash, "created_at": e.created_at.isoformat()}
                                  for e in audit_for_resource(session, run.id)],
        }
        try:
            archive, manifest_digest = create_bundle(artifacts, metadata={
                "kind": "lifecycle_run", "run_id": run.id, "legal_certification": False,
                "audit_chain_scope": "resource_subset_not_full_chain",
                "historical_evidence_complete": "review_evidence" in result and "runtime_evidence" in result,
            })
        except ValueError as exc:
            raise HTTPException(status_code=413, detail="audit bundle exceeds export constraints") from exc
        return Response(content=archive, media_type="application/zip", headers={
            "Content-Disposition": f'attachment; filename="audit-{run.id}.zip"',
            "X-Audit-Manifest-SHA256": manifest_digest,
            "ETag": f'"{digest(archive)}"', "Cache-Control": "no-store",
        })

    @app.get("/v1/audit-chain/verify")
    def verify_chain(
        session: Session = Depends(db_session),
        principal: Principal = Depends(require_roles("auditor")),
    ):
        return verify_audit_chain(session)

    @app.get("/v1/audit/{resource_id}")
    def read_audit(
        resource_id: str,
        session: Session = Depends(db_session),
        principal: Principal = Depends(require_roles("auditor")),
    ):
        events = audit_for_resource(session, resource_id)
        return {
            "resource_id": resource_id,
            "events": [
                {
                    "sequence": event.sequence,
                    "id": event.id,
                    "actor": event.actor,
                    "action": event.action,
                    "request_id": event.request_id,
                    "payload_hash": event.payload_hash,
                    "previous_hash": event.previous_hash,
                    "event_hash": event.event_hash,
                    "created_at": event.created_at.isoformat(),
                }
                for event in events
            ],
        }

    return app


app = create_app()
