from __future__ import annotations

import hashlib
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from src.audit_bundle import canonical_json, create_bundle, digest, runtime_evidence
from src.formalization import formalize_candidates
from src.lifecycle import run_lifecycle
from src.norm_extraction import extract_norms
from src.normative_model import FactContext
from src.review import semantic_fingerprint
from src.verification import audit_extraction

from .config import Settings, get_settings
from .database import Database
from .db_models import WorkflowRecord
from .ip_release import source_scope_gate
from .logging import configure_logging
from .metrics import LIFECYCLE_RUNS
from .middleware import MaxBodySizeMiddleware, request_context_middleware
from .pyreason_release import pyreason_release_gate
from .rate_limit import DatabaseRateLimiter
from .release_policy import case_review_gate, deployment_release_gate
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
from .workflow_routes import router as workflow_router

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

    @asynccontextmanager
    async def lifespan(app):
        if settings.pyreason_warmup:
            from src.pyreason_warmup import warm_native
            app.state.pyreason_startup = await run_in_threadpool(
                warm_native, timeout_seconds=settings.pyreason_warmup_timeout_seconds,
                worker_mode=settings.pyreason_worker_mode)
        try:
            yield
        finally:
            if settings.pyreason_warmup:
                from src.pyreason_pool import close_worker
                close_worker()

    app = FastAPI(
        title="Legal Norm Argumentation Service",
        version=settings.service_version,
        docs_url="/docs" if settings.allow_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.allow_docs else None,
        lifespan=lifespan,
    )
    app.include_router(workflow_router)
    from .workflow_ui import register_ui
    register_ui(app)
    database = Database(settings)
    if settings.auto_create_schema:
        database.create_schema_for_tests()
    app.state.settings = settings
    app.state.pyreason_startup = {"status": "pending" if settings.pyreason_warmup else "not_required"}
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
        healthy = healthy and (not settings.pyreason_warmup or app.state.pyreason_startup["status"] == "ready")
        if not healthy:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return {"status": "ready"}

    @app.get("/v1/release-status")
    def release_status(principal: Principal = Depends(require_roles("reader", "operator", "reviewer", "auditor"))):
        return {"release": deployment_release_gate(settings), "inference_enabled": settings.inference_enabled,
                "build_revision": settings.build_revision, "release_enforcement": settings.release_enforcement,
                "case_review_required": settings.release_enforcement == "strict", "legal_certification": False,
                "pyreason": pyreason_release_gate(settings)}

    @app.get("/metrics", include_in_schema=False, dependencies=[Depends(verify_metrics_token)])
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/release-readiness")
    def release_readiness_status(principal: Principal = Depends(require_roles("reader", "operator", "reviewer", "auditor"))):
        from .release_readiness import release_readiness
        return release_readiness(settings, database)

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

    def evaluation_payload(payload, *, exclude=None):
        data = payload.model_dump(mode="json", exclude=exclude)
        if payload.pyreason_program is None:
            data.pop("pyreason_program", None)
        if payload.reasoning_backend == "core":
            # Preserve legacy idempotency and signed evaluation-target bytes.
            data.pop("reasoning_backend", None)
            data.pop("reasoning_timeout_seconds", None)
        return data

    def reasoning_options(payload):
        if payload.pyreason_program is not None and payload.reasoning_backend != "pyreason":
            raise HTTPException(422, "graph inference requires the PyReason backend")
        if payload.reasoning_backend == "pyreason" and not settings.pyreason_enabled:
            raise HTTPException(503, "PyReason normative backend is not enabled on this deployment")
        if payload.reasoning_backend == "pyreason" and settings.pyreason_warmup and app.state.pyreason_startup["status"] != "ready":
            raise HTTPException(503, "Native backend startup has not completed")
        return {"reasoning_backend": payload.reasoning_backend,
                "pyreason_timeout_seconds": payload.reasoning_timeout_seconds,
                "pyreason_program": payload.pyreason_program, "pyreason_worker_mode": settings.pyreason_worker_mode}

    def target_snapshot(session, source, payload, result):
        snapshot = {"schema": "vle-evaluation-target/v1", "build_revision": settings.build_revision,
                    "source_id": source.id, "source_sha256": source.content_hash, "candidates": source.candidates,
                    "formalizations": result["formalizations"], "reviews": review_evidence_snapshot(session, source),
                    "context": evaluation_payload(payload, exclude={"legal_assessment_id", "high_stakes"})}
        if settings.release_profile != "legacy":
            snapshot["release_profile"] = settings.release_profile
            snapshot["model_revision"] = settings.model_revision
            snapshot["source_provenance"] = source.extraction_audit.get("provenance", {})
        if payload.reasoning_backend == "pyreason" and settings.pyreason_worker_mode != "isolated":
            snapshot["context"]["backend_runtime"] = {"worker_mode": settings.pyreason_worker_mode}
        return snapshot, digest(canonical_json(snapshot))

    @app.post("/v1/sources/{source_id}/release-target")
    def prepare_release_target(source_id: str, payload: EvaluateRequest, session: Session = Depends(db_session),
                               principal: Principal = Depends(require_roles("operator", "reviewer"))):
        source = get_source(session, source_id)
        if source is None:
            raise HTTPException(404, "source not found")
        context = FactContext(facts=payload.facts, actor=payload.actor, observed_actions=tuple(payload.observed_actions),
                              at_date=payload.at_date, metadata=payload.metadata)
        result = run_lifecycle(text=source.content, source_id=source.source_key, candidates=source.candidates,
                               extraction_audit=source.extraction_audit, context=context, review_statuses=review_statuses(session, source),
                               interpretation_selections=interpretation_selections(session, source), **reasoning_options(payload))
        snapshot, fingerprint = target_snapshot(session, source, payload, result)
        return {"target_sha256": fingerprint, "target": snapshot, "research_readiness": result["readiness"],
                "deployment_authorized": False, "legal_certification": False}

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
        options = reasoning_options(payload)
        request_payload = evaluation_payload(payload)
        request_hash = canonical_hash(request_payload)
        try:
            cached = get_idempotent(session, actor=principal.subject, endpoint=endpoint, key=key, request_hash=request_hash)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if cached and not payload.high_stakes:
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
            **options,
        )
        result["review_evidence"] = review_evidence_snapshot(session, source)
        result["runtime_evidence"] = runtime_evidence(revision=settings.build_revision)
        ready = result.get("readiness", {}).get("status") == "research_ready"
        snapshot, target_hash = target_snapshot(session, source, payload, result)
        release = deployment_release_gate(settings) if payload.high_stakes else {"passed": False, "reason": "research_only_run"}
        source_scope = source_scope_gate(settings, source, snapshot["context"])
        assessment = session.get(WorkflowRecord, payload.legal_assessment_id) if payload.legal_assessment_id else None
        case = case_review_gate(settings, assessment, target_hash, source.content_hash,
                                require_grounding=bool(payload.facts or payload.observed_actions or payload.actor or payload.pyreason_program), context=snapshot["context"]) if payload.high_stakes else {"passed": False}
        if assessment is not None and assessment.request_payload.get("source_id") != source.id:
            case = {"passed": False, "reason": "case_review_source_mismatch"}
        # Native computation alone is insufficient. The additional gate requires
        # exact runtime/build, native load evidence and independently signed IP data.
        backend_release = ({"name": "core", "passed": True, "release_qualified": True}
                           if payload.reasoning_backend == "core" else pyreason_release_gate(settings))
        backend_release_qualified = backend_release["passed"]
        release_ready = bool(backend_release_qualified and ready and release["passed"] and case["passed"] and source_scope["passed"])
        external_validation_satisfied = bool(backend_release_qualified and release["passed"] and case["passed"])
        reviewed_operation = bool(payload.reasoning_backend == "core" and payload.high_stakes and ready and settings.release_enforcement == "reviewed" and settings.release_profile == "legacy")
        authorized = release_ready or reviewed_operation
        override_used = bool(authorized and not external_validation_satisfied)
        result["production_release"] = {"authorized": authorized, "evaluation_target_sha256": target_hash,
                                       "enforcement": settings.release_enforcement, "external_validation_satisfied": external_validation_satisfied,
                                       "override_used": override_used,
                                       "authorization_basis": "reviewed_source_operation" if override_used else "validated_release" if release_ready else "none",
                                       "deployment": release, "independent_case_review": case, "legal_certification": False}
        if settings.release_profile != "legacy":
            result["production_release"].update(profile=settings.release_profile, source_scope=source_scope)
        if payload.reasoning_backend == "pyreason":
            result["production_release"]["reasoning_backend"] = backend_release
            if payload.high_stakes and not backend_release_qualified:
                result["readiness"]["gates"].append({"gate": "reasoning_backend_release_qualification", "passed": False,
                                                    "explanation": "Current native runtime, load and independently signed IP qualification evidence is required."})
                result["readiness"]["failed_gates"].append("reasoning_backend_release_qualification")
                result["readiness"]["status"] = "not_ready"
        result["authorization_scope"] = ("organizational_release" if release_ready else "reviewed_operation" if reviewed_operation
                                         else "research_only" if not payload.high_stakes else "blocked")
        if payload.high_stakes and not release_ready and settings.release_enforcement == "strict":
            # The research verdict stays visible as research evidence. It never
            # silently becomes the stronger production authorization predicate.
            checks = [("production_release_evidence", release), ("independent_case_review", case)]
            if settings.release_profile != "legacy":
                checks.append(("supported_source_scope", source_scope))
            for gate, evidence in checks:
                result["readiness"]["gates"].append({"gate": gate, "passed": evidence["passed"], "explanation": evidence.get("reason", "Current release evidence verified")})
                if not evidence["passed"]:
                    result["readiness"]["failed_gates"].append(gate)
            result["readiness"]["status"] = "not_ready"
        if cached:
            old_release = cached.response_payload.get("production_release", {})
            same_basis = (old_release.get("enforcement") == settings.release_enforcement
                          and old_release.get("external_validation_satisfied") == external_validation_satisfied
                          and old_release.get("evaluation_target_sha256") == target_hash)
            if authorized and cached.response_payload.get("decision_authorized") and same_basis:
                return JSONResponse(status_code=cached.status_code, content=cached.response_payload)
            return JSONResponse(status_code=409, content={"run_id": cached.response_payload.get("run_id"),
                "decision_authorized": False, "production_release": result["production_release"],
                "detail": "historical response has different authorization evidence/policy; submit a new Idempotency-Key"})
        if "decision_authorized" in result:
            result["decision_authorized"] = authorized  # Only the existing service release policy can authorize.
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
            payload={"source_id": source.id, "decision_authorized": authorized, "readiness": result.get("readiness"),
                     "release_enforcement": settings.release_enforcement, "override_used": override_used,
                     "external_validation_satisfied": external_validation_satisfied},
        )
        LIFECYCLE_RUNS.labels(str(authorized).lower(), str(payload.high_stakes).lower()).inc()
        status_code = 200 if authorized or not payload.high_stakes else 409
        if authorized or not payload.high_stakes:
            response_payload = {**result, "run_id": run.id, "decision_authorized": authorized}
        else:
            response_payload = {
                "run_id": run.id,
                "decision_authorized": False,
                "readiness": result.get("readiness"),
                "production_release": result["production_release"],
                "detail": "high-stakes decision blocked by research, release or independent legal-review gates",
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
            "authorization_is_historical": True,
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
                         "high_stakes": run.high_stakes,
            "authorization_is_historical": True, "decision_authorized": run.decision_authorized,
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
