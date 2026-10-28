"""Authenticated persistent research operations; provider URLs are server-owned."""
from __future__ import annotations

from copy import deepcopy
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from src.audit_bundle import canonical_json, create_bundle, digest
from src.family_contract import parse_family

from .db_models import WorkflowRecord
from .release_policy import pinned_json, verify_assessment
from .repository import get_idempotent, get_source, store_idempotent
from .security import Principal, require_roles
from .workflows import (
    add_event,
    create_workflow,
    events,
    generation_request,
    queue_status,
    review_workflow,
    runtime_binding,
    state,
    workflow_summaries,
)

router = APIRouter()


class StrictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FamilySubmission(StrictBody):
    family: dict


class GenerationSubmission(StrictBody):
    family_workflow_id: str = Field(min_length=1, max_length=36)
    prompt: str = Field(min_length=1, max_length=32000)
    history_ids: list[str] = Field(default_factory=list, max_length=4)
    max_cycles: int = Field(default=2, ge=1, le=5, strict=True)
    max_tokens: int = Field(default=2048, ge=1, le=8192, strict=True)
    retry_of: str | None = Field(default=None, max_length=36)
    retry_reason: str | None = Field(default=None, min_length=5, max_length=1000)


class WorkflowReview(StrictBody):
    scope: Literal["family", "grounding", "behavior"]
    decision: Literal["approved", "rejected", "needs_revision"]
    notes: str = Field(min_length=1, max_length=4000)


class LegalAssessment(StrictBody):
    target_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assessment: dict


class Interruption(StrictBody):
    reason: str = Field(min_length=5, max_length=1000)


def db_session(request: Request):
    with request.app.state.db.session() as session:
        yield session


def visible(record, principal):
    if record is None:
        raise HTTPException(404, "workflow not found")
    if record.created_by != principal.subject and not principal.has_any({"reviewer", "auditor"}):
        raise HTTPException(403, "workflow belongs to a different principal")
    return record


def mutate(session, request, principal, key, body, action):
    if key is None or not 8 <= len(key) <= 128:
        raise HTTPException(400, "Idempotency-Key must contain 8-128 characters")
    hashed = digest(canonical_json(body))
    endpoint = "POST:" + request.url.path
    try:
        old = get_idempotent(session, actor=principal.subject, endpoint=endpoint, key=key, request_hash=hashed)
        if old:
            return JSONResponse(old.response_payload, status_code=old.status_code)
        code, result = action()
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    store_idempotent(session, actor=principal.subject, endpoint=endpoint, key=key, request_hash=hashed,
                     response_payload=result, status_code=code, ttl_seconds=request.app.state.settings.idempotency_ttl_seconds)
    return JSONResponse(result, status_code=code)


@router.post("/v1/workflows/families")
def submit_family(payload: FamilySubmission, request: Request, session=Depends(db_session),
                  principal: Principal = Depends(require_roles("operator")), idempotency_key: str | None = Header(default=None)):
    def action():
        family = deepcopy(payload.family)
        parse_family(family)
        if family.get("reviews") or "behavior" in family or any(d.get("kind") == "draft" for d in family.get("documents", [])):
            raise ValueError("base families cannot include caller-created approvals or drafts")
        family.pop("reviews", None)
        parse_family(family)
        record = create_workflow(session, "family", principal.subject, family, request.state.request_id)
        return 201, state(session, record)
    return mutate(session, request, principal, idempotency_key, payload.model_dump(), action)


@router.post("/v1/workflows/generations")
def submit_generation(payload: GenerationSubmission, request: Request, session=Depends(db_session),
                      principal: Principal = Depends(require_roles("operator")), idempotency_key: str | None = Header(default=None)):
    if not request.app.state.settings.inference_enabled:
        raise HTTPException(503, "inference worker is disabled")
    def action():
        family = visible(session.get(WorkflowRecord, payload.family_workflow_id), principal)
        if family.created_by != principal.subject:
            raise HTTPException(403, "only the family operator may submit its inference jobs")
        for id in payload.history_ids:
            visible(session.get(WorkflowRecord, id), principal)
        body = {**payload.model_dump(), "runtime_binding": runtime_binding(request.app.state.settings)}
        if bool(payload.retry_of) != bool(payload.retry_reason):
            raise ValueError("an explicit retry requires its prior job ID and investigation reason")
        if payload.retry_of:
            prior = visible(session.get(WorkflowRecord, payload.retry_of), principal)
            if prior.kind != "generation" or state(session, prior)["status"] not in {"generation_failed", "worker_failed", "interrupted"}:
                raise ValueError("only a terminal failed/interrupted job can be explicitly retried")
            for field in ("family_workflow_id", "prompt", "history_ids", "max_cycles", "max_tokens"):
                if prior.request_payload[field] != body[field]:
                    raise ValueError("retry must preserve the prior request; submit changed work as a new job")
        generation_request(session, body, request.app.state.settings)
        record = create_workflow(session, "generation", principal.subject, body, request.state.request_id)
        return 202, state(session, record)
    return mutate(session, request, principal, idempotency_key, payload.model_dump(), action)


@router.post("/v1/workflows/{workflow_id}/reviews")
def submit_review(workflow_id: str, payload: WorkflowReview, request: Request, session=Depends(db_session),
                  principal: Principal = Depends(require_roles("reviewer")), idempotency_key: str | None = Header(default=None)):
    def action():
        record = visible(session.get(WorkflowRecord, workflow_id), principal)
        if record.created_by == principal.subject:
            raise HTTPException(409, "separation of duties forbids self-review")
        if record.kind not in {"family", "generation"} or (record.kind == "generation" and payload.scope == "family"):
            raise ValueError("review scope is invalid for this workflow")
        result = review_workflow(session, record, actor=principal.subject, scope=payload.scope,
                                 notes=payload.notes, decision=payload.decision, request_id=request.state.request_id)
        return 201, result
    return mutate(session, request, principal, idempotency_key, payload.model_dump(), action)


@router.post("/v1/sources/{source_id}/legal-assessments")
def submit_legal_assessment(source_id: str, payload: LegalAssessment, request: Request, session=Depends(db_session),
                            principal: Principal = Depends(require_roles("reviewer")), idempotency_key: str | None = Header(default=None)):
    def action():
        source = get_source(session, source_id)
        if source is None:
            raise HTTPException(404, "source not found")
        if source.created_by == principal.subject:
            raise HTTPException(409, "source operator cannot register its independent legal assessment")
        try:
            result = verify_assessment(request.app.state.settings, payload.assessment)
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
            raise HTTPException(422, "signed assessment failed current trusted-registry verification") from exc
        signed_payloads = [s["payload"] for s in payload.assessment["submissions"]]
        if "adjudication" in payload.assessment:
            signed_payloads.append(payload.assessment["adjudication"]["payload"])
        if any(p.get("target_sha256") != payload.target_sha256 for p in signed_payloads):
            raise ValueError("both independent signatures and any adjudication must bind target_sha256")
        normative = [d for u in payload.assessment["packet"]["units"] for d in u["family"]["documents"] if d["kind"] == "normative"]
        if not normative or any(d["sha256"] != source.content_hash for d in normative):
            raise ValueError("case packet must bind this exact source snapshot")
        record = create_workflow(session, "legal_review", principal.subject,
                                  {"source_id": source_id, **payload.model_dump()}, request.state.request_id)
        if not events(session, record.id):
            registry = pinned_json(request.app.state.settings.reviewer_registry_path, request.app.state.settings.reviewer_registry_sha256)
            add_event(session, record, "finish", principal.subject, {"result": result, "registry_snapshot": registry}, request.state.request_id)
        return 201, {"id": record.id, "status": result["status"], "key_signatures_verified": True}
    return mutate(session, request, principal, idempotency_key, payload.model_dump(), action)


@router.get("/v1/workflows")
def list_workflows(request: Request, session=Depends(db_session), principal: Principal = Depends(require_roles("operator", "reviewer", "auditor")),
                   limit: int = 50):
    if not 1 <= limit <= 100:
        raise HTTPException(422, "limit must be 1-100")
    subject = None if principal.has_any({"reviewer", "auditor"}) else principal.subject
    return {"workflows": workflow_summaries(session, subject=subject, limit=limit)}


@router.get("/v1/workflow-status")
def workflow_status(request: Request, session=Depends(db_session),
                    principal: Principal = Depends(require_roles("operator", "reviewer", "auditor"))):
    subject = None if principal.has_any({"reviewer", "auditor"}) else principal.subject
    return queue_status(session, subject=subject,
                        stale_after_seconds=request.app.state.settings.workflow_stale_after_seconds)


@router.get("/v1/workflows/{workflow_id}")
def get_workflow(workflow_id: str, session=Depends(db_session), principal: Principal = Depends(require_roles("operator", "reviewer", "auditor"))):
    return state(session, visible(session.get(WorkflowRecord, workflow_id), principal))


@router.get("/v1/workflows/{workflow_id}/audit-bundle")
def export_workflow(workflow_id: str, request: Request, session=Depends(db_session), principal: Principal = Depends(require_roles("auditor"))):
    record = visible(session.get(WorkflowRecord, workflow_id), principal)
    artifact = {"request.json": record.request_payload, "state.json": state(session, record),
                "events.json": [{"slot": e.slot, "actor": e.actor, "payload": e.payload} for e in events(session, record.id)]}
    try:
        archive, root = create_bundle(artifact, metadata={"workflow_id": record.id, "legal_certification": False})
    except ValueError as exc:
        raise HTTPException(413, "workflow evidence exceeds export bound") from exc
    return Response(archive, media_type="application/zip", headers={"X-Audit-Manifest-SHA256": root, "Cache-Control": "no-store"})


@router.post("/v1/workflows/{workflow_id}/interrupt")
def interrupt_workflow(workflow_id: str, payload: Interruption, request: Request, session=Depends(db_session),
                       principal: Principal = Depends(require_roles("admin")), idempotency_key: str | None = Header(default=None)):
    def action():
        record = visible(session.scalar(select(WorkflowRecord).where(WorkflowRecord.id == workflow_id).with_for_update()), principal)
        if record.kind != "generation" or any(e.slot == "finish" for e in events(session, record.id)):
            raise ValueError("only unfinished inference jobs can be interrupted")
        add_event(session, record, "finish", principal.subject, {"result": {"status": "interrupted", "reason": payload.reason,
                  "deployment_authorized": False, "legal_certification": False}}, request.state.request_id)
        return 200, state(session, record)
    return mutate(session, request, principal, idempotency_key, payload.model_dump(), action)
