from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, desc, select, text
from sqlalchemy.orm import Session

from src.review import semantic_fingerprint

from .db_models import (
    AuditEventRecord,
    CandidateReviewRecord,
    IdempotencyRecord,
    InterpretationSelectionRecord,
    LifecycleRunRecord,
    RateLimitBucket,
    SourceDocumentRecord,
)


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def idempotency_hash(actor: str, endpoint: str, key: str) -> str:
    return hashlib.sha256(f"{actor}\0{endpoint}\0{key}".encode()).hexdigest()


def create_source(
    session: Session,
    *,
    source_key: str,
    content: str,
    candidates: list[dict[str, Any]],
    extraction_audit: dict[str, Any],
    created_by: str,
) -> SourceDocumentRecord:
    record = SourceDocumentRecord(
        id=str(uuid.uuid4()),
        source_key=source_key,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        candidates=candidates,
        extraction_audit=extraction_audit,
        created_by=created_by,
    )
    session.add(record)
    session.flush()
    return record


def find_source_by_key_hash(session: Session, source_key: str, content_hash: str) -> SourceDocumentRecord | None:
    return session.execute(
        select(SourceDocumentRecord).where(
            SourceDocumentRecord.source_key == source_key,
            SourceDocumentRecord.content_hash == content_hash,
        )
    ).scalar_one_or_none()


def get_source(session: Session, source_id: str) -> SourceDocumentRecord | None:
    return session.get(SourceDocumentRecord, source_id)


def get_candidate(source: SourceDocumentRecord, candidate_id: str) -> dict[str, Any] | None:
    return next((item for item in source.candidates if item.get("id") == candidate_id), None)


def add_review(
    session: Session,
    *,
    source: SourceDocumentRecord,
    candidate: Mapping[str, Any],
    decision: str,
    reviewer: str,
    note: str | None,
) -> CandidateReviewRecord:
    record = CandidateReviewRecord(
        id=str(uuid.uuid4()),
        source_id=source.id,
        candidate_id=str(candidate["id"]),
        candidate_fingerprint=semantic_fingerprint(dict(candidate)),
        decision=decision,
        reviewer=reviewer,
        note=note,
    )
    session.add(record)
    session.flush()
    return record


def latest_review(session: Session, source_id: str, candidate_id: str) -> CandidateReviewRecord | None:
    return session.execute(
        select(CandidateReviewRecord)
        .where(CandidateReviewRecord.source_id == source_id, CandidateReviewRecord.candidate_id == candidate_id)
        .order_by(desc(CandidateReviewRecord.created_at), desc(CandidateReviewRecord.id))
        .limit(1)
    ).scalar_one_or_none()


def review_statuses(session: Session, source: SourceDocumentRecord) -> list[str]:
    statuses: list[str] = []
    for candidate in source.candidates:
        candidate_id = str(candidate.get("id", ""))
        review = latest_review(session, source.id, candidate_id)
        if review is None:
            statuses.append("unreviewed")
        elif review.candidate_fingerprint != semantic_fingerprint(candidate):
            statuses.append("stale")
        else:
            statuses.append(review.decision)
    return statuses


def add_interpretation_selection(
    session: Session,
    *,
    source: SourceDocumentRecord,
    candidate: Mapping[str, Any],
    interpretation_id: str,
    rationale: str,
    reviewer: str,
) -> InterpretationSelectionRecord:
    record = InterpretationSelectionRecord(
        id=str(uuid.uuid4()),
        source_id=source.id,
        candidate_id=str(candidate["id"]),
        candidate_fingerprint=semantic_fingerprint(dict(candidate)),
        interpretation_id=interpretation_id,
        rationale=rationale,
        reviewer=reviewer,
    )
    session.add(record)
    session.flush()
    return record


def latest_selection(session: Session, source_id: str, candidate_id: str) -> InterpretationSelectionRecord | None:
    return session.execute(
        select(InterpretationSelectionRecord)
        .where(
            InterpretationSelectionRecord.source_id == source_id,
            InterpretationSelectionRecord.candidate_id == candidate_id,
        )
        .order_by(desc(InterpretationSelectionRecord.created_at), desc(InterpretationSelectionRecord.id))
        .limit(1)
    ).scalar_one_or_none()


def interpretation_selections(session: Session, source: SourceDocumentRecord) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for candidate in source.candidates:
        candidate_id = str(candidate.get("id", ""))
        selection = latest_selection(session, source.id, candidate_id)
        if selection and selection.candidate_fingerprint == semantic_fingerprint(candidate):
            result[candidate_id] = {
                "selected_id": selection.interpretation_id,
                "rationale": selection.rationale,
            }
    return result


def create_run(
    session: Session,
    *,
    source_id: str,
    requested_by: str,
    high_stakes: bool,
    decision_authorized: bool,
    request_payload: Mapping[str, Any],
    context_payload: Mapping[str, Any],
    result_payload: Mapping[str, Any],
) -> LifecycleRunRecord:
    record = LifecycleRunRecord(
        id=str(uuid.uuid4()),
        source_id=source_id,
        requested_by=requested_by,
        high_stakes=high_stakes,
        decision_authorized=decision_authorized,
        request_hash=canonical_hash(request_payload),
        context_payload=dict(context_payload),
        result_payload=dict(result_payload),
    )
    session.add(record)
    session.flush()
    return record


def get_run(session: Session, run_id: str) -> LifecycleRunRecord | None:
    return session.get(LifecycleRunRecord, run_id)


def append_audit(
    session: Session,
    *,
    actor: str,
    action: str,
    resource_type: str,
    resource_id: str,
    request_id: str,
    payload: Any,
) -> AuditEventRecord:
    if session.get_bind().dialect.name == "postgresql":
        session.execute(text("SELECT pg_advisory_xact_lock(726593201)"))
    previous = session.execute(
        select(AuditEventRecord).order_by(desc(AuditEventRecord.sequence)).limit(1)
    ).scalar_one_or_none()
    created_at = datetime.now(UTC)
    payload_hash = canonical_hash(payload)
    previous_hash = previous.event_hash if previous else None
    event_material = {
        "actor": actor,
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "request_id": request_id,
        "payload_hash": payload_hash,
        "previous_hash": previous_hash,
        "created_at": created_at.isoformat(),
    }
    record = AuditEventRecord(
        id=str(uuid.uuid4()),
        actor=actor,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        request_id=request_id,
        payload_hash=payload_hash,
        previous_hash=previous_hash,
        event_hash=canonical_hash(event_material),
        created_at=created_at,
    )
    session.add(record)
    session.flush()
    return record


def audit_for_resource(session: Session, resource_id: str) -> list[AuditEventRecord]:
    return list(
        session.execute(
            select(AuditEventRecord)
            .where(AuditEventRecord.resource_id == resource_id)
            .order_by(AuditEventRecord.sequence)
        ).scalars()
    )


def get_idempotent(
    session: Session,
    *,
    actor: str,
    endpoint: str,
    key: str,
    request_hash: str,
) -> IdempotencyRecord | None:
    key_hash = idempotency_hash(actor, endpoint, key)
    record = session.execute(
        select(IdempotencyRecord).where(
            IdempotencyRecord.key_hash == key_hash,
            IdempotencyRecord.expires_at > datetime.now(UTC),
        )
    ).scalar_one_or_none()
    if record is not None and record.request_hash != request_hash:
        raise ValueError("idempotency key was already used with a different request")
    return record


def store_idempotent(
    session: Session,
    *,
    actor: str,
    endpoint: str,
    key: str,
    request_hash: str,
    response_payload: Mapping[str, Any],
    status_code: int,
    ttl_seconds: int,
) -> IdempotencyRecord:
    key_digest = idempotency_hash(actor, endpoint, key)
    record = session.get(IdempotencyRecord, key_digest)
    expires_at = datetime.now(UTC) + timedelta(seconds=ttl_seconds)
    if record is None:
        record = IdempotencyRecord(
            key_hash=key_digest,
            actor=actor,
            endpoint=endpoint,
            request_hash=request_hash,
            response_payload=dict(response_payload),
            status_code=status_code,
            expires_at=expires_at,
        )
        session.add(record)
    else:
        record.actor = actor
        record.endpoint = endpoint
        record.request_hash = request_hash
        record.response_payload = dict(response_payload)
        record.status_code = status_code
        record.expires_at = expires_at
        record.created_at = datetime.now(UTC)
    session.flush()
    return record


def verify_audit_chain(session: Session) -> dict[str, Any]:
    events = list(session.execute(select(AuditEventRecord).order_by(AuditEventRecord.sequence)).scalars())
    previous_hash: str | None = None
    for event in events:
        created_at = event.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        else:
            created_at = created_at.astimezone(UTC)
        material = {
            "actor": event.actor,
            "action": event.action,
            "resource_type": event.resource_type,
            "resource_id": event.resource_id,
            "request_id": event.request_id,
            "payload_hash": event.payload_hash,
            "previous_hash": previous_hash,
            "created_at": created_at.isoformat(),
        }
        expected_hash = canonical_hash(material)
        if event.previous_hash != previous_hash or event.event_hash != expected_hash:
            return {
                "valid": False,
                "event_count": len(events),
                "broken_sequence": event.sequence,
                "head_hash": events[-1].event_hash if events else None,
            }
        previous_hash = event.event_hash
    return {
        "valid": True,
        "event_count": len(events),
        "broken_sequence": None,
        "head_hash": previous_hash,
    }


def cleanup_ephemeral_state(session: Session, *, rate_limit_retention_minutes: int = 120) -> dict[str, int]:
    now = datetime.now(UTC)
    idem = session.execute(delete(IdempotencyRecord).where(IdempotencyRecord.expires_at <= now))
    cutoff = now - timedelta(minutes=rate_limit_retention_minutes)
    rate = session.execute(delete(RateLimitBucket).where(RateLimitBucket.window_start < cutoff))
    return {
        "idempotency_records_deleted": int(idem.rowcount or 0),
        "rate_limit_buckets_deleted": int(rate.rowcount or 0),
    }


def review_evidence_snapshot(session: Session, source: SourceDocumentRecord) -> dict:
    """Freeze current review identity and fingerprint with the decision, not export time."""
    reviews = []
    selections = []
    for candidate in source.candidates:
        cid = str(candidate.get("id", ""))
        review = latest_review(session, source.id, cid)
        if review:
            reviews.append({"id": review.id, "candidate_id": cid,
                            "candidate_fingerprint": review.candidate_fingerprint,
                            "decision": review.decision, "reviewer": review.reviewer,
                            "note": review.note, "created_at": review.created_at.isoformat()})
        selection = latest_selection(session, source.id, cid)
        if selection:
            selections.append({"id": selection.id, "candidate_id": cid,
                               "candidate_fingerprint": selection.candidate_fingerprint,
                               "interpretation_id": selection.interpretation_id,
                               "rationale": selection.rationale, "reviewer": selection.reviewer,
                               "created_at": selection.created_at.isoformat()})
    return {"reviews": reviews, "interpretation_selections": selections,
            "statuses": review_statuses(session, source)}
