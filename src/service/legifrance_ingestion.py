from __future__ import annotations

import hashlib

from sqlalchemy import desc, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.legifrance import ScrapeBatch
from src.norm_extraction import extract_norms
from src.verification import audit_extraction

from .db_models import SourceDocumentRecord
from .repository import append_audit, create_source, find_source_by_key_hash, review_statuses


def source_summary(session: Session, source: SourceDocumentRecord) -> dict:
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


def ingest_legal_items(
    session: Session,
    provisions,
    *,
    actor: str,
    request_id: str,
    language: str,
    provider_family: str,
    context: dict | None = None,
) -> dict:
    created = 0
    updated = 0
    unchanged = 0
    items: list[dict] = []

    for provision in provisions:
        provenance = {**provision.provenance, **(context or {})}
        source_key = provision.source_key
        source_text = provision.source_text
        content_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        existing = find_source_by_key_hash(session, source_key, content_hash)
        if existing is not None:
            unchanged += 1
            append_audit(
                session,
                actor=actor,
                action="source.refresh_checked",
                resource_type="source",
                resource_id=existing.id,
                request_id=request_id,
                payload={"source_id": source_key, "content_hash": content_hash, **provenance},
            )
            items.append({"status": "unchanged", **source_summary(session, existing)})
            continue

        previous = _latest_source_for_key(session, source_key)
        candidates = extract_norms(source_text, source_id=source_key)
        extraction_audit = dict(audit_extraction(source_text, candidates))
        extraction_audit["provenance"] = provenance
        extraction_audit["language"] = language
        extraction_audit["review_status"] = "unreviewed"
        try:
            with session.begin_nested():
                source = create_source(
                    session,
                    source_key=source_key,
                    content=source_text,
                    candidates=candidates,
                    extraction_audit=extraction_audit,
                    created_by=actor,
                )
        except IntegrityError:
            source = find_source_by_key_hash(session, source_key, content_hash)
            if source is None:
                raise
            unchanged += 1
            items.append({"status": "unchanged", **source_summary(session, source)})
            continue

        status = "updated" if previous is not None else "created"
        if status == "updated":
            updated += 1
        else:
            created += 1
        append_audit(
            session,
            actor=actor,
            action="source.ingested." + provider_family,
            resource_type="source",
            resource_id=source.id,
            request_id=request_id,
            payload={
                "source_id": source_key,
                "content_hash": source.content_hash,
                "candidate_ids": [candidate.get("id") for candidate in candidates],
                "previous_source_id": previous.id if previous else None,
                **provenance,
            },
        )
        items.append({"status": status, **source_summary(session, source)})

    return {
        "created": created,
        "updated": updated,
        "unchanged": unchanged,
        "items": items,
    }


def ingest_legifrance_batch(session: Session, batch: ScrapeBatch, *, actor: str, request_id: str) -> dict:
    """Backward-compatible manual route, sharing the governed source-version importer."""
    result = ingest_legal_items(
        session, batch.provisions, actor=actor, request_id=request_id, language="fr", provider_family="legifrance"
    )
    return {
        "corpus": batch.corpus,
        "seed_url": batch.seed_url,
        "discovered": batch.discovered,
        "selected": batch.selected,
        "errors": list(batch.errors),
        **result,
    }
