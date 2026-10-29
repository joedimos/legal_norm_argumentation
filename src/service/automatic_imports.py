"""Durable read-only source acquisition. Never runs/reviews/authorizes legal decisions."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import uuid
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError

from src.audit_bundle import canonical_json, digest
from src.courtlistener import IP_QUERIES, CourtListenerClient
from src.legal_transport import ImportTransportError
from src.legifrance_ip import LegifranceIPClient

from .db_models import ImportCursorRecord
from .legifrance_ingestion import ingest_legal_items
from .repository import append_audit


def utcnow():
    return datetime.now(UTC)


def aware(value):
    return value.replace(tzinfo=UTC) if value is not None and value.tzinfo is None else value


def jobs(settings):
    result = {}
    if settings.legifrance_enabled:
        result["legifrance:ip"] = {"provider": "legifrance", "corpus": "ip"}
    if settings.courtlistener_enabled:
        for topic in IP_QUERIES:
            for mode in ("recent", "backfill"):
                result[f"courtlistener:{topic}:{mode}"] = {
                    "provider": "courtlistener",
                    "topic": topic,
                    "mode": mode,
                    "query": IP_QUERIES[topic],
                }
    return result


def configuration_hash(settings, job):
    # Credentials are intentionally excluded: rotation must resume, not restart,
    # a corpus. Changes to scope/schema reset the cursor and are audited.
    return digest(
        canonical_json(
            {
                "schema": "vle-ip-acquisition/v1",
                **job,
                "start": settings.import_start_date,
                "recent_days": settings.import_recent_days,
                "piste_environment": settings.legifrance_piste_environment,
            }
        )
    )


def claim(database, settings, *, now):
    active = jobs(settings)
    with database.session() as session:
        for key, job in active.items():
            if session.get(ImportCursorRecord, key) is None:
                try:
                    with session.begin_nested():
                        session.add(
                            ImportCursorRecord(
                                key=key,
                                config_hash=configuration_hash(settings, job),
                                position={},
                                due_at=now,
                                status="pending",
                                report={},
                            )
                        )
                        session.flush()
                except IntegrityError:
                    pass  # Another worker initialized the same fixed job.
        statement = (
            select(ImportCursorRecord)
            .where(
                ImportCursorRecord.key.in_(active),
                ImportCursorRecord.due_at <= now,
                or_(ImportCursorRecord.lease_expires_at.is_(None), ImportCursorRecord.lease_expires_at <= now),
            )
            .order_by(ImportCursorRecord.due_at, ImportCursorRecord.key)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        record = session.scalar(statement)
        if record is None:
            return None
        token = str(uuid.uuid4())
        changed = session.execute(
            update(ImportCursorRecord)
            .where(
                ImportCursorRecord.key == record.key,
                ImportCursorRecord.due_at <= now,
                or_(ImportCursorRecord.lease_expires_at.is_(None), ImportCursorRecord.lease_expires_at <= now),
            )
            .values(lease_token=token, lease_expires_at=now + timedelta(seconds=settings.import_lease_seconds))
            .execution_options(synchronize_session=False)
        )
        if changed.rowcount != 1:
            return None
        expected = configuration_hash(settings, active[record.key])
        if record.config_hash != expected:
            record.config_hash, record.position = expected, {}
            append_audit(
                session,
                actor="import-worker",
                action="import.scope_reset",
                resource_type="import",
                resource_id=record.key,
                request_id=token,
                payload={"config_hash": expected},
            )
        return {"key": record.key, "token": token, "position": deepcopy(record.position), **active[record.key]}


def build_client(settings, provider):
    if provider == "legifrance":
        if settings.legifrance_piste_environment != "production":
            raise ImportTransportError("automatic_import_requires_production_piste", retry_after=3600)
        return LegifranceIPClient(
            client_id=settings.legifrance_piste_client_id.get_secret_value()
            if settings.legifrance_piste_client_id
            else None,
            client_secret=settings.legifrance_piste_client_secret.get_secret_value()
            if settings.legifrance_piste_client_secret
            else None,
            environment=settings.legifrance_piste_environment,
            timeout=settings.legifrance_timeout_seconds,
            deadline_seconds=settings.import_max_run_seconds,
        )
    return CourtListenerClient(
        token=settings.courtlistener_api_token.get_secret_value() if settings.courtlistener_api_token else None,
        delay=settings.courtlistener_request_delay_seconds,
        timeout=settings.legifrance_timeout_seconds,
        deadline_seconds=settings.import_max_run_seconds,
    )


def collect(job, settings, client, *, now):
    p = deepcopy(job["position"])
    size = settings.import_batch_size
    if not p:
        until = now.date().isoformat()
        since = (
            (now.date() - timedelta(days=settings.import_recent_days)).isoformat()
            if job.get("mode") == "recent"
            else settings.import_start_date
        )
        p = {"until": until, "since": since, "cycle_items": 0}
    context = {"import_job": job["key"], "retrieved_at": now.isoformat(), "review_status": "unreviewed"}
    if job["provider"] == "legifrance":
        if "articles" not in p:
            p.update(client.manifest(p["until"]))
            p["offset"] = 0
        offset = p["offset"]
        items = [
            client.article(a, effective_date=p["until"], manifest_sha256=p["manifest_sha256"])
            for a in p["articles"][offset : offset + size]
        ]
        p["offset"] += len(items)
        complete = p["offset"] >= len(p["articles"])
        context.update(
            manifest_sha256=p["manifest_sha256"],
            manifest_articles=len(p["articles"]),
            excluded_noncurrent=p["excluded_noncurrent"],
        )
    else:
        if not p.get("pending") and not p.get("exhausted", False):
            if p.get("page_count", 0) >= 100000:
                raise ImportTransportError("search_page_bound", retry_after=86400)
            page = client.search_page(topic=job["topic"], since=p["since"], until=p["until"], cursor=p.get("cursor"))
            p.update(page)
            p["exhausted"] = page["cursor"] is None
            p["page_count"] = p.get("page_count", 0) + 1
        selected = p.get("pending", [])[:size]
        items = [result for item in selected if (result := client.opinion(item)) is not None]
        p["pending"] = p.get("pending", [])[len(selected) :]
        complete = p.get("exhausted", False) and not p["pending"]
        context["excluded_outside_scope"] = getattr(client, "exclusions", [])
        context.update(
            query=IP_QUERIES[job["topic"]],
            window={"since": p["since"], "until": p["until"]},
            search_sha256=p.get("search_sha256"),
            coverage="search_targeted_not_complete_IP_corpus",
        )
    p["cycle_items"] += len(items)
    return p, items, complete, context


def passive_result(status):
    return {
        "schema": "vle-import-result/v1",
        "status": status,
        "review_status": "unreviewed",
        "decision_authorized": False,
        "legal_certification": False,
    }


def process_one(database, settings, *, client_factory=None, clock=utcnow):
    if not settings.auto_import_enabled:
        return passive_result("disabled")
    job = claim(database, settings, now=clock())
    if job is None:
        return passive_result("idle")
    started = time.monotonic()
    try:
        client = (client_factory or build_client)(settings, job["provider"])
        p, items, complete, context = collect(job, settings, client, now=clock())
        with database.session() as session:
            record = session.scalar(
                select(ImportCursorRecord)
                .where(
                    ImportCursorRecord.key == job["key"],
                    ImportCursorRecord.lease_token == job["token"],
                    ImportCursorRecord.lease_expires_at > clock(),
                )
                .with_for_update()
            )
            if record is None:
                return passive_result("lease_lost")
            # SQLite's legacy transaction mode does not BEGIN on SELECT. Issue
            # a fenced DML write before nested source savepoints so releasing a
            # savepoint cannot commit sources outside the batch transaction.
            session.execute(
                update(ImportCursorRecord)
                .where(ImportCursorRecord.key == job["key"], ImportCursorRecord.lease_token == job["token"])
                .values(status="running")
            )
            summary = ingest_legal_items(
                session,
                items,
                actor="import-worker:" + job["provider"],
                request_id=job["token"],
                language="fr" if job["provider"] == "legifrance" else "en",
                provider_family=job["provider"],
                context=context,
            )
            now = clock()
            if time.monotonic() - started >= settings.import_max_run_seconds:
                raise ImportTransportError("import_time_bound")
            if aware(record.lease_expires_at) <= now:
                raise ImportTransportError("lease_expired_before_commit")
            delay = (
                (
                    settings.import_reconcile_seconds
                    if job.get("mode") == "backfill"
                    else settings.import_refresh_seconds
                )
                if complete
                else settings.import_poll_seconds
            )
            report = {
                "schema": "vle-import-result/v1",
                "job": job["key"],
                "status": "query_cycle_completed" if complete else "in_progress",
                "summary": summary,
                "cycle_items": p["cycle_items"],
                "scope": "IP_source_acquisition_not_review_or_legal_certification",
                "excluded_outside_scope": context.get("excluded_outside_scope", []),
                "review_status": "unreviewed",
                "decision_authorized": False,
                "legal_certification": False,
                "bounds": {"batch_size": settings.import_batch_size, "run_seconds": settings.import_max_run_seconds},
            }
            append_audit(
                session,
                actor="import-worker",
                action="import.batch",
                resource_type="import",
                resource_id=job["key"],
                request_id=job["token"],
                payload={
                    "status": report["status"],
                    "items": len(items),
                    "excluded_outside_scope": context.get("excluded_outside_scope", []),
                    "position_sha256": digest(canonical_json(p)),
                },
            )
            record.position = {} if complete else p
            record.report, record.status = report, report["status"]
            record.due_at = now + timedelta(seconds=delay)
            record.last_success_at, record.failure_count = now, 0
            record.lease_token, record.lease_expires_at = None, None
        return report
    except Exception as exc:
        # No source text, credential, upstream response body or exception repr leaks.
        # Failed batches do not advance the cursor or commit partial source versions.
        error = str(exc) if isinstance(exc, ImportTransportError) else "unexpected_import_failure"
        with database.session() as session:
            record = session.scalar(
                select(ImportCursorRecord)
                .where(ImportCursorRecord.key == job["key"], ImportCursorRecord.lease_token == job["token"])
                .with_for_update()
            )
            if record is None:
                return passive_result("lease_lost")
            record.failure_count += 1
            delay = max(min(3600, 30 * 2 ** min(record.failure_count, 7)), getattr(exc, "retry_after", 60))
            report = {
                "schema": "vle-import-result/v1",
                "job": job["key"],
                "status": "retry_scheduled",
                "reason": error,
                "retry_seconds": delay,
                "review_status": "unreviewed",
                "decision_authorized": False,
                "legal_certification": False,
            }
            record.status, record.report, record.due_at = report["status"], report, clock() + timedelta(seconds=delay)
            record.lease_token, record.lease_expires_at = None, None
            if error == "upstream_http_429":
                # Rate limits are token/provider-wide, not just query-specific.
                # Persist the cooldown across every job before another worker
                # can claim a different topic with the same credential.
                from sqlalchemy import case

                cooldown = clock() + timedelta(seconds=delay)
                session.execute(
                    update(ImportCursorRecord)
                    .where(ImportCursorRecord.key.startswith(job["provider"] + ":"))
                    .values(
                        due_at=case((ImportCursorRecord.due_at < cooldown, cooldown), else_=ImportCursorRecord.due_at)
                    )
                )
                report["provider_wide_cooldown"] = True
                record.report = dict(report)
            append_audit(
                session,
                actor="import-worker",
                action="import.retry_scheduled",
                resource_type="import",
                resource_id=job["key"],
                request_id=job["token"],
                payload={"reason": error, "retry_seconds": delay},
            )
        return report


def import_status(session, settings):
    active = jobs(settings)
    records = session.scalars(select(ImportCursorRecord).order_by(ImportCursorRecord.key)).all()
    missing = []
    if settings.legifrance_enabled and not settings.has_piste_credentials:
        missing.append("PISTE client id and secret")
    if settings.courtlistener_enabled and not settings.courtlistener_api_token:
        missing.append("CourtListener API token")
    return {
        "schema": "vle-import-status/v1",
        "review_status": "unreviewed",
        "enabled": settings.auto_import_enabled,
        "configured_jobs": sorted(active),
        "missing_credentials": missing,
        "jobs": [
            {
                "key": r.key,
                "enabled": r.key in active,
                "status": r.status,
                "due_at": aware(r.due_at).isoformat(),
                "leased": bool(r.lease_token),
                "lease_expires_at": aware(r.lease_expires_at).isoformat() if r.lease_expires_at else None,
                "last_success_at": aware(r.last_success_at).isoformat() if r.last_success_at else None,
                "failure_count": r.failure_count,
                "report": r.report,
            }
            for r in records
        ],
        "decision_authorized": False,
        "legal_certification": False,
    }


def main():
    from .config import get_settings
    from .database import Database

    parser = argparse.ArgumentParser(description="Automatic French/US IP-source import worker")
    parser.add_argument("--once", action="store_true", help="Process one bounded due batch")
    parser.add_argument("--status", action="store_true", help="Read acquisition state without contacting providers")
    args = parser.parse_args()
    settings = get_settings()
    db = Database(settings)
    try:
        if not db.ready(require_migrations=settings.environment == "production"):
            raise SystemExit("database schema is not ready; run alembic upgrade head")
        if args.status:
            with db.session() as session:
                print(json.dumps(import_status(session, settings), sort_keys=True))
            return
        with tempfile.TemporaryDirectory(prefix="vle-import-") as directory:
            while True:
                (Path(directory) / "heartbeat").touch()
                result = process_one(db, settings)
                print(json.dumps(result, sort_keys=True), flush=True)
                if args.once:
                    raise SystemExit(1 if result["status"] in {"retry_scheduled", "lease_lost"} else 0)
                time.sleep(settings.import_poll_seconds)
    finally:
        db.dispose()


if __name__ == "__main__":
    main()
