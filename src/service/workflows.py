"""Append-only workflow state and at-most-once model execution claims."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
import uuid
from copy import deepcopy
from pathlib import Path

from sqlalchemy import exists, select
from sqlalchemy.exc import IntegrityError

from src.audit_bundle import canonical_json, digest
from src.family_contract import record_family_review
from src.interpretation_family import analyze_family
from src.live_generation import generate_candidate
from src.model_runtime import ollama_chat, vllm_chat

from .db_models import WorkflowEventRecord, WorkflowRecord
from .repository import append_audit


def events(session, id):
    return list(session.scalars(select(WorkflowEventRecord).where(WorkflowEventRecord.workflow_id == id)
                               .order_by(WorkflowEventRecord.sequence)))


def add_event(session, workflow, slot, actor, payload, request_id):
    session.add(WorkflowEventRecord(workflow_id=workflow.id, slot=slot, actor=actor, payload=payload))
    session.flush()
    append_audit(session, actor=actor, action="workflow." + slot.split(":")[0], resource_type="workflow",
                 resource_id=workflow.id, request_id=request_id, payload={"slot": slot, "sha256": digest(canonical_json(payload))})


def create_workflow(session, kind, actor, payload, request_id):
    hashed = digest(canonical_json(payload))
    query = select(WorkflowRecord).where(WorkflowRecord.kind == kind, WorkflowRecord.created_by == actor,
                                         WorkflowRecord.request_hash == hashed)
    old = session.scalar(query)
    if old:
        return old
    try:
        with session.begin_nested():
            record = WorkflowRecord(id=str(uuid.uuid4()), kind=kind, created_by=actor,
                                     request_hash=hashed, request_payload=payload)
            session.add(record)
            session.flush()
    except IntegrityError:
        return session.scalar(query)
    append_audit(session, actor=actor, action="workflow.submitted", resource_type="workflow", resource_id=record.id,
                 request_id=request_id, payload={"kind": kind, "request_sha256": hashed})
    return record


def state(session, record):
    log = events(session, record.id)
    finished = next((event.payload for event in log if event.slot == "finish"), None)
    status = "stored" if record.kind != "generation" else "queued"
    if any(event.slot == "claim" for event in log):
        status = "running_or_interrupted"
    if finished:
        status = finished["result"]["status"]
    review_request = deepcopy(record.request_payload) if record.kind == "family" else None
    if finished and record.kind == "generation":
        review_request = finished["result"].get("review_request")
    for event in log:
        if event.slot.startswith("review:"):
            review_request = event.payload["request"]
    return {"id": record.id, "kind": record.kind, "created_by": record.created_by, "created_at": record.created_at.isoformat(),
            "status": status, "request_sha256": record.request_hash, "review_request": review_request,
            "result": finished["result"] if finished else None,
            "execution_request": finished.get("execution_request") if finished else None}


def generation_request(session, payload, settings):
    family_record = session.get(WorkflowRecord, payload["family_workflow_id"])
    if family_record is None or family_record.kind != "family":
        raise ValueError("family workflow not found")
    family = state(session, family_record)["review_request"]
    if analyze_family(family)["review_findings"]:
        raise ValueError("current authenticated source/grounding review is required")
    history = []
    for id in payload.get("history_ids", []):
        old = session.get(WorkflowRecord, id)
        if (old is None or old.kind != "generation" or old.request_payload["family_workflow_id"] != family_record.id
                or old.request_payload["prompt"] != payload["prompt"]):
            raise ValueError("revision history must belong to the same family and prompt")
        previous = state(session, old)
        bound = previous["review_request"]
        if not bound or not previous["result"] or previous["result"].get("candidate") is None:
            raise ValueError("history draft is not available")
        candidate = deepcopy(previous["result"]["candidate"])
        candidate["reviews"] = {k: v for k, v in bound.get("reviews", {}).items() if k in {"grounding", "behavior"}}
        history.append(candidate)
    result = {"schema": "vle-live/v1", "family": family, "prompt": payload["prompt"], "history": history,
              "provider": settings.model_provider, "model": settings.model_name,
              "max_cycles": payload.get("max_cycles", 2), "max_tokens": payload.get("max_tokens", 2048)}
    if settings.model_endpoint:
        result["endpoint"] = settings.model_endpoint
    return result


def review_workflow(session, record, *, actor, scope, notes, decision, request_id):
    record = session.scalar(select(WorkflowRecord).where(WorkflowRecord.id == record.id).with_for_update())
    current = state(session, record)["review_request"]
    if current is None:
        raise ValueError("workflow has no reviewable family/draft")
    updated = record_family_review(current, scope=scope, reviewer=actor, notes=notes, decision=decision)
    add_event(session, record, "review:" + str(uuid.uuid4()), actor, {"request": updated}, request_id)
    return state(session, record)


def process_one(database, settings, *, transport=None):
    if not settings.inference_enabled:
        return {"status": "disabled"}
    id = None
    # Unique durable claim is committed before any network call. A claimed job is
    # never retried automatically, including after loss of the worker/process.
    with database.session() as session:
        query = select(WorkflowRecord).where(WorkflowRecord.kind == "generation", ~exists().where(
            WorkflowEventRecord.workflow_id == WorkflowRecord.id,
            WorkflowEventRecord.slot.in_(["claim", "finish"]))).order_by(WorkflowRecord.created_at).limit(1)
        record = session.scalar(query.with_for_update(skip_locked=True))
        if record is None:
            return {"status": "idle"}
        try:
            with session.begin_nested():
                add_event(session, record, "claim", "worker", {"status": "claimed"}, "worker:" + record.id)
        except IntegrityError:
            return {"status": "contended"}
        id, payload = record.id, record.request_payload
    execution = None
    try:
        with database.session() as session:
            execution = generation_request(session, payload, settings)
        def configured_transport(body, **kwargs):
            kwargs["api_key"] = settings.model_api_key.get_secret_value() if settings.model_api_key else None
            return (ollama_chat if settings.model_provider == "ollama" else vllm_chat)(body, **kwargs)
        result = generate_candidate(execution, transport=transport or configured_transport)
    except Exception:
        # Includes unexpected worker faults; preserve a terminal failure without
        # exposing source text/credentials in errors or retrying a provider call.
        result = {"status": "worker_failed", "deployment_authorized": False, "legal_certification": False}
    with database.session() as session:
        record = session.scalar(select(WorkflowRecord).where(WorkflowRecord.id == id).with_for_update())
        slot = "late:" + str(uuid.uuid4()) if any(e.slot == "finish" for e in events(session, id)) else "finish"
        add_event(session, record, slot, "worker", {"result": result, "execution_request": execution}, "worker:" + id)
    return {"id": id, "status": result["status"]}


def main():
    from .config import get_settings
    from .database import Database
    parser = argparse.ArgumentParser(description="Governed inference worker; never automatically retries claimed jobs")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=2)
    args = parser.parse_args()
    if not 0.1 <= args.poll_seconds <= 30:
        parser.error("poll-seconds must be 0.1-30")
    settings = get_settings()
    database = Database(settings)
    try:
        if not database.ready(require_migrations=settings.environment == "production"):
            raise SystemExit("database schema is not ready")
        with tempfile.TemporaryDirectory(prefix="vle-worker-") as marker_dir:
            while True:
                (Path(marker_dir) / "heartbeat").touch()
                result = process_one(database, settings)
                if args.once:
                    print(json.dumps(result))
                    return
                time.sleep(args.poll_seconds)
    finally:
        database.dispose()


if __name__ == "__main__":
    main()
