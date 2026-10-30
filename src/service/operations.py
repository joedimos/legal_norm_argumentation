"""Executable deployment smoke/load/source probes and restore verification.

These tools operate on explicitly selected staging/isolated targets. Reports are
observations for release custodians, never independent legal certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from urllib.parse import urlsplit

import sqlalchemy as sa
import urllib3

from src.audit_bundle import canonical_json, digest, verify_bundle
from src.model_runtime import strict_json

from .config import Settings
from .database import Database
from .release_policy import pinned_json
from .repository import verify_audit_chain

GOVERNED_TABLES = ("source_documents", "candidate_reviews", "interpretation_selections", "lifecycle_runs",
                  "audit_events", "workflow_requests", "workflow_events")
OPERATIONAL_TABLES = ("import_cursors", "idempotency_records", "rate_limit_buckets")


class SnapshotBound(ValueError):
    pass


def extended_database_snapshot(database, *, max_rows=100000, max_bytes=67108864, max_seconds=30, monotonic=time.monotonic):
    """Consistent bounded v2 snapshot, including all three operational tables."""
    if (type(max_rows) is not int or not 1 <= max_rows <= 10000000
            or type(max_bytes) is not int or not 1 <= max_bytes <= 1073741824
            or type(max_seconds) is not int or not 1 <= max_seconds <= 300):
        raise ValueError("invalid snapshot bounds")
    start, rows, bytes_used = monotonic(), 0, 0

    def budget(payload=None):
        nonlocal bytes_used
        if payload is not None:
            bytes_used += len(canonical_json(payload))
        if rows > max_rows or bytes_used > max_bytes or monotonic() - start >= max_seconds:
            raise SnapshotBound("snapshot_bound")

    if not database.ready(require_migrations=True):
        raise ValueError("migrated database is not ready")
    tables = {}
    with database.session() as session:
        if database.engine.dialect.name == "postgresql":
            session.connection(execution_options={"isolation_level": "REPEATABLE READ"})
            session.execute(sa.text("SELECT set_config('statement_timeout', :timeout, true)"), {"timeout": str(max_seconds * 1000)})
        elif database.engine.dialect.name == "sqlite":
            session.execute(sa.text("BEGIN"))
        else:
            raise ValueError("unsupported consistent snapshot backend")
        chain = verify_audit_chain(session, max_events=max_rows, on_event=lambda event: budget())
        if not chain["valid"]:
            if chain.get("status") == "unknown":
                raise SnapshotBound("snapshot_audit_bound")
            raise ValueError("audit chain is invalid")
        for name in GOVERNED_TABLES + OPERATIONAL_TABLES:
            budget()
            table = sa.Table(name, sa.MetaData(), autoload_with=session.connection())
            hasher, count = hashlib.sha256(), 0
            hasher.update(b"[")
            query = sa.select(table).order_by(*table.primary_key.columns).execution_options(yield_per=100)
            for row in session.execute(query).mappings():
                rows += 1
                data = {k: (v.replace(tzinfo=UTC) if v.tzinfo is None else v.astimezone(UTC)).isoformat()
                        if isinstance(v, datetime) else v for k, v in row.items()}
                raw = canonical_json(data)
                budget(data)
                if count:
                    hasher.update(b",")
                hasher.update(raw)
                count += 1
            hasher.update(b"]")
            tables[name] = {"count": count, "sha256": hasher.hexdigest()}
        budget()
    return {"schema": "vle-database-snapshot/v2", "tables": {n: tables[n] for n in GOVERNED_TABLES},
            "operational_tables": {n: tables[n] for n in OPERATIONAL_TABLES}, "audit_chain": chain,
            "bounds": {"max_rows": max_rows, "max_bytes": max_bytes, "max_seconds": max_seconds},
            "consistency": "single_read_transaction"}


def database_snapshot(database):
    if not database.ready(require_migrations=True):
        raise ValueError("migrated database is not ready")
    tables = {}
    with database.session() as session:
        chain = verify_audit_chain(session)
        if not chain["valid"]:
            raise ValueError("audit chain is invalid")
        for name in GOVERNED_TABLES:
            table = sa.Table(name, sa.MetaData(), autoload_with=database.engine)
            rows = []
            for row in session.execute(sa.select(table).order_by(*table.primary_key.columns)).mappings():
                rows.append({key: value.isoformat() if isinstance(value, datetime) else value for key, value in row.items()})
            tables[name] = {"count": len(rows), "sha256": digest(canonical_json(rows))}
    return {"schema": "vle-database-snapshot/v1", "tables": tables, "audit_chain": chain}


class ProbeClient:
    def __init__(self, base):
        url = urlsplit(base)
        if (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment
                or url.path not in {"", "/"} or (url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"})):
            raise ValueError("probe requires an HTTPS origin or loopback HTTP origin")
        self.base = base.rstrip("/")
        self.pool = urllib3.PoolManager()

    def call(self, path, *, token=None, payload=None, binary=False):
        headers = {"Accept-Encoding": "identity"}
        if token:
            headers["Authorization"] = "Bearer " + token
        if payload is not None:
            headers.update({"Content-Type": "application/json", "Idempotency-Key": str(uuid.uuid4())})
        response = self.pool.request("POST" if payload is not None else "GET", self.base + path,
                                     body=canonical_json(payload) if payload is not None else None, headers=headers,
                                     timeout=urllib3.Timeout(total=30), retries=False, redirect=False, preload_content=False)
        try:
            raw = response.read(16 * 1024 * 1024 + 1, decode_content=False)
            if response.status != 200 and response.status != 201:
                raise ValueError("deployment probe received unsuccessful HTTP status")
            if len(raw) > 16 * 1024 * 1024 or response.headers.get("Content-Encoding", "identity") != "identity":
                raise ValueError("probe response violates size/encoding bound")
            return (raw, response.headers) if binary else strict_json(raw)
        finally:
            response.close()


def smoke(client, operator, reviewer, auditor, *, reasoning_backend="core"):
    if reasoning_backend not in {"core", "pyreason"}:
        raise ValueError("unsupported backend")
    identities = [client.call("/v1/session", token=token)["subject"] for token in (operator, reviewer)]
    if identities[0] == identities[1]:
        raise ValueError("smoke requires separate operator and reviewer identities")
    client.call("/health/live")
    client.call("/health/ready")
    source = client.call("/v1/sources", token=operator, payload={"source_id": "staging-smoke:" + str(uuid.uuid4()),
                         "text": "The operator must retain logs."})
    id = source["id"]
    for candidate in source["candidates"]:
        client.call(f"/v1/sources/{id}/reviews", token=reviewer,
                    payload={"candidate_id": candidate["id"], "decision": "approved", "note": "Synthetic staging smoke fixture only."})
    run = client.call(f"/v1/sources/{id}/evaluate", token=operator,
                      payload={"high_stakes": False, "observed_actions": ["retain logs"],
                               **({"reasoning_backend": reasoning_backend} if reasoning_backend != "core" else {})})
    if run["decision_authorized"] or run["authorization_scope"] != "research_only" or run["compliance"]["status"] != "compliant":
        raise ValueError("research/production authorization boundary failed")
    if reasoning_backend == "pyreason" and run["compliance"]["reasoning"]["backend"]["status"] != "computed":
        raise ValueError("native smoke did not compute")
    archive, headers = client.call(f'/v1/runs/{run["run_id"]}/audit-bundle', token=auditor, binary=True)
    if not verify_bundle(archive, expected_manifest_digest=headers["X-Audit-Manifest-SHA256"])["valid"]:
        raise ValueError("smoke audit verification failed")
    return {"source_id": id, "run_id": run["run_id"], "synthetic": True, "audit_verified": True}


def load_probe(client, token, *, count=100, concurrency=4, max_p95_ms=1000, evaluation_source=None, reasoning_backend="core", worker_mode="isolated"):
    if reasoning_backend not in {"core", "pyreason"} or worker_mode not in {"isolated", "pooled"}:
        raise ValueError("unsupported backend/worker mode")
    if not 1 <= count <= 1000 or not 1 <= concurrency <= 32 or max_p95_ms <= 0:
        raise ValueError("load probe bounds exceeded")
    def once(_):
        start = time.monotonic()
        try:
            if evaluation_source:
                result = client.call(f"/v1/sources/{evaluation_source}/evaluate", token=token,
                                     payload={"high_stakes": False, "observed_actions": ["retain logs"],
                               **({"reasoning_backend": reasoning_backend} if reasoning_backend != "core" else {})})
                if result.get("decision_authorized") or result.get("compliance", {}).get("status") != "compliant":
                    raise ValueError("load workload boundary failed")
                if reasoning_backend == "pyreason":
                    backend = result["compliance"]["reasoning"]["backend"]
                    if backend["status"] != "computed" or backend["worker_mode"] != worker_mode:
                        raise ValueError("native load did not compute with the selected transport")
            else:
                client.call("/v1/session", token=token)
            return (time.monotonic() - start) * 1000, True
        except Exception:
            return (time.monotonic() - start) * 1000, False
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        observations = list(pool.map(once, range(count)))
    times = sorted(x[0] for x in observations)
    p95 = times[max(0, (95 * count + 99) // 100 - 1)]
    return {"passed": all(x[1] for x in observations) and p95 <= max_p95_ms,
            "requests": count, "concurrency": concurrency, "errors": sum(not x[1] for x in observations),
            "p95_ms": p95, "median_ms": statistics.median(times), "slo_p95_ms": max_p95_ms,
            **({"reasoning_backend": reasoning_backend, "worker_mode": worker_mode,
                "backend_computations_checked": bool(evaluation_source)} if reasoning_backend == "pyreason" else {}),
            "coverage": "authenticated_evaluation_with_audit_writes" if evaluation_source else "authenticated_session_endpoint_only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "load", "snapshot", "restore", "legifrance", "ip-sources"), required=True)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--build-revision", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--reasoning-backend", choices=("core", "pyreason"), default="core")
    parser.add_argument("--worker-mode", choices=("isolated", "pooled"), default="isolated")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-p95-ms", type=float, default=1000)
    parser.add_argument("--with-evaluation", action="store_true", help="Create a reviewed synthetic source and exercise persisted evaluation/audit writes")
    parser.add_argument("--baseline")
    parser.add_argument("--baseline-sha256")
    parser.add_argument("--confirm-isolated", action="store_true")
    parser.add_argument("--include-operational", action="store_true", help="Use bounded v2 snapshots including import/idempotency/rate state")
    parser.add_argument("--snapshot-max-rows", type=int, default=100000)
    parser.add_argument("--snapshot-max-bytes", type=int, default=67108864)
    parser.add_argument("--snapshot-max-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.include_operational and args.mode not in {"snapshot", "restore"}:
        parser.error("--include-operational applies only to snapshot/restore")
    if len(args.build_revision) != 40 or any(x not in "0123456789abcdef" for x in args.build_revision):
        parser.error("build-revision must be a full lowercase Git SHA")
    report = {"schema": "vle-operational-evidence/v1", "kind": args.mode, "passed": False,
              "deployment_id": args.deployment_id, "build_revision": args.build_revision,
              "observed_at": datetime.now(UTC).isoformat()}
    if args.mode == "ip-sources":
        from .ip_release import SCOPE, source_probe_bounds
        report.update(schema="vle-operational-evidence/v2", kind="ip_sources", scope=SCOPE, status="unknown",
                      evidence_origin="live_provider_probe", bounds=source_probe_bounds(),
                      decision_authorized=False, legal_certification=False)
    if args.include_operational:
        report["bounds"] = {"max_rows": args.snapshot_max_rows, "max_bytes": args.snapshot_max_bytes,
                            "max_seconds": args.snapshot_max_seconds}
    database = None
    client = None
    try:
        if args.mode in {"snapshot", "restore"}:
            url = os.environ.get("VLE_RESTORE_DATABASE_URL" if args.mode == "restore" else "VLE_DATABASE_URL")
            if not url:
                raise ValueError("database URL environment variable is required")
            if args.mode == "restore":
                original = sa.engine.make_url(os.environ["VLE_DATABASE_URL"])
                restored = sa.engine.make_url(url)
                if not args.confirm_isolated or (original.host, original.port, original.database) == (restored.host, restored.port, restored.database):
                    raise ValueError("restore verification requires an explicitly isolated target database")
            database = Database(Settings(environment="test", database_url=url, _env_file=None))
            if args.include_operational:
                report["bounds"] = {"max_rows": args.snapshot_max_rows, "max_bytes": args.snapshot_max_bytes,
                                    "max_seconds": args.snapshot_max_seconds}
                snapshot = extended_database_snapshot(database, max_rows=args.snapshot_max_rows,
                                                      max_bytes=args.snapshot_max_bytes, max_seconds=args.snapshot_max_seconds)
            else:
                snapshot = database_snapshot(database)
            if args.mode == "snapshot":
                print(json.dumps(snapshot, sort_keys=True))
                return
            baseline = pinned_json(args.baseline, args.baseline_sha256)
            if baseline != snapshot or not any(t["count"] for t in snapshot["tables"].values()):
                raise ValueError("restored governed data does not match nonempty frozen baseline")
            report.update(passed=True, snapshot_sha256=digest(canonical_json(snapshot)))
            if args.include_operational:
                report.update(snapshot_schema=snapshot["schema"], operational_tables_verified=list(OPERATIONAL_TABLES))
        elif args.mode == "ip-sources":
            from .ip_release import ip_source_evidence
            report = ip_source_evidence(Settings(), deployment_id=args.deployment_id, build_revision=args.build_revision)
        elif args.mode == "legifrance":
            from .legifrance_provider import build_legifrance_client
            provider = build_legifrance_client(Settings())
            coverage = {kind: provider.fetch_corpus(kind, limit=1, offset=0).selected for kind in ("ip", "competition")}
            if any(n < 1 for n in coverage.values()):
                raise ValueError("live source probe returned no provisions")
            report.update(passed=True, corpora=coverage, coverage="first_article_in_each_configured_corpus")
        else:
            client = ProbeClient(args.base_url or "")
            if args.mode == "smoke":
                report.update(smoke(client, os.environ["VLE_PROBE_OPERATOR_TOKEN"], os.environ["VLE_PROBE_REVIEWER_TOKEN"],
                                    os.environ["VLE_PROBE_AUDITOR_TOKEN"], reasoning_backend=args.reasoning_backend), passed=True)
            else:
                fixture = smoke(client, os.environ["VLE_PROBE_OPERATOR_TOKEN"], os.environ["VLE_PROBE_REVIEWER_TOKEN"],
                                os.environ["VLE_PROBE_AUDITOR_TOKEN"], reasoning_backend=args.reasoning_backend) if args.with_evaluation else None
                report.update(load_probe(client, os.environ["VLE_PROBE_OPERATOR_TOKEN"], count=args.requests,
                                          concurrency=args.concurrency, max_p95_ms=args.max_p95_ms,
                                          evaluation_source=fixture["source_id"] if fixture else None,
                                          reasoning_backend=args.reasoning_backend, worker_mode=args.worker_mode))
    except SnapshotBound:
        report.update(passed=False, status="unknown", reason="snapshot_bound", bound_hit=True)
    except Exception:
        report.update(passed=False, reason="operational_probe_failed")
    finally:
        if client:
            client.pool.clear()
        if database:
            database.dispose()
    print(json.dumps(report, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
