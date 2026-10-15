# Production Operations Runbook

This document defines the supported production posture for the VLE-LEGATO service layer. It covers service engineering; it does **not** convert research readiness into legal or regulatory certification.

## Supported deployment

- Python 3.12-3.13; the reference container uses Python 3.13.
- PostgreSQL 18.x for production data.
- Schema changes are applied only through Alembic. `VLE_AUTO_CREATE_SCHEMA=true` is rejected in production.
- Run one ASGI process per container and scale at the container/orchestrator layer. Apply HTTPS at the ingress/load balancer.

## Required production configuration

`VLE_ENVIRONMENT=production` activates fail-fast validation. Startup fails unless PostgreSQL is configured, interactive docs and schema auto-creation are disabled, wildcard CORS is absent, distinct operator/reviewer identities exist when separation of duties is enabled, and a strong metrics token is configured.

Secrets must be injected by the deployment platform's secret manager. Never store `.env` files or raw API tokens in source control, images, logs or telemetry.

## Deployment sequence

1. Back up PostgreSQL and verify backup completion.
2. Build the container from the tagged commit.
3. Run `alembic upgrade head` once as a migration job/init step.
4. Start the application containers.
5. Require `/health/live` = 200 and `/health/ready` = 200 before accepting traffic.
6. Run an authenticated smoke test covering source ingestion, review, evaluation and audit retrieval.
7. Promote traffic gradually at the ingress.

## Backups and recovery

Use automated PostgreSQL backups, point-in-time recovery/WAL archiving for high-stakes workloads, encrypted backup custody separated from application credentials, and regular isolated restore exercises. Recovery acceptance requires successful migration-state verification, readiness checks, and audit-chain validation.

## Availability and scaling

- Run at least two application replicas behind a health-checking load balancer for HA deployments.
- Rate limiting and idempotency are stored in PostgreSQL and therefore apply across workers/replicas.
- Size connection pools against database limits and replica count.
- The reference Uvicorn command limits concurrency to 200 per container; tune only after load testing.

## Observability

Logs are JSON and include request IDs. `/metrics` requires a dedicated bearer token in production. Prometheus metrics cover requests, in-flight requests, latency and lifecycle authorization outcomes. Configure `VLE_OTLP_ENDPOINT` for distributed traces. Request/response bodies and secrets are not intentionally logged.

Suggested alerts include readiness failure, elevated 5xx, p95 latency SLO breach, connection-pool exhaustion, sustained 429s, unexpected high-stakes authorization blocks, and audit-chain validation failure.

## High-stakes governance

An operator ingests a source; a distinct reviewer approves/rejects each candidate and resolves ambiguous interpretations; only then may a high-stakes evaluation return an authorized result. A high-stakes run returns HTTP 409 when readiness gates fail. Full blocked-run internals remain restricted to reviewer/auditor roles.

## Incident response

For suspected incorrect normative behavior, stop automated downstream actioning, capture request/run/source IDs and the release tag, preserve database/audit evidence, reproduce on the same version, create new review records rather than mutating prior evidence, deploy a new version, and rerun affected cases. Rotate compromised credentials immediately.

## Data retention

The service intentionally exposes no DELETE/UPDATE endpoints for governed source, review, selection, run or audit records. PostgreSQL triggers reject mutation of those tables. Any legally required retention/deletion process must be a separately governed administrative procedure.
