# Security Model

## Trust boundaries

The service assumes TLS termination and network admission control at a trusted ingress. The application independently authenticates API requests using bearer API keys, applies role checks, request-size limits, database-backed rate limits, and idempotency.

## Roles

- `reader`: view governed sources and authorized runs.
- `operator`: ingest sources and request evaluations.
- `reviewer`: approve/reject extracted candidates and select interpretations.
- `auditor`: inspect audit records and blocked high-stakes runs.
- `admin`: role superset. Separation-of-duties checks still prevent the source creator from reviewing when enabled.

## Secret handling

API tokens are read from secret-injected configuration. Only SHA-256 token digests are retained in application memory for comparisons. Raw keys are never persisted to the service database or included in audit/log payloads.

## Tamper evidence and immutability

State changes generate audit events with a canonical payload hash and a hash of the previous event. PostgreSQL uses an advisory transaction lock while appending so the global chain has a single order across replicas. Production migration triggers reject UPDATE/DELETE on source documents, reviews, interpretation selections, lifecycle runs and audit events.

The hash chain provides tamper evidence, not an external trust anchor. Export audit-chain heads periodically to an independently controlled log/SIEM or immutable object store for stronger non-repudiation.

## Abuse resistance

- request bodies are capped before application parsing;
- Pydantic bounds the number/size of facts, actions and metadata fields;
- database-backed fixed-window limits apply per API key across replicas;
- state-changing endpoints require an idempotency key;
- errors do not echo stack traces to callers;
- server-identifying headers are disabled by the reference Uvicorn command;
- production docs/OpenAPI are disabled.

## Security verification

CI runs unit/integration tests, Ruff, Bandit, dependency auditing and a container build with HIGH/CRITICAL vulnerability scanning. Direct runtime dependencies are version-pinned.

## Out of scope

The repository cannot itself provide cloud IAM, TLS certificates, WAF/bot filtering, database/storage encryption keys, host patching, backup custody, SIEM retention or organization-specific incident response. Those controls belong to the deployment platform and must be configured before public exposure.
