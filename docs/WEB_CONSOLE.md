# Governed Web Console

The integrated web console at `/` is an operations surface over the existing VLE–LEGATO service. It does not implement a parallel reasoning path and does not weaken any backend authorization, human-review, interpretation-selection, separation-of-duties, readiness, or audit controls.

## Role-aware workflow

The browser keeps an API key only in `sessionStorage` for the current tab. `GET /v1/session` returns only the authenticated subject and role names so the UI can show relevant controls; it never returns the bearer token, its hash, or configured secrets.

The console supports the following governed path:

1. **Operator** — collect a bounded IP or competition-law batch from the configured Légifrance provider and persist source versions with provenance.
2. **Reader** — inspect stored source versions, extraction evidence, candidates, formalization previews, and summarized lifecycle history.
3. **Reviewer** — approve/reject/request revision of extracted candidates. If formalization is ambiguous, explicitly select one formal interpretation and provide a rationale. Existing separation-of-duties rules remain authoritative.
4. **Operator** — evaluate the reviewed source against facts, actor, observed actions, date, and high-stakes flag. High-stakes decisions remain unauthorized unless the service readiness gates pass.
5. **Auditor** — verify the hash-linked audit chain and export deterministic audit bundles for persisted lifecycle runs.

## Console-specific read APIs

`GET /v1/console/sources/{source_id}/state` exposes only workflow state needed by the browser: the source creator and current non-stale interpretation selections.

`GET /v1/console/runs?source_id=<uuid>&limit=50` exposes summarized run metadata only: run/source IDs, requester, time, high-stakes flag, decision authorization, readiness status/failed gates, and compliance status. It intentionally omits the stored fact context and full reasoning payload. Complete evidence remains available through the auditor-only audit-bundle endpoint.

## Security properties

- All mutating review, interpretation, and evaluation requests still use the core service endpoints and require their existing roles.
- Review and interpretation writes still require `Idempotency-Key` headers.
- Reviewer controls in the UI are only convenience affordances; the server remains the enforcement point.
- Source acquisition remains restricted to configured Légifrance transports and fixed corpora.
- The page is served with `Cache-Control: no-store`, a restrictive Content Security Policy, `Referrer-Policy: no-referrer`, and `X-Content-Type-Options: nosniff`.
- The console does not persist API keys outside the current browser tab.
- Lifecycle run summaries do not disclose stored fact/context payloads.
- Audit-bundle downloads remain restricted to auditor/admin principals.

## High-stakes behavior

A high-stakes evaluation that fails research-readiness gates is persisted but returned as blocked. This is intentional: the blocked decision remains auditable without becoming authorized. The console displays failed gates and never converts a blocked result into a compliance or certification claim.

## Testing

`tests/test_web_console_workflow.py` verifies the browser surface, role/session endpoint, human review and interpretation persistence, summarized run history without fact-context leakage, security headers, and auditor-only bundle export. The normal production CI still supplies Python 3.12/3.13 tests, Ruff, Bandit, dependency audit, PostgreSQL migration integration, semantic verification, container runtime smoke, Trivy HIGH/CRITICAL gating, and final-image SBOM generation.
