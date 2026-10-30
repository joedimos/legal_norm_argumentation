# Governed production release and persistent inference

For the 0.9.0 supervised French/U.S. IP profile, installed-package migrations and
versioned v2 evidence requirements, follow [RELEASE_CANDIDATE.md](RELEASE_CANDIDATE.md).
The v1/legacy policy below remains supported. Current schema head is
`0004_automatic_imports`; older migration descriptions below are historical.

Version 0.6 adds stricter release authorization, persistent research workflows,
an inference worker and executable staging probes. It does not supply a model
server, real independent legal annotations, PISTE credentials, TLS infrastructure
or organizational approval. Missing evidence keeps production decisions blocked.

## Authorization policies

Version 0.6.1 removes the external-validation hard block by default.
`VLE_RELEASE_ENFORCEMENT=reviewed` permits high-stakes report authorization after
current source review, interpretation resolution and determinate reasoning.
Missing release/independent-case evidence is advisory and stays visible in
`production_release`; it is never converted into passing validation or legal
certification. Authorized responses use `authorization_scope: reviewed_operation`
when external validation has not passed, with `override_used: true`. The configured
policy and override are retained in the run and audit event.

Set `VLE_RELEASE_ENFORCEMENT=strict` to restore mandatory external validation.
API callers cannot change this deployment policy. Both modes preserve authentication,
source review, separation of duties, reasoning checks and evidence integrity.
A policy, target or validation-status change requires a new Idempotency-Key for a
new evaluation; old immutable receipts do not become new authorizations.

The strict policy behaves as follows:

`research_ready` is a necessary research result, not production authorization.
A high-stakes evaluation under `strict` requires all three predicates:

1. Current source review, interpretation selection and determinate reasoning.
2. A separately pinned, unexpired deployment release evidence graph for this build.
3. A favorable independently signed case review for the exact evaluation target.

Missing/invalid evidence returns HTTP 409 and persists the blocked evaluation.
Research-only evaluations return HTTP 200 with `decision_authorized: false` and
`authorization_scope: research_only`. **Compatibility change:** HTTP 200 and the
caller-supplied `high_stakes: false` flag never authorize a production decision.
The API's authorization is an organizational release decision for a report;
it is not legal certification or permission to enact an action.

`POST /v1/sources/{id}/release-target` takes an evaluation context and returns the
complete target plus `target_sha256`. The hash binds source content/candidates,
formalizations, current review/selection snapshots, facts/actions/actor/date,
metadata, explicit jurisdiction/capacity and exact configured build revision.
Production cases require nonempty actor, capacity, jurisdiction and canonical
`at_date`; every signed unit must match that exact applicability scope. Give the full target to reviewers
before they sign. Every signed submission and any adjudication must add the same
`target_sha256` to its plain payload **before signing**.

Upload `{target_sha256, assessment}` to
`POST /v1/sources/{id}/legal-assessments` under a separate reviewer API identity.
The signed packet must quote this exact normative source. Every effective case
judgment must support applicability, faithfulness, conditions/exceptions and
response behavior; factual context needs supported observation grounding;
harmful-assistance judgments must be adverse to the harmful-assistance proposition
or not applicable. Disagreement, uncertainty, wrong source and stale target block
release. The source operator cannot register its own independent assessment.

Add `legal_assessment_id` to `/evaluate`. Current registry signatures, revocation,
validity and case binding are rechecked on every high-stakes evaluation, including
idempotent replays. A previously authorized response cannot reauthorize after
context/review/trust changes. Read-only historical run records remain immutable
and are explicitly historical, not new authorizations.

The registry custodian still verifies real-world identity, expertise, independence
and conflicts. Key signatures prove attribution under that trust anchor only.
No classifier, solver or cryptographic primitive establishes legal faithfulness.

## Deploy and migrate

Use `deployment.env.example` as the configuration checklist, supplying actual
separate operator/reviewer/auditor API principals and secret-managed credentials.
The example intentionally contains no working secrets or approval evidence.

```bash
docker compose --env-file deployment.env config --quiet
docker compose --env-file deployment.env up --build -d
```

Compose now passes the PISTE selection and credentials to the app and worker.
When enabled, production defaults to PISTE and refuses missing/empty credentials.
For an explicitly scoped installation without collection, set
`VLE_LEGIFRANCE_ENABLED=false`. Explicit `html` fallback remains available but
must be monitored; it is not an invisible fallback from failed PISTE credentials.

Alembic revision `0002_governed_workflows` creates append-only workflow requests
and events. PostgreSQL triggers reject updates/deletions. Schema readiness checks
require this exact revision, not merely the presence of `alembic_version`.
Back up first. Migration rollback drops the new evidence tables and is destructive;
use forward recovery for deployed evidence, not an unattended downgrade.

Both API and worker share PostgreSQL and server-owned model configuration.
Inference is disabled by default. Enable it only with the intended model name and
HTTPS endpoint (or a loopback listener accessible inside the worker container).
A host-loopback model is not automatically accessible from a container. Use your
approved TLS/network/SSH-forwarding arrangement; HTTP to arbitrary hosts is rejected.
The `worker` service runs separately and records a private temporary heartbeat;
Compose checks it with a five-minute freshness window. Monitor that signal and
queue age. A socket timeout is not remote inference cancellation.

Mount a read-only evidence directory at `/evidence`. Configure registry/release
paths inside it, each with a separately distributed canonical SHA-256 pin. The
pins and files must be updated together by an authorized deployment rollout.
The service never accepts caller-selected trust-registry files, pins or model URLs.
The image now includes the signature-verification backend.

## Persistent workflow API and browser console

Open `/workflows` from the source console. The interface uses the same API-key
roles and same-origin API; text is rendered as text, never injected as HTML.

| Operation | Endpoint | Role |
| --- | --- | --- |
| Store unreviewed base family | `POST /v1/workflows/families` | operator |
| Review family/grounding or generated behavior | `POST /v1/workflows/{id}/reviews` | separate reviewer |
| Queue initial draft or revision | `POST /v1/workflows/generations` | family operator |
| Read/list persistent work | `GET /v1/workflows`, `GET /v1/workflows/{id}` | owner, reviewer, auditor |
| Export complete immutable workflow evidence | `GET /v1/workflows/{id}/audit-bundle` | auditor |
| Mark an unfinished job interrupted | `POST /v1/workflows/{id}/interrupt` | admin |
| Inspect deployment gate status | `GET /v1/release-status` | authenticated principal |

Mutations require an `Idempotency-Key`. Permanent deduplication also reuses the
same workflow for identical kind/actor/payload after ephemeral idempotency expiry.
Family submissions cannot carry caller-created approval receipts or draft text.
Reviews identify the authenticated principal and forbid self-review. Generated
candidates cannot overwrite normative approval. Full prompts/results are private
to their owner and review/audit roles; list responses contain summaries only.

Generation input has `family_workflow_id`, `prompt`, optional ordered `history_ids`
(up to four), `max_cycles` (1–5) and `max_tokens` (1–8192). It has no endpoint,
provider, model, receipt or fact-invention fields. History must belong to the same
family and prompt. The worker reconstructs current reviews and rechecks all prior
candidates. Only reviewed violations permit another draft; reviewed satisfaction,
uncertainty or missing/revoked review stops it.

The worker commits a unique database claim **before** calling the model. It never
automatically retries a claimed job. If a process dies between provider execution
and result persistence, the result is uncertain and the job stays
`running_or_interrupted`. An admin must investigate and mark it interrupted; do
not blindly retry a possibly billed/executed call. After investigation, the operator
may submit an explicit retry with `retry_of` and `retry_reason`; it must preserve
the original request and point to a terminal failed/interrupted job. The retry is
a separate permanently deduplicated job with its own at-most-once claim.
A reviewer must still assess any resulting draft. Interruption does not cancel
remote compute. Any later-arriving result is retained as a late audit event and
cannot overwrite the interruption. This is at-most-once initiation per job, not
an exactly-once promise about a remote model server.

## Release evidence graph

`VLE_RELEASE_EVIDENCE_PATH` names a `vle-release-evidence/v1` JSON manifest with
exactly `schema`, `build_revision`, `deployment_id`, timezone-aware `expires_at`,
at least two distinct actual `approved_by` custodian IDs, and `evidence`.
Each evidence entry is `{filename,sha256}` using a flat JSON filename beside the
manifest and a hash of canonical JSON. No path traversal or external symlink is
accepted. The seven required entries are:

| Entry | Required evidence |
| --- | --- |
| annotations | Complete `vle-signed-assessment/v1` held-out gold records, reverified against the current registry |
| benchmark | `vle-production-benchmark/v1` with this build, `split: test`, `case_count`, extraction F1, compliance accuracy, `corpus_sha256` equal to the frozen packet hash and `independent_assessment_sha256` binding the signed gold artifact |
| model | Actual successful `vle-live-result/v1`, one generation call, valid candidate/bound behavior contract, response/prompt hashes, and the configured provider/model/endpoint |
| smoke | Successful `vle-operational-evidence/v1`, matching build/deployment, `kind: smoke`, `audit_verified: true` |
| load | Successful matching operational report with persisted-evaluation/audit workload, request count, zero errors and measured p95 |
| restore | Successful matching isolated restore report with a governed-data `snapshot_sha256` |
| legifrance | Successful matching live-source report with at least one provision from both IP and competition corpora |

Defaults require at least 100 independently annotated held-out units, extraction
F1 and compliance accuracy at least 0.9, at least 100 persisted evaluation load
requests, and p95 at most 1000 ms. These are configurable deployment policies,
not scientifically universal thresholds. Gold unit count and test splits are
checked; uncertainty, negative cases and abstentions must remain in the benchmark
denominator. The custodian must retain the complete predictions/evaluation script,
source permissions, expert credentials and protocol, weights hashes and server
configuration, and verify the reported metrics. The gate validates a pinned
organizational evidence graph and declared metrics; it does not independently
reproduce a legal study or prove that its reports are truthful.

No test fixture, simulated provider response or fabricated reviewer may be used
as production evidence. The passing release fixtures in unit tests are explicitly
synthetic regression inputs, not a supplied release manifest.

## Execute staging probes and recovery drills

Set `VLE_PROBE_OPERATOR_TOKEN`, `VLE_PROBE_REVIEWER_TOKEN` and
`VLE_PROBE_AUDITOR_TOKEN` through the secret manager. Use a staging origin under
your control. Smoke/load tests create permanent, clearly named synthetic records;
they do not delete audit evidence or authorize high-stakes decisions.

```bash
python -m src.service.operations --mode smoke --base-url https://YOUR-STAGING-HOST \
  --deployment-id YOUR_DEPLOYMENT --build-revision YOUR_BUILD_SHA > smoke.json
python -m src.service.operations --mode load --with-evaluation \
  --base-url https://YOUR-STAGING-HOST --requests 100 --concurrency 4 \
  --max-p95-ms 1000 --deployment-id YOUR_DEPLOYMENT --build-revision YOUR_BUILD_SHA > load.json
python -m src.service.operations --mode legifrance \
  --deployment-id YOUR_DEPLOYMENT --build-revision YOUR_BUILD_SHA > legifrance.json
```

Without `--with-evaluation`, the load tool exercises authenticated session reads
only; that lightweight report cannot satisfy the release gate. The bounded tool
is a baseline workload, not an exhaustive capacity or penetration test. Test your
actual concurrency, model queue/resource budgets, sustained traffic, failure modes
and downstream consumers separately before choosing production SLOs.

Before backup, use `--mode snapshot` against the quiesced staging database under
`VLE_DATABASE_URL` and keep the canonical snapshot digest separately. Perform a
real `pg_dump`/`pg_restore` or platform PITR into an isolated PostgreSQL database.
Do not restore over production. Then set `VLE_RESTORE_DATABASE_URL` to that target:

```bash
python -m src.service.operations --mode restore --confirm-isolated \
  --baseline baseline.json --baseline-sha256 TRUSTED_BASELINE_SHA256 \
  --deployment-id YOUR_DEPLOYMENT --build-revision YOUR_BUILD_SHA > restore.json
```

This read-only verifier compares all governed tables and the audit chain against
the nonempty baseline and rejects the same source/target database. Quiesce writes
while freezing/dumping the baseline; otherwise a valid live backup can legitimately
differ. CI additionally performs a real server-version-matched PostgreSQL dump and
restore into an isolated database, checks append-only triggers, and tests parallel
worker claims. CI evidence validates the software exercise, not your deployment's
backup custody, RPO/RTO, TLS, alerts, retention policy or incident response.

## Runtime binding and operational diagnostics (0.6.2)

New generation jobs record the server-selected provider, model, endpoint, optional
`VLE_MODEL_REVISION` and build revision at submission. The worker compares that
binding before dispatch. Configuration drift or a legacy unbound queued job fails
as `worker_failed` with `reason: runtime_binding_changed_or_missing` and zero model
calls. An explicit investigated retry creates a new job bound to the current
runtime. Secret rotation does not expose keys in the binding. A model revision is
a custodian-supplied identifier, not automatically verified weights provenance.

`GET /v1/workflow-status` reports database-derived status counts, oldest queued
age and stale claims (ID list capped at 100, total count uncapped). Operators see
their own work; reviewer/auditor roles see aggregate authorized work. The default
stale threshold is 300 seconds, configurable with `VLE_WORKFLOW_STALE_AFTER_SECONDS`.
List queries project status fields instead of retrieving complete prompt/result
JSON. Staleness is a diagnostic and never automatically triggers a model retry.
Duplicate configured API tokens are rejected, including when assigned to different
principals, so separation-of-duty configuration cannot appear to contain distinct
credentials while authenticating only one key.

Run Alembic through `0003_workflow_indexes` before this release. It adds owner/time
and kind/time indexes for private listings and pending-job selection. Schema
readiness requires the new revision; no governed records are rewritten.

For PyReason, additionally use the exact-runtime [backend qualification gate](PYREASON_NATIVE_INTEGRATION.md).
Core-only artifacts and legacy reviewed-operation policy cannot authorize it.
