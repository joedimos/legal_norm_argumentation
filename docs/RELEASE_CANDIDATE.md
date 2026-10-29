# Release-candidate preparation: 0.9.0

This release prepares a supported, human-supervised French/U.S. intellectual-
property research and decision-support platform. It is not a 1.0 release or an
attestation that any deployment is production ready. Live provider access,
independent legal annotations and deployment qualification remain external
deliverables. Synthetic regressions cannot satisfy them.

## Supported scope and compatibility

The reference service supports Python 3.12/3.13, PostgreSQL 18, Alembic forward
migrations, HTTPS ingress, separate operator/reviewer/auditor identities and
explicitly configured Ollama/vLLM inference. French acquisition covers current
Code de la propriété intellectuelle articles through production PISTE. U.S.
acquisition covers CourtListener full opinions found by the four declared IP
queries. Neither provider establishes a complete applicable-law corpus.

The `/v1/` source/review/evaluation APIs and existing schemas remain compatible.
The original `legacy` release profile retains its reviewed/strict policies and
`vle-release-evidence/v1` graph, including IP/competition source requirements.
Historical audit bundles remain verifiable. No governed migration is rewritten.
New formats are explicitly versioned. Existing command names/arguments remain
valid; default benchmarks now resolve the shipped corpus independently of CWD.
Explicit input paths retain ordinary caller-relative meaning and never silently
fall back to bundled data.

Phase 4/5 diagnostics, DDL interchange, ASPIC+, temporal research, obstruction,
geometry and review planning remain bounded experimental interfaces. They do not
automatically become service verdicts or human approvals. Their negative and
unavailable results remain part of the release record. Lean mechanization and
positive results for the research hypotheses are not release requirements.

For 1.0, freeze supported `/v1/` request/response semantics and schema identifiers;
add fields compatibly, introduce a new version for incompatible meanings and
document deprecation and migration before removal. CLI research schemas retain
their experimental status until individually promoted. Performance limits and
abstention are part of the semantic contract, not implementation accidents.

## Strict IP deployment profile

Use `deployment.rc.env.example` as a checklist, inject real secrets through your
deployment platform and provide an exact build SHA and custodian-verified model
revision. A model revision string is not automatically verified weights evidence.

```bash
docker compose --env-file deployment.rc.env -f docker-compose.yml -f docker-compose.rc.yml config --quiet
docker compose --env-file deployment.rc.env -f docker-compose.yml -f docker-compose.rc.yml up --build -d
docker compose --env-file deployment.rc.env -f docker-compose.yml -f docker-compose.rc.yml exec app vle-legato-release-check
```

The overlay forces `ip-supervised-v1`, strict enforcement and both production IP
providers for app, worker and importer. Environment interpolation cannot weaken
those overlay values. The settings validator rejects a downgraded policy,
missing model/build binding, disabled acquisition or sandbox PISTE. Ordinary
Compose remains backward compatible; use the overlay explicitly for this profile.

This profile accepts only `vle-release-evidence/v2`, with the exact deployed build,
scope `fr-us-ip-supervised/v1`, model revision, expiry, two actual custodian IDs
and canonical hash-pinned artifacts. It requires eight entries: benchmark,
annotations, model, smoke, load, restore, ip_sources and review_protocol. The
machine-readable formats are in `schemas/*v2.schema.json`,
`ip-source-evidence.schema.json` and `ip-review-protocol.schema.json`.

The held-out benchmark must bind the signed annotation packet, prediction-file
hash and review-protocol hash. Every case must remain in every dimension's
denominator, including abstentions. Dimensions are language (fr/en), modality
(obligation/prohibition/permission), exception presence and source kind
(code_article/judicial_opinion). Each stratum reports count, correct, abstentions,
false compliance and extraction F1. Accounting must agree across dimensions;
every stratum must meet the configured quality floors. All three direct/symbolic/
gold disagreement counts are required. The default false-compliance budget is
zero observed cases per stratum; an authorized deployment can set an explicit
budget with `VLE_RELEASE_MAX_FALSE_COMPLIANCE_RATE`. Zero observed errors does
not imply a zero population error rate. Pre-register sample sizes, budgets and
uncertainty intervals before evaluation; retain complete predictions and scripts.

The gate validates a pinned organizational attestation and arithmetic, not the
truth of the experiment, the legal correctness of its gold labels or the
authenticity of a person's credentials. Custodians must verify those independently.
The existing minimum of 100 held-out units is a provisional policy floor, not a
guarantee of sufficient power in each language or IP topic. The existing annotation
packet bound is 200 units; larger studies need a separately versioned aggregation
protocol rather than silently exceeding or weakening the bound.

The judicial review protocol explicitly covers holdings/dicta, quoted reasoning,
majority/separate opinions, precedential status, jurisdiction/date, conditions,
exceptions and references, changes to sources and independent adjudication. Its
instruction document is hash-bound. Import records mark judicial authority as
unassessed. No lexical classifier supplies controlling authority. Exact-case
signed applicability/faithfulness/grounding review remains mandatory. Unsupported
document kinds, missing upstream provenance and jurisdiction mismatches block
authorization in this scoped profile. Cross-jurisdiction application and complete
precedent/treatment analysis are unsupported and require a different reviewed model.

## Execute live and operational qualification

Supply actual PISTE production credentials and a CourtListener token. Start the
importer and observe successful current batches for all five recent jobs. Its
status distinguishes liveness, successful batches and completed query cycles.
Historical backfill can remain ongoing only if the supported corpus boundary is
declared; none of these statuses establishes corpus completeness.

```bash
vle-legato-importer --status
python -m src.service.operations --mode ip-sources --deployment-id YOUR_DEPLOYMENT --build-revision YOUR_BUILD_SHA > ip_sources.json
```

The source probe reads at most one manifest/page and one full text per declared
topic, recording five content/upstream-payload receipts. It performs no source
imports or reviews. Every topic has its configured run deadline; the combined
bound is five times that deadline. Empty, excluded or unsupported first results
produce failure, not a substitute snippet or successful completeness claim.
Evidence must be recent (default 172800 seconds); future timestamps fail.

Use the existing `smoke`, persisted-evaluation `load`, `snapshot` and isolated
`restore` probes from PRODUCTION_RELEASE.md on your actual staging deployment.
Exercise restart recovery, credential rotation, stale leases, 429 cooldown,
source amendments and alert delivery. Set service targets for your workload;
the default 100-request/p95-1000ms probe is only a baseline. Verify TLS, database
access, encrypted backup custody, audit-head anchoring and incident recovery.
Do not publish provider credentials or submit synthetic/expert-lookalike evidence.

## Readiness reporting

`vle-legato-release-check` and authenticated read-only
`GET /v1/release-readiness` inspect configuration, shipped assets, migration
readiness, five current import jobs and the pinned external evidence graph.
They make no provider/model calls and never mutate a review or create audit
events. Missing tables, stale or future stamps, scope/configuration drift,
import failures and unavailable evidence remain blocking. The CLI returns 1
for blocked prerequisites and sanitizes configuration/connection failures.

`prerequisites_met` means these deployment prerequisites were supplied, not that
a case is authorized. Every result has `decision_authorized=false` and
`legal_certification=false`; exact-case review and the existing lifecycle gates
still apply. These diagnostics do not turn research evidence into certification.
No real deployment, model evaluation or independent legal study was performed
as part of the 0.9.0 preparation.

## Distribution and upgrade qualification

Wheels and source distributions now contain versioned schemas, examples,
benchmarks and migration scripts. CI rebuilds the wheel from the sdist, compares
its bytes under a fixed SOURCE_DATE_EPOCH and checks resource hashes against
the checkout. The installed module must resolve inside a fresh environment;
benchmarks and migrations execute from an unrelated directory with network access
blocked. CI retains artifacts, checksums and the distribution-check report.
This is reproducibility for a fixed build environment, not a cross-platform
bit-for-bit promise or a signed provenance attestation.

For an installed wheel, outside a checkout:

```bash
vle-legato-migrate --check
vle-legato-migrate
vle-legato-migrate --check
vle-legato benchmark
vle-legato-release-check
```

The migration entry point supports inspection and upgrade-to-head only; it has
no downgrade command. Existing `alembic upgrade head` workflows still work.
Back up and run an isolated restore before promotion. Roll back application
traffic only to a build compatible with the migrated schema; recover governed
data forward rather than deleting evidence. Version 0.9.0 adds no DB revision.

## Maintainer actions before a 1.0 RC tag

1. Enforce main-branch review and the required CI checks through a GitHub ruleset
   or branch protection; disable force pushes/deletions and document bypass
   custody. Current repository access cannot administer those settings. A passing
   workflow is not branch protection. No protection change is claimed here.
2. Choose and publish the project's license/copyright terms. No license has been
   inferred or assigned on the author's behalf. Confirm source/provider usage,
   retention, access and redaction policies with the responsible custodian.
3. Identify actual maintainers, vulnerability-reporting channel, supported release
   lifetime and incident owner. Do not treat a generated file as an SLA.
4. Obtain the live data/model, independent expert and real staging evidence above;
   resolve adverse findings rather than lowering a threshold after seeing results.
5. Freeze the supported contracts, rerun CI for the exact RC SHA, retain the wheel,
   sdist, image digest, checksums/SBOM and evidence graph, and have the custodians
   approve promotion. Publishing/tagging/deployment is a separate release action.

Automated acquisition remains unattended after setup. Human interpretation,
independent legal judgment, real credentials and organizational release approval
are deliberately not fabricated or auto-completed.
