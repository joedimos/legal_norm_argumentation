# Native PyReason integration, 0.9.3

PyReason is a required computational backend when selected. It derives relational
and temporal facts, computes norm activation, and supplies those activations to
the existing priority, conflict and compliance engines. A separate finite oracle
checks the supported fragment. A disagreement blocks the computation; the oracle
never substitutes its answer for a failed native call.

## Interfaces

Install `pip install -e '.[pyreason]'`. The extra pins PyReason 3.8.0 and its
numeric/JIT runtime on Python 3.12/3.13. No GPU, model server or model call is used.

```sh
python -m src.cli pyreason-infer --input examples/pyreason-graph.json --worker-mode pooled --bundle graph.zip
python -m src.pyreason_qualification --iterations 128 --timeout-seconds 900 --revision FULL_GIT_SHA
```

The standalone graph command reports unreviewed intervals, exact Boolean
bindings, input/program hashes, symbol maps, bounds and oracle agreement. Both
standalone commands keep authorization and legal certification false. `norm-evaluate`
accepts optional `pyreason_program` and `worker_mode` fields. The Python
`evaluate` and `run_lifecycle` APIs accept `pyreason_program` and
`pyreason_worker_mode`. The authenticated evaluate/release-target API accepts
`pyreason_program`; worker policy comes from server configuration. Core requests
with a graph program are rejected rather than silently ignoring it.

Graph bindings feed actual norm conditions/exceptions. In the shipped synthetic
example, a licence record at tick 0 establishes an effective licence at tick 1,
then a licensed relation at the same tick. No licence is assumed before this or
after expiry. An exact licence binding defeats the relevant prohibition; absence
of a licence observation otherwise leaves applicability unknown. Conflicting
asserted and derived facts block the entire native result.

## Supported finite fragment

`vle-pyreason-graph/v1` declares entities, unary/binary predicates, time-bounded
facts, range-restricted variable-only rules, constant head intervals, nonnegative
delays, ground queries and optional Boolean fact bindings. Normalized `n0`/`p0`
symbols enter the engine; supplied identifiers never enter executable rule text.
The schema is `schemas/pyreason-graph.schema.json`; Python also checks arities,
variable safety, uniqueness and bounds.

Missing facts have interval `[0,1]`. A rule body cannot use `[0,1]` as evidence.
Soft evidence cannot produce a crisp Boolean head. Only `[1,1]` and `[0,0]` can
bind ordinary norm facts; other intervals remain explicit uncertain bindings.
Intervals express information bounds, not calibrated probabilities. Conflicting
interval constraints produce unknown, not an invented compromise.

Facts are inclusive on start/end ticks. Frames do not persist automatically.
Zero-delay rules saturate the current frame; positive delays schedule heads in a
later frame. Relational joins can infer new edges. Seeded positive cycles reach a
finite fixed point; unseeded cycles establish nothing. Ticks are model indices,
not inferred calendar dates: a reviewer must confirm their mapping to source
provisions and the evaluation date.

Limits are 16 entities, 32 predicates, 256 facts, 32 rules, 128 queries, horizon 8,
8,192 groundings and 10,000 interval updates. Native execution has an aggregate
1–300 second budget, a two-megabyte output cap and one worker slot per service
process. Validation, oracle checks, inference and activation share the time
budget. Timeout, saturation, missing output or mismatch yields unknown with no
positive activations or projected facts.

This is a deliberately bounded part of PyReason's language. Arbitrary source
rules, custom annotation functions, neural integrations, unrestricted quantifiers
and interval-to-probability conversions are not exposed. Deontic modalities,
legal priorities and unresolved conflicts remain in the toolkit's typed symbolic
layer. Native computation alone does not prove source faithfulness.

## Worker and deployment

Isolated execution remains the default for compatibility. `pooled` reuses one
private child process, serializes access, resets native facts/rules/settings on
each request, and recycles after 128 requests, 30 idle seconds, a bad receipt or
transport failure. Requests are never replayed automatically. A timed-out child
is killed. The pool is qualified on Linux; use isolated execution on other
platforms. Each process has its own worker; capacity exhaustion fails closed.

```sh
docker compose -f docker-compose.yml -f docker-compose.rc.yml -f docker-compose.pyreason.yml up --build -d
```

The native target installs the actual engine, initializes and exercises caches
at build time, runs as UID/GID 10001, and uses a writable named cache volume with
a read-only root filesystem. The default image remains core-only. The native
overlay forces enablement, pooled transport and explicit startup warmup; readiness
stays 503 until the native probe and graph inference complete. Startup is bounded
by 300 seconds. CPU/thread, memory and PID limits are explicit. Use a new cache
volume after changing the pinned native runtime or image. Caches are performance
artifacts, not evidence or trusted rules.

The native CI image check disables container networking and runs the real
qualification corpus under these restrictions. Separate Python 3.12/3.13 jobs
exercise native inference, state isolation, the API, and mixed 1/8/32/64-norm
performance. Deterministic stand-ins remain in the default offline unit jobs.

For representative staging load, use the existing authenticated smoke/load tool:

```sh
python -m src.service.operations --mode load --base-url https://YOUR_STAGING_ORIGIN --deployment-id YOUR_ID --build-revision FULL_GIT_SHA --with-evaluation --reasoning-backend pyreason --worker-mode pooled --concurrency 1 --requests 100
```

Configure the existing probe credentials separately. This is synthetic staging
traffic, not independent legal data. Load counts only computed native responses,
checks the chosen transport, and retains errors. Single-process saturation is
intentional; qualify the intended number of service processes and concurrency
under the actual deployment load rather than extrapolating these small tests.

## Release qualification

There is no longer an unconditional core-only production block. The native gate
requires strict `ip-supervised-v1` policy and ALL existing organizational release,
source-scope, approved interpretation and exact-case legal-review gates. The
legacy reviewed-operation override cannot authorize PyReason.

A separate externally pinned `vle-pyreason-release/v1` manifest binds the exact
build, deployment, worker policy, release-manifest hash, expiry and at least two
custodian approvers to a native qualification artifact. Native evidence must
match installed engine-source/dependency/Python/adapter fingerprints and the
shipped corpus; all checks and at least 32 mixed-workload measurements must
complete. See `schemas/pyreason-release.schema.json`.

The existing production IP benchmark must additionally identify
`reasoning_backend: pyreason` and `backend_target_sha256`. Both independent
annotation signatures (and adjudication, if present) must bind the canonical hash
of this object:

```json
{"schema":"vle-pyreason-qualification-target/v1","build_revision":"FULL_GIT_SHA","worker_mode":"pooled","runtime_sha256":"IDENTITY_HASH","native_sha256":"NATIVE_REPORT_HASH","predictions_sha256":"IP_PREDICTIONS_HASH","packet_sha256":"HELD_OUT_PACKET_HASH"}
```

The production load artifact must name `reasoning_backend: pyreason`, the exact
`worker_mode` and `backend_computations_checked: true`, in addition to existing
persisted evaluation/audit/SLO requirements. Configure the manifest with
`VLE_PYREASON_RELEASE_PATH` and a separately supplied canonical
`VLE_PYREASON_RELEASE_SHA256`. Read-only release-status/readiness routes expose
the gate. No status route invokes the native engine, a source provider or a model.

A content hash establishes integrity, not the truth of reported experiments or
expert credentials. Custodians remain responsible for artifact acquisition and
independence checks. Synthetic test keys and fabricated test attestations only
test gate mechanics. They are not shipped release approvals. Real independent
held-out French/U.S. IP judgments are not supplied by this integration. Without
those and actual deployment evidence, high-stakes native authorization remains
blocked for a stated evidence reason. Computational results always remain
uncertified; only the existing service policy can authorize an exact case after
all gates pass.

## Evidence and limitations

`PYREASON_NATIVE_PROTOCOL.md` specifies the engineering checks before the mixed
latency analysis. `benchmarks/pyreason-native-v1.json` contains 162 synthetic
activation contexts and graph programs for expiry/delay, chaining, relational
joins, soft/exact intervals and seeded/unseeded cycles. Native checks compare to
an independent finite oracle. Mutation tests reject corrupted, incomplete,
late and soft-as-Boolean outputs; API tests preserve review gates and exact-target
revocation. Audit ZIPs retain the entire program, observations and computation;
old bundles still verify offline. Integrity verification is not an independent
proof of native execution or legal accuracy.

Recorded native reports report startup separately, preserve every latency, and
state whether native is slower. No speedup or legal-accuracy claim follows from
these synthetic workloads. The core evaluator remains substantially faster on
small flat norms; PyReason supplies relational/temporal inference and a second
computational implementation. Full deployment-scale performance and independent
legal evaluation remain external release requirements.
