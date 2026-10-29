# Required PyReason norm activation (0.9.2)

The `pyreason` normative backend participates in computation: its activation results
feed the existing priority/Dung reasoner and compliance evaluator. Selecting it
makes completion mandatory. A core-only result cannot silently substitute for a
missing, timed-out, partial or disagreeing engine result.

## Executable path

Install `pip install -e '.[pyreason]'` (Python 3.12/3.13), then run:

```sh
python -m src.cli norm-evaluate --input examples/pyreason-normative.json --bundle normative.zip
```

The synthetic, unreviewed example deliberately reports a violation (exit 1).
Setting the `licensed` fact to true produces a compliant computation; removing it
leaves an indeterminate computation. No expert or legal validation is implied.
Exit 0 denotes a compliant computation, 1 denotes non-compliant/indeterminate,
and 2 rejects malformed requests. Bundles include the request and full result.

The Python APIs are `compliance.evaluate(norms, context, backend="pyreason",
pyreason_timeout_seconds=60)` and `defeasible.reason` with the same keywords.
`run_lifecycle(..., reasoning_backend="pyreason", pyreason_timeout_seconds=60)`
requires approved, unambiguous source formalizations before invoking the engine.
The CLI `comply` and `lifecycle` commands accept `--reasoning-backend pyreason`;
raw `norm-evaluate`/`comply` computations are unreviewed research outputs.

For the authenticated service, install the extra **in its runtime environment**,
set `VLE_PYREASON_ENABLED=true`, and request:

```json
{"reasoning_backend":"pyreason","reasoning_timeout_seconds":60,"facts":{},"observed_actions":[],"high_stakes":false}
```

The console exposes the same selection. A disabled deployment rejects the request
with 503. An enabled deployment with an unavailable engine persists an
indeterminate result; it does not fall back to core. The default container image
contains the core dependencies only; setting the flag does not install packages
or make that image an engine-qualified deployment.

## Bounded semantics

Compilation covers the existing point-context activation fragment: actor scope,
effective dates, conjunctions of conditions, disjunctions of exceptions, and
negated predicates. Each predicate evaluation is represented by explicit Boolean
true/false/unknown indicator facts. Unknown indicators route to an undetermined
activation, never to known permission, obligation fulfilment or compliance.
All rule bodies use exact Boolean flags; confidence thresholds are not used.

The compiler preserves the reference evaluator's order: ineffective scope,
unknown evaluation date, actor mismatch, false conditions, unknown conditions,
applicable exceptions, unknown exceptions, then activation. Eight outcome flags
are generated with safe node markers so one norm's rules cannot activate another
norm. The grammar accepts no source-supplied executable rule text.

PyReason computes these flags. Exactly one outcome must be obtained for every
norm. A separate reference activation check must match before the computed
activations enter priority resolution. A mismatch discards *all* activations,
records the differing norm IDs/statuses, and blocks the computation. This check
is not a fallback. The existing priority/Dung layer resolves legal conflicts;
the existing compliance layer preserves O/P/F/recommendation distinctions.
Object-specific observations and natural-language deadlines still return unknown;
this backend does not add LTL, compensatory obligations or new legal semantics.

Bounds: 64 norms, 32 predicates per condition/exception list, 4,096 rules,
16,384 indicator facts, 65,536 body clauses, one inclusive engine timestep,
1–300 seconds including initialization, 2 MB worker receipt, and one concurrent
native worker per service process. Hitting any bound yields unknown. There is no
unbounded queue or automatic retry. CPU/RAM quotas remain deployment settings.
Independent Uvicorn processes each have their own worker slot.

## Review, release and audit

The result carries the compiled program, node map, source semantic hashes,
context hash, worker input hash, annotations, bounds and independent-check status.
Lifecycle evidence adds explicit formalization → backend → compliance edges.
Normal persisted-run bundles retain these artifacts and remain offline-verifiable
for integrity. Hash integrity does not independently prove legal interpretation.
Legacy core idempotency and signed-target bytes are preserved; selecting a backend
or changing its timeout changes the evaluation target/request basis.

All new outputs keep `decision_authorized=false` and `legal_certification=false`.
The existing release attestations qualify the core backend only. They cannot be
reused to authorize this new backend, including under the legacy reviewed-operation
policy. High-stakes authorization stays blocked until a separately specified
release-qualification process covers this backend. Computational compliance and
organizational authorization remain distinct outputs.

The legacy `OptimizedAAF.run_pyreason()` graph-annotation path remains available
with its established advisory semantics. This active backend is a separate,
versioned interpretation of typed norms; legacy graph labels are never promoted
into normative activations.

## Validation and limits

CI uses a deterministic synthetic provider with the upstream-shaped Fact/Rule
interface. Tests exhaust all three-valued combinations of two conditions and two
exceptions (with negation), actor/date scope and all four modalities; they compare
against the independent core. Tests inject wrong/partial/soft outcomes, unavailable
workers, timeouts, resource saturation and malformed programs. HTTP tests check
review gates, persistence, independent authorization and offline audit exports.
Installed-wheel checks exercise the actual child process and normative CLI outside
a checkout with the labelled stand-in.

Three real PyReason 3.8.0 synthetic IP-shaped cases separately check non-compliance,
exception-based compliance and missing-exception indeterminacy. A further five-norm
context checks mixed active, undetermined, defeated and inactive outputs without
cross-norm rule leakage. The recorded requests/results are in
`PYREASON_NORMATIVE_QUALIFICATION.json`; command results and source hashes are in
`RC_0_9_2_EVALUATION.json`. These small cases
are implementation checks, not independent legal annotation or full engine
qualification. No legal accuracy or certification is established by this work.
