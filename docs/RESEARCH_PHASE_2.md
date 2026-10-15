# Research phase 2: temporal reasoning, conflicts, verification and evidence

This phase adds executable, bounded research semantics to the existing service.
It does not establish completeness of a legal corpus, faithfulness of an unreviewed
formalisation, unbounded system correctness, or legal certification.

## Reproduce

```bash
python -m pip install -e '.[dev,verification]'
python -m pytest -q
python -m src.cli research --input examples/temporal-research.json
python -m src.cli verify --input examples/verify-response.json
python -m src.cli benchmark --backend exhaustive
python -m src.cli benchmark --backend z3
python -m src.cli verify --input examples/verify-response.json \
  --bundle audit-bundle.zip --revision "$(git rev-parse HEAD)"
python -m src.cli verify-bundle --input audit-bundle.zip
```

Capture the emitted `bundle_manifest_digest` in a separately trusted record and
supply it with `--expected-manifest-digest` when verifying a retrieved archive.
The CLI refuses to overwrite an existing bundle. Exit codes: 0 for a passing
benchmark, verified bounded property, compliant research trace or valid archive;
1 for a non-passing result (including unknown/unavailable); 2 for invalid input.
`verify` and `research` consume reviewed JSON inputs. Natural-language extraction
does not automatically infer temporal windows or policy preferences.

## Finite-trace temporal logic

`src/temporal.py` implements a typed AST with `true`, `false`, `atom`, `not`,
`and`, `or`, strong `next`, `weak_next`, `eventually`, `always`, and strong `until`.
Traces are nonempty, have at most 128 states, and are indexed from zero. Strong
next is false at the final state; weak next is true there. `p until q` requires
an observed q witness, with p at every earlier state. No infinite suffix is
assumed. Complete Boolean valuations use classical LTLf semantics.

Missing atoms and explicit JSON null represent unknown observations. Evaluation
uses strong-Kleene Boolean connectives: false dominates conjunction and true
dominates disjunction. This is sound but intentionally incomplete relative to
supervaluation: `p or not p` is unknown when p is unknown. The verifier quantifies
over complete Boolean valuations, so that same formula is a bounded tautology.
The AST parser rejects unknown operators/fields, limits depth to 48 and nodes to
512, and never evaluates source code. Atom labels are opaque, case-sensitive
identifiers, not claims of natural-language synonymy.

## Operational deontic fragment

`TemporalNorm` wraps a reviewed `NormInterpretation` with an inclusive `[start,end]`
window, where `0 <= start <= end <= 32`. Steps are discrete observations; they are
not elapsed seconds or calendar days. The caller must justify the clock, sampling
completeness and mapping of textual deadlines into steps.

- Achievement obligation: observe the action at least once inside the window.
- Maintenance obligation: observe it at every step in the window.
- Prohibition: observe no performance at any step in the window.
- Permission: creates no duty; a strong permission/prohibition clash remains a
  normative conflict even when hard action constraints are satisfiable.

Actions are namespaced JSON tuples of subject, action and object; facts use
`fact:<predicate>`. Missing action evidence is unknown, never false. Applicability
is sampled once at `start`: all conditions hold and no exception holds. Later
exceptions do not retroactively cancel an activated duty. Dated authorities need
an explicit activation date. Existing textual `deadline` fields must be reviewed,
translated into a window and cleared before constructing a temporal norm.
Recommendations are not accepted as hard temporal norms.

An early achievement witness is sufficient; an observed maintenance/prohibition
failure is a violation immediately. Before a fully observed deadline, otherwise
unresolved open traces are pending. A closed trace that stops before the window
ends is unknown, not compliant. The original duty remains violated after a missed
deadline; compensatory/reparation and recurring obligations require separate
explicit norms and are not inferred by this fragment.

`constraint()` compiles hard duties to the AST. Use `verify_temporal_norm()` to
ensure the verification horizon includes the entire window. An inactive guard
makes its hard constraint true, so the caller must also consider activation
coverage. The verifier separately rejects globally inconsistent assumptions.

## Structured conflicts and preferences

Conflict records include stable identities, norm IDs, modality pair, subject,
action, object, a satisfying activation assignment, and temporal overlap.
Literal conditions are conjunctive; exceptions are disjunctive (their negations
join the activation conjunction). Contradictory activation assignments are
excluded. Conditions sampled at different activation steps are independent.

Conflicts are detected only for exact structured targets in the same explicitly
stated jurisdiction, or when jurisdiction metadata is absent. Distinct
jurisdictions require a reviewed conflict-of-laws mapping before comparison;
absence of a record is not evidence of no cross-jurisdiction legal conflict.
Effective periods must overlap. A prohibition covering the whole achievement
window makes the duty incompatible. A partial overlap is potential: achievement
may still occur outside the prohibition. Maintenance overlaps are incompatible.
Arbitrary action incompatibilities, resource competition and semantic aliases
are not inferred from text.

The existing lifecycle now uses these scoped conflicts and a grounded defeat
graph. `lex_specialis` requires strict conjunctive scope inclusion, not a count
of conditions. `lex_posterior` requires known equal authority ranks; two unknown
ranks no longer establish equality. Explicit ranks are treated as caller-declared
comparable metadata. Priority defeats can be reinstated: in A defeats B defeats C,
A and C are accepted. Cyclic or unresolved defeats remain undecided and block
readiness. Unknown applicability also blocks readiness. The legacy single-state
compliance evaluator returns unknown for object-scoped or textual-deadline norms
that need structured observations.

`src/preferences.py` accepts a strict, acyclic preference order with an explicit
rationale for every supplied edge. Transitive closure is used. An attack a→b is
blocked exactly when b is strictly preferred to a; incomparability preserves the
attack. Grounded, preferred and stable semantics operate on the resulting defeat
graph. Preferred/stable enumeration is capped at 16 arguments. If no stable
extension exists, the skeptical acceptance set is empty; no vacuous acceptance
is reported. Original conflict records and suppressed attacks remain visible.
This is a specific preference-based Dung reduction, not a claim to implement
all structured argumentation systems or ASPIC+.

`run_research()` composes temporal observations, compatible activation scopes,
conflicts, explicit preferences and grounded acceptance. Its input schema is
`vle-research/v1`; recommendations and unsupported fields fail validation. Unknown
facts, pending windows and unresolved conflicts yield an indeterminate result.
This path is available through Python and the CLI. The existing HTTP lifecycle
uses the scoped single-state reasoner; it does not silently reinterpret ordinary
service requests as temporal research inputs.

## Formal verification backends

`src/model_checking.py` checks `assumptions => property` over **all complete traces
of exactly the declared horizon**. Checking horizon N does not establish validity
at smaller horizons or for infinite traces. It uses either:

1. Exhaustive Boolean trace enumeration, independently evaluated by the AST
   interpreter. Default cap: 65,536 traces; maximum configurable cap: 1,000,000.
2. Optional pinned Z3 backend with a separate Boolean variable for each atom and
   step and an independent witness-based until encoding. Each invocation owns its
   solver context. Reported counterexamples are replayed through the interpreter.

Both backends distinguish `verified_bounded`, `counterexample`,
`inconsistent_assumptions`, and `unknown`. A missing optional solver is
`unavailable`. Invalid input raises an error. Encoding budgets, trace budgets,
solver timeouts and unknown answers never become a proof. Defaults: 5 seconds;
maximum configured timeout: 60 seconds. There are at most 64 assumptions and 64
atoms. Z3 also limits the node-count × horizon² encoding estimate to 250,000.
The bounded result includes atom vocabulary, horizon, backend/version, reason and
replayable counterexample where one exists. An unsatisfiability result is not a
portable proof certificate; independent re-execution is the current evidence.

## Benchmark protocol

`benchmarks/research-v1.json` contains 62 synthetic, explicitly labelled cases in
six families: temporal boundaries (12), partial observations (10), bounded
verification (12), temporal deontic behavior (14), structured conflicts (6), and
preference argumentation (8). Labels are authored from the documented semantics,
not generated by the implementation under test. The corpus is a regression suite,
not a held-out legal benchmark; scores must not be presented as real-world legal
accuracy. Reports include corpus SHA-256, version, backend, per-case results and
family totals. CI runs both backends and retains both JSON reports.

Tests additionally compare independent backends on 198 formula/horizon/assumption
combinations, replay counterexamples, enumerate deontic traces, verify reinstatement,
exercise invalid inputs and test evidence tampering. Existing extraction/service
and exhaustive three-argument graph tests remain enabled.

## Audit bundles and service integration

`src/audit_bundle.py` creates deterministic ZIP_STORED archives with canonical
UTF-8 JSON, fixed timestamps/modes, sorted members, and a versioned manifest with
SHA-256 and length for every artifact. It rejects non-JSON values, NaN, duplicate
JSON keys, duplicate ZIP members, unlisted files, paths, symlinks, encryption,
compression and oversized inputs. It never extracts files. Limits: 31 artifacts,
16 MiB for archive and uncompressed contents, JSON depth 64.

`GET /v1/runs/{run_id}/audit-bundle` requires the auditor role. It includes the
source text/hash, candidates/audit, original request/context, formalizations,
reasoning/lineage/readiness, original authorization result, review and selection
identities/fingerprints captured at evaluation time, runtime package versions,
and the run's audit events. Exporting a blocked high-stakes run does not authorize
it. Later approvals do not alter the historical evidence snapshot. Historical
runs without snapshots are explicitly marked incomplete. Resource audit events
are a subset of the global chain, not a standalone proof of chain completeness.

Set `VLE_BUILD_REVISION` to the deployed full Git SHA. Missing revision is reported
explicitly (`revision_recorded: false`); code provenance is then incomplete.
Runtime data is captured when evaluating, not reconstructed from the export host.
Exports are marked `Cache-Control: no-store` and expose an archive ETag plus
`X-Audit-Manifest-SHA256`. The archive contains sensitive source/review information;
handle it under the same access and retention policy as the underlying records.

Offline verification checks structural integrity. A separately trusted digest
binds the archive to an external record. An internal manifest hash alone cannot
prove origin; no digital signature, external timestamp, legal certification or
independent audit conclusion is claimed. Retain the manifest digest in the
organisation's signed receipt or protected audit system.

## CI and main protection

Develop on a branch based on the inspected main commit and submit a PR. Do not
push feature changes to main. The `production-ci` workflow retains both Python
versions, lint/security, PostgreSQL integration and container/Trivy/SBOM gates;
the added `research-verification` job installs the solver extra, runs research
regressions and exports both benchmark reports. No security threshold is relaxed.
A PR must remain unmerged while any required check is failed, pending or absent.
Repository administrators should require these checks in their branch rule;
workflow configuration alone does not create a GitHub branch-protection rule.

## Primary references

- De Giacomo and Vardi, *Linear Temporal Logic and Linear Dynamic Logic on Finite
  Traces*, IJCAI 2013: https://www.ijcai.org/Proceedings/13/Abstracts/132.html
- Dung, *On the acceptability of arguments and its fundamental role in nonmonotonic
  reasoning, logic programming and n-person games*, Artificial Intelligence 77
  (1995), 321–357: https://doi.org/10.1016/0004-3702(94)00041-X
- Z3 Python Solver API (`check`, `model`, `reason_unknown`, timeout parameters):
  https://z3prover.github.io/api/html/classz3py_1_1_solver.html
- Z3 programming guide: https://z3prover.github.io/papers/programmingz3.html

The operational deontic fragment and conflict policy above are explicit project
choices. They should not be attributed to a theorem from those references.
