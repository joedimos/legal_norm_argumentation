# Research phase 4 (0.7.0)

This additive research layer supplies **review evidence, not legal decisions**.
All new results explicitly remain unreviewed, `decision_authorized=false` and
`legal_certification=false`. Consensus, satisfiability, argument acceptance and
readiness diagnostics never issue approvals, reviewer signatures or certificates.
Source/grounding review and existing signed-annotation/service controls remain
separate. The production reviewed/advisory policy is unchanged.

## Running and reproducing

Install the dev extra; install the optional verification extra for Z3. No GPU,
live provider or network is needed to run analyses/tests on the fixtures.

```bash
python -m pytest -q
python -m src.cli benchmark --backend exhaustive
python -m src.cli benchmark --backend z3
python -m src.cli experiment-benchmark --backend exhaustive
python -m src.cli experiment-benchmark --backend z3
python -m src.cli family-benchmark
python -m src.cli faithfulness --input benchmarks/phase4-v1.json
```

New commands take versioned JSON inputs and optional `--bundle audit.zip`:

| Command | Input / result contract |
| --- | --- |
| `aspic --input X.json --lifting last` | `vle-aspic/v1` → `vle-aspic-result/v1` |
| `redundant-formalize --source-id ID --text-file F --candidates 3 --provider fake` | Source text → `vle-redundancy-generation/v1` |
| `redundancy-report --input X.json` | `vle-redundancy/v1` or complete generation → `vle-redundancy-report/v1` |
| `assumptions --input X.json --backend exhaustive --max-mcs 32` | `vle-assumptions/v1` → `vle-assumption-report/v1` |
| `ddl --input X.json` | `vle-ddl/v1`, operation export/import/segment/refine/metrics → `vle-ddl-interchange/v1` |
| `faithfulness --input X.json` | `vle-faithfulness-corpus/v1` → `vle-faithfulness-report/v1` |

Exit 0 means analysis completed, including a review-required result, **not**
authorization. Exit 1 means a resource/solver unknown. Invalid inputs return exit
2 with `vle-phase4-error/v1`. Argparse usage errors retain normal CLI behavior.
Bundle digests go to stderr, leaving stdout a schema-valid report. Existing
archives are never overwritten. Schemas are in `schemas/` and tested for every
new command, including bounded failures and untrusted positive-result mutations.

`ollama`/`vllm` adapters reuse `resolve_config` and the existing bounded transports:
`VLE_MODEL_NAME`, `VLE_MODEL_ENDPOINT`, optional `VLE_MODEL_API_KEY`; HTTPS or
loopback HTTP only. Explicit `--provider ollama|vllm --model MODEL` enables
redundant generation. `faithfulness --live --provider ... --model ...` requires a
prepared corpus of at most eight cases. No command automatically discovers a
server, retries a model call, accepts a model-proposed review or contacts a model
in default offline mode. Live gold labels and cached answers are never sent.
Live autoformalization currently accepts typed research norms; model-generated
ASPIC theories/priorities are rejected as unknown. The structured layer itself
has no model-based conflict resolver. Inference outputs, including their proposed facts, are untrusted encodings being
studied, not independently established factual evidence. Full payload hashes,
parameters, source hashes and timestamps permit audit; model names do not pin
weights or establish independent samples. Live paths have been tested with fake
providers only; actual provider compatibility/performance needs deployment tests.

## Structured arguments (ASPIC+ fragment)

Input premises have `id`, `literal`, `kind` (`axiom` or `ordinary`). Rules have
`id`, `antecedents`, `conclusion`, `kind` (`strict` or `defeasible`). Rule/premise
IDs are globally unique; the `premise:` namespace is reserved for internal
attack-site markers. Literals are explicit propositional strings; `~p` is
syntactic negation. Caller-supplied directed `contraries` supplement it.
`~rule:R` undercuts defeasible rule R. Undermining targets ordinary premises;
rebutting targets defeasible conclusions; attacks propagate through all
superarguments. Axioms and strict conclusions cannot be directly rebutted.

Contradictory undermining/rebutting can fail when its target subargument is
strictly preferred. Asymmetric contrary attacks and undercutters bypass that
preference check. Last-link collects the last defeasible-rule frontier through
strict steps; weakest-link collects every defeasible rule. Both include ordinary
premises. The recorded elitist lifting says A is strictly weaker than B when
some fallible item of A is dominated by every fallible item of B. Empty fallible
sets are firm/strict; incomparability preserves attacks. Explicit preference
edges must be acyclic; their transitive closure is used.

Successful attacks form the defeat graph passed to the unchanged Dung engine.
Grounded, preferred and stable extensions are supported. No stable extensions
means no acceptance; the empty family is never vacuous support. Rationality
output separately checks strict transposition, consistency of axioms and their
strict closure, subargument/strict closure and direct/indirect consistency of
computed extensions. These finite observations do not establish a universal
ASPIC+ rationality theorem. No contraposition or axiom repair is invented.

`analyze_norms(norms, FactContext(...))` bridges existing typed norms using
`activation`, `detect_conflicts` and `resolve_pair` (lex superior/specialis/
posterior). Unknown activation suppresses acceptance. The bridge checks legacy
single-context priority semantics, not the temporal-window semantics of a
partially overlapping achievement/prohibition pair. Such temporal constraints
may be jointly satisfiable even when the single-context norms conflict. The
512 three-node tests embed arbitrary directed attacks into flat structured
arguments; they validate the reduction, not legal applicability or every
structured-argument postulate. The 62-case corpus retains its original labels;
only cases within the bridge's actual signature are compared.

## Redundant samples, alignment and inclusion

Generation produces 2–8 explicitly unresolved members of an InterpretationSet.
Templates, seed and temperature vary. Repeated outputs are not counted as
statistically independent evidence; the report states independence is not
established. The default fake uses exactly one extracted norm, never selects
one of several clauses. Recorded FakeProvider inputs can exercise alternatives.

Alignment uses the most expressive candidate vocabulary, breaking ties by ID.
Exact names precede case/Unicode/separator normalization. There is no guessed
synonym, morphological lemma or embedding acceptance. Unmapped names, ambiguous
normalizations and many-to-one collisions block consensus. Explicit maps require
reviewer/rationale attestations; this metadata is not an authenticated signature.

Within a shared, explicit achievement horizon, two implication checks compare
satisfying complete traces using existing exhaustive or Z3 amendment analysis.
`strictly_stronger` means the left candidate permits strictly fewer satisfying
behaviors; the reverse is `strictly_weaker`. Witnesses record trace and horizon.
Modal distinctions are compared separately, so O cannot collapse into P.
Permission-scope and permission-target distinctions are also checked separately:
two duties-free constraints are not automatically equivalent permissions.
Different active permissions are incomparable in the normative-output fragment;
there is no invented ordering of permissive legal strength. Unsupported textual
deadlines, unresolved references/assumptions or calendar applicability yield
unknown rather than extrapolation.

Only fully checked pairwise-equivalence cliques enter classes. All largest
classes are retained on ties. Consensus requires a class of size >=k (default
all samples), no unresolved alignment and complete pair computations. The
implementation is conservatively stricter than accepting k samples while other
pair checks remain unknown. It never selects a candidate or changes a review.
The existing signature-based `behaviorally_equivalent` remains a diagnostic,
not an entailment oracle. Agreement can reflect a shared systematic omission.

## Assumptions, corrections and review prompts

A request supplies an expected-verdict formula, hard formulas, and named soft
assumptions with IDs, kind, source spans and descriptions. Kinds cover background
facts, predicate groundings, interpretation choices, horizon/closure and Boolean
completions. Each requires an explicit bounded formula encoding. A label cannot
silently change the integer horizon, selected interpretation or an observation.

Let H be hard constraints including the expected formula, and A the assumptions.
A correction C is subset-minimal when H with A minus C is satisfiable, but no
strict subset of C restores satisfiability. A relative MUS U is an unsatisfiable
subset of A under H for which every proper subset is satisfiable. Both are
relative to this encoding. Z3 guards soft formulas with assumption literals and
records an unsat core separately; that core is not claimed minimal. Every SAT
witness replays through the independent existing evaluator. Exhaustive and SMT
oracles are different implementations and agree on small differential fixtures.

Relaxation establishes an existential completion, not universal compliance.
No correction when H alone is unsatisfiable is `hard_conflict` in the supplied
encoding, not a judgment about an actual statutory contradiction. An empty
correction means expected already feasible. Ranked prompts retain spans,
descriptions, verdict change and witnesses. They do not write ReviewLedger.
Existing review can bind an exact report using `assumption_review_subject`;
changing any report data invalidates that fingerprint. No report self-approves.

## DDL grammar, stages and completeness

Local DDL/1 lines have this form (example is synthetic/unreviewed):

```text
DDL/1
r: ["risk"] => O("notify") UNLESS ["waiver"] COMPENSATE [["O","repair",false]]
s: [] -> ~P("notify") UNLESS [] COMPENSATE []
superiority s > r
```

Arrows are strict `->`, defeasible `=>`, defeater `~>`. `~P("a")` is outer modal
negation, not `P("~a")`. Compensation order is preserved as a sequence. This
is an interchange representation, not an implementation of compensatory or
negated-modal inference. No SPINdle/external dialect compatibility is asserted.
Unsupported grammar is rejected, never ignored. Superiority must be acyclic.

Typed norms map into defeasible DDL rules. Recommendations and fields whose
meaning cannot be expressed are returned in `lossy_fields`; metadata can be
preserved in a sidecar. Import verifies that a sidecar regenerates the exact
text before restoring data. It cannot turn historical selections into new
review or legal authorization. Plain DDL without such a sidecar is not silently
projected into the less expressive NormInterpretation model. Round-trip tests
establish syntactic preservation, not equivalence to a legal source or reasoner.

Segmentation is a deterministic cue-based proposal with exact Unicode spans.
It cannot resolve exception attachment, implicit duties or cross-provision
scope. Refinement compares syntax, known-predicate consistency, heuristic
exception presence and ReferenceResolver results against a supplied index.
Missing inventories remain unknown. Proposed diffs include before/after hashes;
a nonworsening vector permits human review, never automatic acceptance.

Metric vectors retain syntax, modality accuracy, negation placement, exception
coverage, completeness of normative units (including rule kind, priorities and
compensation order) and existing span/omission source diagnostics. Implicit
obligations must be represented in the expected units or flagged for human
review. No scalar collapses these dimensions. No expected units gives an
undefined completeness value, not perfect completeness.

## Faithfulness and advisory readiness

`benchmarks/phase4-v1.json` contains 12 synthetic/unreviewed fake recordings over
six families. The harness computes symbolic evaluation of each **own** model
formalization and reports direct-model vs symbolic, direct-model vs expected,
and symbolic vs expected disagreement. It retains denominators, abstentions and
all groupings by family, modality and exception presence. Shared unknown answers
are not positive evidence: determinate denominators are shown separately.
The expected labels are not independent expert judgments, and accuracy is not
the headline measure. Corpus version/hash and field-contract hash are archived.

`certification.readiness_report(phase4_evidence=...)` adds separate advisory gates
`redundancy-coverage`, `faithfulness-gap-reported`, `assumption-review-completed`.
They cannot change the existing readiness status, failed gates or deployment
policy. Assumption review must use the existing ledger bound to the full report.
A ledger record is not an authenticated expert signature; existing enrolled
reviewer/signed-annotation mechanisms remain the route for such evidence.

## Explicit resource limits

| Analysis | Bounds and unknown behavior |
| --- | --- |
| ASPIC | <=128 input rules/premises, <=256 arguments, depth<=32, work<=1m, <=60s; cyclic derivation unknown. Preferred/stable <=16 arguments and subset budget. Late results discarded. |
| Redundancy | 2–8 candidates, horizon1–33, <=140 solver calls total; each has <=1m trace cap and <=60s. Record per-call budget; no unbounded equivalence claim. |
| Assumptions | <=16 assumptions, <=32 hard formulas plus expected, <=64 atoms, horizon1–33, <=1m work/trace caps, <=60s, <=256 MCS/MUS each. Cap hit => unknown/truncated, no partial minimality. |
| DDL | <=128 rules, <=256 superiority edges, <=64 predicates/rule, <=8 compensations/rule, <=1MiB text. Segmentation <=32k characters; <=64 references and <=512 declared predicates in refinement. |
| Faithfulness | <=128 cases, <=8 research norms/case, explicit work/time budget, existing AST/argument limits. No partial rates after a bound hit. Live <=8 cases/16 bounded transport calls. |

Input limits reject invalid requests; exhausted computation reports unknown.
Timestamps and solver witnesses can vary in live/SMT runs; canonical ordering,
recorded parameters and deterministic fixture inputs make regression outcomes
reproducible. Timeouts themselves are necessarily machine dependent.

## CI and remaining limitations

Core tests have a network guard and use fixtures/fake providers. A dedicated
Z3-extra job runs differential tests, all legacy suites and coverage checks for
every new module (minimum 80%, the baseline overall level). Dependency installation
and existing supply-chain audit jobs still require package/security feeds;
no analysis or test calls an external legal source or model. Fully air-gapped
CI needs independently provisioned dependency/security artifacts.

No TDA package or Lean project exists, so optional Track F is inapplicable.
Optional HTTP is deferred; no routes or review-state mutations were added.
No legal accuracy, independent annotation, mechanized Python verification,
novelty, certification, deployment throughput or model-quality claim follows
from these synthetic regressions. The paper's original 0.6.2 empirical snapshot
is retained; its limitations now identify Phase 4 as a separate unreviewed
research extension rather than silently replacing the historical results.
