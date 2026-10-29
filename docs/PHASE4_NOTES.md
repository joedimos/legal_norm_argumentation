# Phase 4 implementation notes

## M0: inspected baseline

Baseline: `61b3f785fd6bd879b0bc25b13c3596cfb82c8202` (0.6.2).
Read every module listed in the brief, plus the existing runtime, experiments,
priority, activation, review, reference resolver, tests, packaging and CI.

| Existing API | Extension point / constraint |
| --- | --- |
| `NormInterpretation`, `InterpretationSet` | Immutable typed norms; no strict/defeasible distinction, defeaters, reparations or outer negated modalities. New DDL types wrap this fragment rather than changing it. |
| `candidate_to_interpretations` | Can select a deterministic singleton. Model adapters must explicitly remove that selection. |
| `behaviorally_equivalent` | Compares condition/exception **signatures**, including ordering. Not semantic entailment; retained as a diagnostic, never used alone to establish consensus. |
| `run_experiment(kind='amendment')`, `verify` | Existing bounded implication checks and replayable counterexamples; use for redundancy inclusion. Stronger means fewer satisfying traces, relative to the shared horizon. |
| `OptimizedAAF`, `analyze_preferences` | Existing Dung reduction; extension enumeration limited to 16 arguments. Structured argument construction requires additional budgets. |
| `resolve_pair`, `activation`, `reason` | Reuse scoped authority/specificity/date comparisons and preserve unknown activation. ASPIC+ is a separate explicit semantics, not a replacement for legacy verdicts. |
| `ollama_chat`, `vllm_chat`, `resolve_config` | No existing `ModelProvider` class. Add a protocol/adapter around these bounded transports. Live generation currently generates response drafts, not normative interpretations. |
| `ReviewLedger` | Fingerprint-based human decisions, not a durable review queue. Attach review **prompts** without manufacturing decisions; service routes stay unchanged. |
| `ReferenceResolver`, `verify_candidate` | Supplied-corpus references and source-span/omission diagnostics, not proof of legal textual fidelity. |
| `create_bundle`, `verify_bundle` | Canonical artifact map already supports additive artifacts; retain v1 archive verification and format. |
| `readiness_report` | Keep existing gates/status unchanged; new research evidence belongs in a separate advisory-gates field. |
| `run_benchmarks`, experiment/family suites | Preserve the 62/19/15 baseline corpus bytes and expected results. |

## Semantic decisions and deltas from the brief

- Milestone table governs order: M0, D, A, C, B, E, integration. PRs will
  form an explicit dependency stack; no unverified changes to protected main.
- Redundant samples vary prompts and parameters but do not establish statistical
  independence. Consensus is review triage, never legal correctness or authorization.
- Alignment uses exact names and conservative lexical normalization (case,
  spacing/separators), not speculative stemming/synonym inference. Potential
  many-to-one collisions block analysis. Explicit maps must name their reviewer
  and rationale; these are attestations, not verified identities/signatures.
- Inclusion of compliant behaviors alone collapses permissions into `true` in the
  existing temporal constraint. Redundancy must compare modality/target as well;
  unlike modalities are reported as divergent, never equivalent.
- Assumption diagnosis targets bounded **satisfiability** of a reviewer-supplied
  expected-verdict formula with hard constraints and relaxable assumptions.
  Relaxation allows a completion; it does not establish universal compliance.
  Legal verdict translation is explicit input, not guessed from prose.
- ASPIC+ rationality is conditional: verify finite closure and consistency on
  returned extensions, expose failed transposition/order/axiom preconditions,
  and never claim global postulates solely because one example passes.
- DDL interchange is a documented local versioned dialect, not a claimed
  SPINdle-compatible serialization. Unsupported semantics must be identified.
- All new reports have `decision_authorized=false`, `legal_certification=false`
  and `review_status=unreviewed`; no new review acceptance path is added.
- Neither `src/tda/` nor a Lean toolchain/lake project is present: Track F is
  inapplicable. Optional HTTP is deferred until the core analyses are verified.

## Research grounding verification

Primary arXiv abstracts checked: [R1](https://arxiv.org/abs/2511.09008v2)
and [R2](https://arxiv.org/abs/2506.08899v3). R1's current author list starts
with Chenyang An; its reported >99% soundness is not a result for this toolkit.
R2 v3 is dated 31 December 2025, rather than a verified 2026 revision.
R3/R4 primary pages could not be retrieved in this environment; their brief's
claims are motivations supplied by the requester, not independently reproduced
results. R5's full text/interface has not been established; no implementation
compatibility or reproduced performance is claimed. R6 is identifiable as
Calegari et al., *Defeasible Systems in Legal Reasoning: A Comparative
Assessment*, 2019, DOI [10.3233/FAIA190320](https://doi.org/10.3233/FAIA190320),
with a related author-hosted [2021 overview](https://apice.unibo.it/bin/view/Publication/DefeasibleReasoningJmdpi20).

## Verification protocol

Each milestone runs `python -m pytest -q`, `python -m src.cli benchmark`,
`python -m src.cli experiment-benchmark`, and `python -m src.cli family-benchmark`
in the project environment. SMT differential tests require the existing
`verification` extra. Network/model calls are never used by the tests.
Every bounded search must return unknown on exhaustion without publishing
partial acceptance/minimality as a complete result. Inputs and budgets are
hashed in reports; fixtures remain synthetic and unreviewed.

## M1: structured arguments

`analyze_aspic` constructs finite acyclic propositional arguments, propagates
undermining/rebutting/undercutting to superarguments, and reduces successful
attacks to `OptimizedAAF`. Last-link uses the frontier of final defeasible
rules through strict steps; weakest-link uses all defeasible rules. Both
include ordinary premises and use explicitly recorded elitist set lifting.
Empty fallible sets are firm/strict. Directed contraries are supported.
Undercutters bypass preferences. Cyclic derivation systems return unknown;
they are not approximated by simple paths. Arguments <=256, depth <=32,
construction/attack work <=1,000,000, timeout <=60s; non-grounded extension
search <=16 arguments and within the configured subset budget. Dung calls
are finite, with a deadline check that discards late results.

Rationality is reported as observed checks, not a universal theorem.
Strict-transposition and directly consistent axioms are checked explicitly;
subargument closure, direct consistency and strict closure are checked on
computed extensions. Indirect consistency follows only when the reported
strict closure and direct consistency both hold. No automatic strict
transposition, contraposition, axiom repair or preference repair is performed.

The 512 directed three-node graph fixtures match Dung under all three
semantics via flat defeasible arguments and explicit directed contraries.
The 62-case corpus is preserved: its 18 common-start, no-explicit-order
research cases cross-check the *activation/priority* bridge against `reason`;
other temporal/verification/abstract-preference cases lack this bridge's
signature. Full temporal verdict equivalence is not claimed. In particular,
partially overlapping achievement/prohibition windows may be jointly
satisfiable in temporal research but conflict in single-context reasoning.
The adapter uses legacy exception activation rather than invented exception
arguments; general ASPIC+ inputs can instead name explicit undercutters.

Verification: full suite 298 passed, 3 PostgreSQL-only skipped; five legacy
benchmark runs all green (62/62 twice, 19/19 twice, 15/15). No fixtures changed.
The new `aspic` command uses `--input`, `--lifting last|weakest`, and optional
`--bundle`; every analysis remains unreviewed and unauthorized.

## M2: redundant formalization

The new provider protocol wraps existing transports rather than replacing live
response-generation. Default fake generation requires one extracted norm; a
multi-unit source must be segmented first or supplied as recorded candidates.
Samples vary templates, seeds and temperatures; statistical independence is not
asserted. Provenance binds source and user prompt hashes, model, parameters and
timestamp. Fake timestamps are fixed for reproducibility; live timestamps record
actual generation. Every member is explicitly unselected/unreviewed.

Alignment chooses the most expressive candidate's vocabulary deterministically;
extra atoms there need no invented counterparts in narrower scopes. Unknown
names in other candidates require an explicit reviewer map. Lexical normalization
is deliberately not morphological/synonym inference. Many-to-one collisions
block consensus. Reviewer maps are unverified attestations, not signatures.

Pair inclusion uses the existing amendment checker on a shared achievement
window (1-33 steps), plus a separate normative modal-distinction witness to
prevent O/P collapse. Textual deadlines, unresolved references/assumptions and
calendar scope are outside this fragment and yield unknown. Comparisons are
bounded, not all-time equivalence. Consensus requires zero unaligned atoms,
complete pair checks and at least k equivalent candidates. It never selects a
member or changes review. Budget is per solver call, at most 84 calls for 8
candidates, with each call bounded by the recorded trace/time limits.

`redundancy-report --bundle` retains candidates, alignments, pair checks and
witnesses in additive v1 artifacts; old archive verification is unchanged.
Full suite: 308 passed, 3 skipped, plus the added small-family symmetry test;
new redundancy+codec coverage 91% (every module >=81%). All five legacy
benchmark runs passed. New schema/CLI reports contain no approval path.

## M3: assumptions and minimal corrections

The request carries a reviewer-supplied expected formula, hard formulas, and
0-16 named relaxable assumptions. Kinds distinguish background facts,
grounding, interpretation choice, horizon/closure and Boolean completion;
all kinds need explicit bounded formula encodings. Structural changes such as
changing the integer horizon cannot be silently inferred from these labels.
Source pointers retain quote/offsets but require external document validation.

Exhaustive and optional Z3 oracles enumerate retained subsets, then compute
subset-minimal corrections and relative minimal unsatisfiable subsets. Z3 uses
guarded assumption literals and records its non-minimal unsat core separately;
all SAT witnesses replay through the existing evaluator. MUS is relative to
the hard constraints **including the expected formula**. If even the empty
assumption set is unsatisfiable, report hard_conflict rather than inventing
legal assumptions. That is a conflict of the supplied encoding, not proof of
an irreconcilable statute. An empty correction means expected already feasible.

Budgets cover subset enumeration, trace search/encoding and solver calls.
Correction/MUS cap hits discard partial lists, return unknown/truncated and
make no minimality claim. Rank by cardinality then assumption IDs; solver
witnesses may differ, but the exact correction/MUS sets and ordering agree.
Review prompts are additive queue payloads, never `ReviewLedger.record` calls.

Verification: full suite 317 passed, 3 skipped, plus temporal-operator parity
added in the focused suite (9 passed). Both modules exceed baseline coverage
(95% / 93%). All five legacy benchmark runs passed. No network/model used.

## M4: DDL interchange and two-stage proposals

`DDL/1` is a local textual grammar, not SPINdle syntax. It preserves rule kind,
O/P/F with distinct outer modal negation, ordered compensation sequences,
conditions, exceptions and acyclic superiority. Modalities reuse `Modality`;
DDL-only features use small wrappers. Unknown syntax is rejected, not ignored.
Rule/predicate/compensation counts and text sizes have fixed bounds. There is
no external reasoner, semantic import acceptance or inferred priority.

Typed norms export to defeasible rules. Unsupported calendar/source/deadline/
reference/assumption fields are listed as lossy and retained in a sidecar.
Sidecar restoration verifies it re-exports to the exact DDL text, preserving
original metadata and any historical selection as **data**, not review approval.
DDL without a sidecar is not automatically projected into less expressive
NormInterpretation objects. Local syntactic round-trip is not semantic fidelity.

Segmentation is lexical with complete half-open source spans, not a legal scope
parser. Refinement compares syntax, predicate inventory (unknown when omitted),
exception-presence heuristics and supplied-corpus references. It emits a hashed
diff; nonworsening only grants eligibility for human review, never acceptance.
Metric vectors separately report syntax, modality, inner/outer negation,
exception coverage, normative units (including rule kind, compensation order,
and superiority), and existing source-span/omission diagnostics. Implicit duties
must be explicit expected units; none are invented. No aggregate score is used.

Verification: 24 labelled synthetic/unreviewed DDL fixtures, golden export,
round-trip and schema tests; full suite 353 passed, 3 skipped. Every DDL module
has >=92% statement coverage (96% combined). All five legacy benchmarks passed.
