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
