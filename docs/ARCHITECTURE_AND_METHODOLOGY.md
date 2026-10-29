# Architecture and Methodology

## Design objective

The system separates the normative AI lifecycle into independently inspectable stages. No stage is allowed to turn uncertainty in an earlier stage into certainty merely because a downstream representation is executable.

```text
normative source
    │
    ▼
extraction + source spans ──────► extraction audit
    │                              │
    ▼                              ▼
interpretation set ◄──────── human review
    │
    ▼
typed formal norm ───────────────► versioned knowledge base
    │                                    │
    ├────────► cross-reference graph ◄───┘
    │
    ▼
activation / exceptions / temporal scope
    │
    ▼
conflict detection ──► priority analysis ──► unresolved conflict gate
    │
    ▼
compliance evaluation / monitoring
    │
    ├────────► bounded counterexamples
    │
    ▼
evidence graph ──► benchmark metrics ──► certification-readiness gates
```

## LEGATO side

- `norm_extraction.py`: source-bound candidate extraction.
- `formalization.py`: candidate → typed `NormInterpretation`, preserving ambiguity as unresolved state.
- `normative_model.py`: modalities, predicates, authority/temporal metadata, provenance and interpretation sets.
- `references.py`: deterministic reference resolution against a caller-supplied corpus.
- `review.py`: semantic-fingerprint human review invalidation.
- `knowledge_base.py`: append-only revision history for normative knowledge.
- `verification.py`: source-span and omission audit.
- `benchmark.py`: extraction field metrics and uncertainty-propagation metrics.

## VLE side

- `defeasible.py`: condition/exception/temporal activation and priority-aware applicability.
- `priority.py`: explicit lex superior / lex specialis / lex posterior decisions; unresolved when metadata is insufficient.
- `compliance.py`: obligation/prohibition/permission evaluation and bounded counterexample generation.
- `monitoring.py`: incremental behavioural trace evaluation.
- `argumentation.py`: Dung AAF semantics and argument graph analysis.
- `pyreason_connector.py`, `pyreason_worker.py`: optional isolated interval-graph
  diagnostics. These advisory annotations never replace Dung or compliance
  semantics, set review state, or authorize decisions.
- `pyreason_normative.py`, `pyreason_program.py`: optional required-when-selected
  norm activation, independently checked before feeding priority/compliance.
  Incomplete computations block instead of falling back to core.
- `certification.py`: evidence gates that explicitly do **not** claim legal certification.

## Joint layer

- `lifecycle.py`: connects formalisation to compliance and constructs a source-to-decision evidence graph.
- `traceability.py`: immutable evidence nodes, typed links, lineage and graph hashes.
- `research.py`, `temporal.py`, `temporal_deontic.py`: reviewed finite-trace research inputs and bounded deontic semantics.
- `model_checking.py`: exhaustive and optional Z3 bounded verification.
- `conflicts.py`, `preferences.py`: scoped conflicts and preference-based defeat.
- `audit_bundle.py`: deterministic, offline-verifiable evidence archives.
- `cli.py`: executable entry point for extraction, audit, formalisation, compliance and lifecycle runs.

## Semantic commitments

### Faithfulness before executability

A representation is not accepted merely because it parses. Every candidate keeps a source identifier, exact span, quote, modality, actor, action, condition, exception, references, ambiguity flags and confidence. Formalisation fails if essential fields are unknown.

### Alternative interpretations

`InterpretationSet` is the formalisation unit. Multiple interpretations may coexist. Selection requires an explicit identifier and rationale. Even a singleton interpretation remains unresolved when ambiguity flags block automatic selection.

### Predicate semantics

The reference reasoner uses a small three-valued boolean predicate model: true, false, or unknown. Compound natural-language clauses remain atomic unless explicitly decomposed; naïve conjunction splitting can change legal scope.

### Defeasibility

A norm is `inactive` when conditions fail or temporal scope excludes it; `undetermined` when relevant facts are unknown; `defeated` when an exception applies or justified priority defeats it; and `active/applicable` otherwise.

### Conflict priority

`lex superior` is used only with explicit comparable authority ranks; `lex specialis` only with represented specificity evidence; `lex posterior` only for equally ranked sources with explicit dates. Otherwise the conflict remains unresolved.

### Compliance

Obligations require the action; prohibitions forbid it; permissions never become obligations merely because they exist; recommendations remain advisory. Unknown applicability and unresolved conflicts propagate as indeterminate outcomes.

## Fail-closed rules

1. Unknown modality cannot be formalised.
2. Missing actor/action cannot be formalised.
3. Unselected interpretation sets cannot drive compliance conclusions.
4. Unknown conditions/exceptions do not become satisfied facts.
5. Conflicts are not resolved without explicit priority evidence.
6. Permissions are not treated as duties.
7. Evidence records are immutable; changes require new version IDs.
8. Readiness reports never claim legal/regulatory certification.

## Threat model

Addressed failure classes include omission, scope distortion, reference loss, temporal error, invented priority, permission/obligation collapse, stale review, evidence mutation and uncertainty laundering.

Not solved by this repository alone: authoritative statutory interpretation, conflict-of-laws analysis, legal enforceability, full temporal/deontic/action logic, proof that a source corpus is complete, host/model/corpus supply-chain security, or regulator-recognised conformity assessment.

## Deployment posture

Use the toolkit as an inspectable research/evidence layer with expert review. Preserve source documents independently, pin corpus versions, treat unresolved states as blocking for high-stakes decisions, and retain external audit logs for operational deployments.

## Phase 2 semantic contract

See [RESEARCH_PHASE_2.md](RESEARCH_PHASE_2.md) for exact temporal boundaries,
preference reduction, bounded solver claims, benchmark protocol and audit-export
trust assumptions. The legacy lifecycle now uses scope inclusion and grounded
defeat semantics; unknown applicability also blocks readiness.


## Automatic IP acquisition (0.8.1)

A separate supervised importer discovers French IP-code articles from a dated
PISTE manifest and U.S. IP opinions through CourtListener search/full-opinion APIs.
Mutable database cursors, conditional leases, provider-wide cooldowns and fenced
atomic batch commits provide restart recovery without mutating source/review
history. Full texts enter the existing extraction/source registry as unreviewed
candidates. This acquisition layer does not classify controlling holdings or
supply independent legal review. See `AUTOMATIC_IP_IMPORTS.md` for interfaces,
coverage boundaries and operational limits.
