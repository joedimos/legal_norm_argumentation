# Legal Norm Argumentation — VLE/LEGATO End-to-End Research Toolkit

A research implementation for the **full normative-AI lifecycle**: extracting norms from natural language, preserving source meaning and ambiguity, producing typed formal interpretations, reasoning about exceptions and conflicts, evaluating system behaviour for compliance, generating counterexamples, tracking provenance, benchmarking accuracy, and producing evidence-gated readiness reports.

The architecture is aligned with the 2026 VLE-LEGATO agenda: **LEGATO** covers autoformalisation and validation of norms; **VLE** covers integration, normative reasoning, verification, auditing, monitoring and certification evidence.

> This repository is a research toolkit. It does not provide legal advice and does not itself certify legal, ethical, regulatory, or standards compliance.

## Coverage

| Research requirement | Implementation |
|---|---|
| Norm extraction with source traceability | `src/norm_extraction.py` |
| French obligation/prohibition/permission extraction | bilingual rules in `src/norm_extraction.py` |
| Official French-law acquisition | `src/legifrance.py`, `src/service/legifrance_ingestion.py` |
| Web operations console | `src/service/web_ui.py`, `src/service/app_with_ui.py` |
| Conditions, exceptions, cross-references | extraction + `src/references.py` |
| Typed formal representation | `src/normative_model.py` |
| Alternative interpretations / ambiguity | `InterpretationSet`, `src/formalization.py` |
| Semantic/behavioural equivalence testing | `behaviorally_equivalent` |
| Human-in-the-loop validation | `src/review.py` |
| Versioning / normative knowledge governance | `src/knowledge_base.py` |
| Source-faithfulness auditing | `src/verification.py` |
| Dung argumentation semantics | `src/argumentation.py` |
| Exceptions and defeasible activation | `src/defeasible.py` |
| Lex superior / specialis / posterior | `src/priority.py` |
| Compliance evaluation | `src/compliance.py` |
| Counterexample generation | `generate_counterexamples` |
| Continuous compliance monitoring | `src/monitoring.py` |
| Source→decision evidence lineage | `src/traceability.py`, `src/lifecycle.py` |
| Extraction/compliance benchmark metrics | `src/benchmark.py` |
| Upstream uncertainty propagation metric | `propagation_report` |
| Certification-readiness evidence gates | `src/certification.py` |
| Optional probabilistic reasoning backend | `src/pyreason_connector.py` |
| Bounded finite-trace temporal/deontic semantics | `src/temporal.py`, `src/temporal_deontic.py` |
| Structured target/scope/window conflicts | `src/conflicts.py` |
| Explicit preference-aware argumentation | `src/preferences.py`, `src/research.py` |
| Exhaustive and optional Z3 verification | `src/model_checking.py` |
| Versioned 62-case semantic benchmark | `benchmarks/research-v1.json` |
| Deterministic auditor-only evidence exports | `src/audit_bundle.py`, `/v1/runs/{run_id}/audit-bundle` |
| Reproducible CI | `.github/workflows/ci.yml` |

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev]'
```

Optional NER model:

```bash
python -m spacy download en_core_web_sm
```

Optional integrations:

```bash
python -m pip install -e '.[transformers]'
python -m pip install -e '.[verification]'
```

## Web console and French-law ingestion

The production service includes a self-contained web console at `/` and a bounded public-source collector for two official Légifrance corpora:

- **Code de la propriété intellectuelle** — French intellectual-property law.
- **Code de commerce, Livre IV** — French pricing and competition law.

Start the application:

```bash
vle-legato-api
# or
uvicorn src.service.app_with_ui:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/`. The console accepts an API key for the current browser tab only, lets an operator collect bounded article batches, and exposes the resulting source versions, candidate norms, provenance and review state.

The collector accepts no arbitrary URL. It follows only validated HTTPS links under `www.legifrance.gouv.fr/codes/`, records source provenance in the extraction audit, creates a new source version when official text changes, and records both ingestion and unchanged refresh checks in the existing audit chain.

For unattended refreshes:

```bash
# One bounded page from each corpus
vle-legato-maintenance --sync-legifrance all --limit 50

# Traverse every discovered article in bounded pages
vle-legato-maintenance --sync-legifrance all --limit 50 --full
```

See [`docs/LEGIFRANCE_INGESTION.md`](docs/LEGIFRANCE_INGESTION.md) for API examples, deployment controls, full-sync bounds and operational caveats.

## CLI

```bash
python -m src.cli extract --source-id policy-1 \
  --text "If risk is high, the operator must notify unless notification is legally barred."

python -m src.cli audit --source-id policy-1 --text-file policy.txt
python -m src.cli formalize --source-id policy-1 --text-file policy.txt

python -m src.cli comply --source-id policy-1 --text-file policy.txt \
  --facts '{"risk is high": true, "notification is legally barred": false}' \
  --actions '["notify"]'

python -m src.cli lifecycle --source-id policy-1 --text-file policy.txt \
  --facts '{}' --actions '[]'
```

`--assume-reviewed` exists only for demonstrations/tests. Real workflows should use actual review records.

## Core semantic rule: uncertainty is preserved

The system deliberately fails closed. An ambiguous candidate is not silently selected; an unknown condition does not become true; an unresolved normative conflict does not become compliance; and a passing research-readiness report never becomes a claim of legal certification.

## Architecture

```text
authoritative/manual normative source
    │
    ├──── Légifrance bounded acquisition + provenance/version hash
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

## Key modules

- `legifrance.py`: fixed-corpus public HTML discovery and bounded article retrieval from Légifrance.
- `service/legifrance_ingestion.py`: source hashing, version persistence, provenance and audit events.
- `service/legifrance_routes.py`: authenticated collection and source-registry API plus the console route.
- `service/web_ui.py`: dependency-free web operations console.
- `normative_model.py`: modalities, predicates, authority/temporal metadata, provenance and interpretation sets.
- `formalization.py`: candidate → typed interpretation; alternative interpretations require explicit selection.
- `priority.py`: explicit `lex superior`, `lex specialis`, `lex posterior`; unresolved when metadata is insufficient.
- `defeasible.py`: conditions, exceptions, temporal applicability and priority defeat.
- `compliance.py`: obligation/prohibition/permission evaluation and bounded counterexamples.
- `monitoring.py`: incremental event-trace compliance evaluation.
- `traceability.py`: immutable evidence nodes/edges and lineage.
- `benchmark.py`: extraction, compliance and error-propagation metrics.
- `certification.py`: research-readiness gates that explicitly do **not** claim legal certification.
- `lifecycle.py`: end-to-end LEGATO→VLE orchestration.

## Tests

```bash
python -m pytest -q
```

The argumentation suite exhaustively checks all **512 directed attack graphs on three labelled arguments**. The expanded lifecycle tests cover formalisation, ambiguity, exception/temporal activation, priority, compliance, counterexamples, provenance, benchmark metrics, governance, reference resolution, monitoring and readiness gates. Légifrance tests use deterministic HTML/network fixtures so CI does not depend on external availability while still testing URL restrictions, parsing, French extraction, versioning and authenticated ingestion.

## Documentation

- `docs/ARCHITECTURE_AND_METHODOLOGY.md` — semantics, fail-closed rules, threat model and limitations.
- `docs/BENCHMARKING.md` — benchmark design and reproducibility requirements.
- `docs/LEGIFRANCE_INGESTION.md` — French-law acquisition, web console, API and unattended refresh operations.
- `schemas/norm-interpretation.schema.json` — machine-readable interchange schema.

## Research phase 2

```bash
python -m pip install -e '.[dev,verification]'
python -m src.cli research --input examples/temporal-research.json
python -m src.cli verify --input examples/verify-response.json
python -m src.cli benchmark --backend exhaustive
python -m src.cli benchmark --backend z3
```

See [the phase 2 semantic contract and audit workflow](docs/RESEARCH_PHASE_2.md).
The existing service uses the scoped single-state lifecycle. Reviewed temporal
inputs run through the Python/CLI research path. The auditor endpoint exports
historical source, request, reasoning, review and runtime evidence as a deterministic
archive with offline integrity verification.

## Research phase 3: uncertainty and amendment impact

The framework now supports bounded reasoning over incomplete observations and
behavioral comparison of alternative or amended formal interpretations:

```bash
python -m src.cli experiment --input examples/uncertainty-experiment.json
python -m src.cli experiment --input examples/amendment-experiment.json
python -m src.cli experiment-benchmark --backend exhaustive
python -m src.cli experiment-benchmark --backend z3
```

Uncertainty analysis quantifies over Boolean completions at an explicit horizon,
returning definite truth/falsity or two witnesses establishing contingency.
Amendment analysis reports equivalence, strict strengthening, strict weakening,
or incomparable changes, with added/removed behavior witnesses. Inconsistent
evidence and incomplete computation never count as a positive result. The
existing Kleene evaluator, HTTP lifecycle and authorization gates retain their
semantics. `verify` also accepts partial `observations`, records those constraints,
and enumerates only unknown cells.

The new 19-case synthetic suite supplements the existing 62 research fixtures.
See [the phase 3 semantics and research protocol](docs/RESEARCH_PHASE_3.md) for
reproduction commands, audit exports, limitations, and four falsifiable research
directions. These capabilities enable research; novelty and legal accuracy are
not established by passing synthetic tests.

## Evidence acquisition research

```bash
python -m src.cli plan-evidence --input examples/evidence-plan.json
```

The bounded evidence planner computes an adaptive strategy with minimum worst-case
observation cost over all feasible Boolean completions. It returns a decision DAG,
or opposite-verdict witnesses proving that the available queries cannot resolve
the result. Enumeration/state/time limits produce `unknown` without a partial
optimality claim. Costs and query availability are explicit assumptions.

See [the evidence-planning contract and study protocol](docs/EVIDENCE_PLANNING.md)
for bounds, audit exports, the weighted adaptive example, and independent tests.

## Interpretation families and statutory alignment

```bash
python -m src.cli family --input examples/interpretation-family.json
python -m src.cli family-benchmark
```

The next research milestone distinguishes factual uncertainty from formal
interpretation disagreement. It preserves source/version/quote provenance,
requires content-bound family and grounding reviews for a reviewed result,
retains infeasible alternatives as blockers, and computes bounded factual-query
plans across interpretation worlds. Opposite verdicts under identical complete
facts yield a review witness instead of an impossible factual clarification plan.

The 15 new fixtures are synthetic and explicitly unreviewed. Read the
[semantic and governance contract](docs/INTERPRETATION_FAMILIES.md) and
[independent annotation protocol](docs/FAMILY_ANNOTATION_PROTOCOL.md). The [bounded response workflow](docs/RESPONSE_ALIGNMENT.md) additionally checks
provided candidate revisions against separately reviewed hypothetical behavior
mappings. `response:` actions cannot be promoted to observations or factual
queries. It stops on stale mappings, interpretive/grounding disagreement,
repeated text and revision limits. Run:

```bash
python -m src.cli statutory-revise --input examples/statutory-revision.json
```

The controller makes no live model calls. Authenticated expert review and
production authorization remain outside this CLI research path.

## Important limitations

This is substantially more complete than an extractor/argument graph, but it intentionally does not pretend to solve open problems. The temporal extension is an explicit finite-trace operational fragment with bounded verification, not full temporal deontic logic or unbounded certification; authority ranks and specificity must come from validated metadata; cross-reference resolution is only as authoritative as its supplied corpus; public Légifrance HTML is an external interface whose structure can change; and high-stakes conclusions require expert review and external governance.

## Live generation and independent annotation

`live-generate` calls an explicitly configured Ollama model and returns an
unreviewed answer plus behavior-mapping proposals. It pauses for human review
after every generated candidate; only a reviewed violation permits a revision.
`annotation-prepare` freezes blinded study packets; `annotation-assess` validates
two separate reviewer submissions, reports per-dimension agreement and preserves
third-reviewer adjudication and unresolved interpretations.

See [the setup and annotation guide](docs/LIVE_GENERATION_ANNOTATION.md). No actual
independent expert annotations or live provider evaluation results are shipped.
