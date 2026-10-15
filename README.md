# Legal Norm Argumentation — VLE/LEGATO End-to-End Research Toolkit

A research implementation for the **full normative-AI lifecycle**: extracting norms from natural language, preserving source meaning and ambiguity, producing typed formal interpretations, reasoning about exceptions and conflicts, evaluating system behaviour for compliance, generating counterexamples, tracking provenance, benchmarking accuracy, and producing evidence-gated readiness reports.

The architecture is aligned with the 2026 VLE-LEGATO agenda: **LEGATO** covers autoformalisation and validation of norms; **VLE** covers integration, normative reasoning, verification, auditing, monitoring and certification evidence.

> This repository is a research toolkit. It does not provide legal advice and does not itself certify legal, ethical, regulatory, or standards compliance.

## Coverage

| Research requirement | Implementation |
|---|---|
| Norm extraction with source traceability | `src/norm_extraction.py` |
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
python -m pip install -e '.[pyreason]'
```

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

## Key modules

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

The argumentation suite exhaustively checks all **512 directed attack graphs on three labelled arguments**. The expanded lifecycle tests cover formalisation, ambiguity, exception/temporal activation, priority, compliance, counterexamples, provenance, benchmark metrics, governance, reference resolution, monitoring and readiness gates.

## Documentation

- `docs/ARCHITECTURE_AND_METHODOLOGY.md` — semantics, fail-closed rules, threat model and limitations.
- `docs/BENCHMARKING.md` — benchmark design and reproducibility requirements.
- `schemas/norm-interpretation.schema.json` — machine-readable interchange schema.

## Important limitations

This is substantially more complete than an extractor/argument graph, but it intentionally does not pretend to solve open problems. The built-in predicate/effect semantics are a reference model, not full temporal deontic logic; authority ranks and specificity must come from validated metadata; cross-reference resolution is only as authoritative as its supplied corpus; and high-stakes conclusions require expert review and external governance.
