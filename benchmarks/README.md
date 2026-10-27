# Research regression corpus

`research-v1.json` is a versioned, synthetic 62-case semantic regression corpus.
See `docs/RESEARCH_PHASE_2.md` for the semantics, family counts and limitations.

Run `python -m src.cli benchmark --backend exhaustive` or `--backend z3`.
A non-passing case returns exit status 1. CI saves complete JSON reports for both
backends, including the corpus SHA-256. Changing a label requires a documented
semantic justification and corpus-version change. This is not a held-out dataset
and its passing score is not a claim about legal extraction accuracy.

## Completion and amendment experiments

`experiments-v1.json` adds 19 synthetic cases for exact-horizon partial-evidence
reasoning and directional amendment impact. Run `python -m src.cli
experiment-benchmark --backend exhaustive` or `--backend z3`. These are separate
from the 62 phase-2 fixtures; no existing labels or thresholds are changed.
See `docs/RESEARCH_PHASE_3.md` for the semantic contract and research protocol.

## Interpretation families

`families-v1.json` contains 15 synthetic, unreviewed source-to-model fixtures.
Run `python -m src.cli family-benchmark`. The score tests computational analysis
labels only; every fixture remains `review_required` at top level. No independent
legal annotations, approvals, corpus coverage, or legal-accuracy claims are
implied. See `docs/FAMILY_ANNOTATION_PROTOCOL.md` for independent collection.
