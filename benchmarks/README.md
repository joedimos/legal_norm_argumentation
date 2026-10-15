# Research regression corpus

`research-v1.json` is a versioned, synthetic 62-case semantic regression corpus.
See `docs/RESEARCH_PHASE_2.md` for the semantics, family counts and limitations.

Run `python -m src.cli benchmark --backend exhaustive` or `--backend z3`.
A non-passing case returns exit status 1. CI saves complete JSON reports for both
backends, including the corpus SHA-256. Changing a label requires a documented
semantic justification and corpus-version change. This is not a held-out dataset
and its passing score is not a claim about legal extraction accuracy.
