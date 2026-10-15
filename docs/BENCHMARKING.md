# Benchmarking and Reproducibility

The repository separates evaluation into representation quality and downstream compliance quality.

## Extraction task

Gold records should minimally contain source span, modality, actor, action, condition and exception. Metrics include exact-span precision/recall/F1 plus field accuracy on span-matched candidates. Exact spans are intentionally strict because approximate overlap can hide omitted condition/exception language.

## Formalisation task

Evaluate semantic field agreement, unresolved ambiguity rate, expert selection agreement, behavioural equivalence over a declared finite context suite, cross-reference resolution rate, and stale-review rate after revisions.

## Compliance task

Gold cases pair a formal norm set and fact/action trace with labels such as `compliant`, `non_compliant`, or `indeterminate`. Report accuracy, macro-F1 and per-label precision/recall/F1.

## Joint error propagation

Each benchmark item should record whether upstream ambiguity/error changes the downstream compliance result. `propagation_report` computes the conditional propagation rate. This measures whether representational uncertainty matters behaviorally rather than only syntactically.

## Reproducibility record

A benchmark publication should record:

- source corpus/version and licensing status;
- annotation guidelines and adjudication process;
- train/dev/test split hashes;
- software commit SHA;
- Python/dependency versions;
- optional model identifiers and prompts;
- random seeds where applicable;
- human-review policy;
- evaluation date and applicable normative source versions.

## Certification evidence

Benchmark thresholds may be configured as readiness gates, but the output remains a research evidence report. Passing numerical thresholds does not constitute legal, regulatory, standards, or conformity-assessment certification.
