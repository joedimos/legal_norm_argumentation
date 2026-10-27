# Research phase 3: incomplete evidence and amendment impact

This extension supplies executable experiments for two open research directions:
uncertainty-aware normative reasoning and semantic change analysis. Its algorithms
use established Boolean completion, implication and bounded verification ideas.
The integration is a research platform; originality or superiority to published
methods requires a literature review and independent empirical evaluation.

## Reproduce

```bash
python -m pip install -e '.[dev,verification]'
python -m pytest -q tests/test_experiments.py
python -m src.cli experiment --input examples/uncertainty-experiment.json
python -m src.cli experiment --input examples/amendment-experiment.json
python -m src.cli experiment-benchmark --backend exhaustive
python -m src.cli experiment-benchmark --backend z3
python -m src.cli experiment --input examples/amendment-experiment.json \
  --bundle amendment-evidence.zip --revision "$(git rev-parse HEAD)"
python -m src.cli verify-bundle --input amendment-evidence.zip
```

Both example experiments deliberately exit 1: one is contingent and the other
finds a strict strengthening. For `experiment`, exit 0 means `definitely_true` or
`equivalent_bounded`; exit 1 includes all other results, including definite falsity,
change, unknown, unavailable, and inconsistent evidence. Exit 2 means invalid
input. The benchmark command exits 0 only when every expected label matches.
All existing service authorization, review and certification gates remain in place.
These experiments run through Python/CLI, not through a new HTTP endpoint.

## Shared model: completions at an exact finite horizon

A request uses `schema: vle-experiment/v1`, a `kind`, and a positive `horizon`.
The property, before/after interpretations and optional assumptions use the
existing temporal AST. Optional `observations` is a nonempty array of partial
states, no longer than the horizon. Missing values, explicit nulls and unobserved
suffix steps range over both Boolean values. Explicit true and false are fixed.
They are constraints on possible completions, not defeasible claims.

Let C be the complete traces of exactly horizon h that agree with observed cells
and satisfy all assumptions. The vocabulary is the union of formula, assumption
and observation atoms. The exhaustive backend enumerates only unfixed cells;
Z3 constrains the corresponding Boolean variables. SMT witnesses are replayed
through the independent interpreter, including every fixed observation.

No variable is assigned a probability. No open-ended future, infinite trace,
calendar-time interpretation or corpus completeness is inferred. The existing
strong-Kleene evaluator is unchanged. Its sound but sometimes incomplete answer
is shown as a baseline, without assumptions, when observations are supplied.

Every verification result records `observation_constraints` as
`(step, atom, Boolean value)` tuples and the number of `unknown_cells`. A proof
conditional on observed evidence must not be presented as a proof over all traces.
The exact request's canonical SHA-256 binds observations, formulas, assumptions,
backend and budgets. This is a request identity, not a semantic equivalence hash.

## Uncertainty experiment

Supply `kind: uncertainty` and `property`.

| Result | Meaning |
| --- | --- |
| `definitely_true` | C is nonempty and every completion satisfies the property |
| `definitely_false` | C is nonempty and every completion violates it |
| `contingent` | A satisfying and a violating completion both exist |
| `inconsistent_evidence` | No completion satisfies the assumptions and observations |
| `unknown` | A resource/encoding/solver limit prevented the required conclusion |
| `unavailable` | The requested optional backend is not installed |

For example, `p OR NOT p` is definitely true over Boolean completions even when
p is unobserved. It remains unknown in the existing strong-Kleene evaluator.
This difference is intentional, not a change to the legacy runtime semantics.

Contingency returns both witnesses. `witness_differences` identifies cells on which
these particular witnesses differ. They are useful candidates for expert questions
or additional observations; they are **not** a minimal observation set, a causal
explanation, or a guarantee that observing one cell resolves the property.
Unknown/unavailable outcomes preserve any witness obtained in a completed check.

## Amendment experiment

Supply `kind: amendment`, `before`, and `after`. Both formulas operate over the
same assumptions, observations and horizon. They represent conjunctions of hard
constraints, or any other explicitly justified Boolean behavior criterion.
The experiment checks `before => after` and `after => before` independently.

| Result | Relation between satisfying behavior sets within C |
| --- | --- |
| `equivalent_bounded` | The sets are equal |
| `strict_strengthening` | After is a strict subset of before |
| `strict_weakening` | Before is a strict subset of after |
| `incomparable` | Each permits a behavior the other rejects |
| `inconsistent_evidence` | C is empty, so no meaningful impact conclusion is drawn |
| `unknown` / `unavailable` | The required checks did not establish a relation |

`removed_behavior` satisfies before and violates after. `added_behavior` does the
reverse. These names describe modeled admissibility, not a legal permission.
An empty after-language can be a strict strengthening; that label alone is not a
claim that the amended policy is feasible or desirable. A separate feasibility
analysis is required. Permissions, competing normative reasons and authority
relations are not reconstructed from hard-constraint trace inclusion. Use the
existing structured conflict/preference tools when those distinctions matter.

The supplied waiver example changes `waiver OR notify` to `notify`. It exposes a
removed behavior where a waiver holds and notification does not occur. This is
a synthetic model choice, not a statement about a French legal provision.

## Bounds, evidence and reproducibility

Existing formula depth/node, 128-step horizon, 64-atom, 64-assumption, enumeration,
encoding and solver-time limits apply. A known observation reduces the exhaustive
completion space. The timeout and trace cap apply **per check**, with at most two
checks per experiment; the experiment is not a single shared timeout budget.
Wrapped negations/implications must themselves fit existing formula bounds.
Invalid requests fail explicitly. Duplicate JSON request keys are rejected.
No truncated search is reported as a proof or an equivalence.

`--bundle` reuses the deterministic audit format with request, result, and runtime
revision/package evidence. Exports refuse overwrite. Preserve the manifest digest
in a separately trusted system; integrity alone does not authenticate a source or
certify compliance. Solver witnesses may differ across solver versions, so compare
semantic verdicts and replay evidence rather than requiring identical witnesses.

The versioned `benchmarks/experiments-v1.json` contains 19 hand-labelled synthetic
cases. Reports contain every request-bound result, corpus digest, backend, expected
and actual verdict, and pass counts. They complement the existing 62-case research
corpus. They are regression fixtures, not held-out evidence of legal accuracy.

Tests independently enumerate all ternary partial valuations of two atoms over
one and two steps for nine formulas (810 combinations), check complete traces as
an oracle, compare backends on partial temporal observations, replay amendment
witnesses, check non-vacuity, and verify CLI evidence exports and failure behavior.

## Research agenda and falsifiable study designs

| Direction | Research question and hypothesis to test | Protocol and measures |
| --- | --- | --- |
| Evidence-sensitive autoformalization | Can completion analysis reduce unnecessary expert deferrals while preserving soundness relative to a reviewed finite model? | Compare strong-Kleene and completion verdicts on independently annotated missingness patterns; report decisiveness, disagreements, latency and resource-limit rate separately |
| Semantic amendment regression | Do trace-inclusion witnesses expose omissions that syntax/schema checks miss? | Expert-curated amendments and omission mutations; blinded witness review; report detection precision/recall, false alarms and interpretation disagreement |
| Assumption drift | How often is apparent equivalence caused by restrictive assumptions or evidence? | Repeat each pair under baseline, weakened assumptions and revised observations; retain all request hashes; report relation transitions and vacuous-context rates |
| Expert evidence acquisition | Can witness-guided questions reduce review effort? | Compare witness-cell suggestions against random and expert-chosen queries; measure questions/time to decision and unresolved cases; do not assume the current suggestions are optimal |

Before making publication claims, preregister the intended semantics, source
rights, annotation protocol, train/development/held-out separation, ambiguity
adjudication and stopping rules. Stratify by language, modality, conditions,
exceptions and temporal scope. Keep expert disagreement as data; do not label one
plausible legal interpretation as ground truth merely because a solver accepts it.

## Primary technical references

- De Giacomo and Vardi, *Linear Temporal Logic and Linear Dynamic Logic on Finite
  Traces*, IJCAI 2013: https://www.ijcai.org/Proceedings/13/Abstracts/132.html
- Z3 project programming guide, *Programming Z3*:
  https://z3prover.github.io/papers/programmingz3.html
- Z3 solver API: https://z3prover.github.io/api/html/classz3py_1_1_solver.html

This finite-completion experiment is a declared project semantics. It must not be
conflated with unbounded three-valued monitoring of infinite-trace LTL.
