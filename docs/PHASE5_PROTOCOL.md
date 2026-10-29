# Phase 5 pre-analysis protocol v1

Committed before any Phase 5 analysis code. Baseline 0.7.0; client date 28 October
2026. This is a repository-frozen protocol, not an external study registration.
All generated fixtures carry synthetic=true and review_status=unreviewed.
We hypothesize the effects below; negative, null and unavailable outcomes are
retained. No adaptive threshold, fixture or exclusion changes after results.
Any amendment needs a dated justification and must retain this original text.

## Common design and stopping rule

Use seeds 0–7 for finite synthetic suites, seed 17 for random-formula differential
checks and seed 23 for sampled geometry. Publish input/output hashes, caps,
denominators, unknown counts and failures. Default limits:16 cells/points,
12 questions,4096 worlds,65536 work states and5000 ms; overrides recorded and
hard-capped. Bound exhaustion reports unknown without partial exact claims.
Run full pytest and fixed legacy62/19/15 suites after each milestone. Stop each
experiment at the frozen corpus size/cap; never generate extra cases to seek a
positive effect. Real expert/model inference is excluded from CI.

## N4 / H4: independent certificate checker

Use all existing temporal fixtures plus128 seed 17 generated formulas at horizons
1–3, including missing observations. Independently compare Kleene truth values.
Certificates cover explicit finite formulas/assumptions, enumeration counts and
Merkle roots, trace violations, minimal unsatisfiable families and bound-only
unknowns. Checker imports only the standard library; total checker package <=500
logical source lines. No proof that Python implements the stated mathematics.

Mutants: unknown treated true, off-by-one horizon, and swapped priority/query
binding. Test tampered witnesses, omitted assumptions, changed formulas/horizons,
roots, counts and context hashes. Report per-mutant attempts, certificates,
rejections, escapes and no-certificate cases. Desired detection among produced
invalid certificates=100%; report shortfalls rather than suppress them. A valid
certificate for the wrong encoding only detects the error if a separately
trusted expected-query digest is supplied; report that trust-boundary escape.

## N1 / F1–F3: local/global consistency

Planted suites: equality/inequality triangle and4-cycle, labelled synthetic
exception-scope loop and reference loop; eight renamings/seeds per structure.
These labels describe formal encodings, not extracted legal evidence. Compare
pair SAT, the existing typed pairwise detector where defined, and exact global
SAT. Require the expected inclusion-minimal obstructing family and check every
single deletion (monotonicity entails all proper subfamilies satisfiable).
Record a nerve graph cycle only when one exists.

F2: include acyclic join-tree relations with agreeing projections and the fixed
acyclic chain counterexample to pair-SAT sufficiency. Run full semijoin reduction
as the shortcut, verify the join-tree running-intersection property, replay any
witness and differential-check against global enumeration. Success criterion:
zero contradictory answers on complete computations. Report the unqualified
pair-SAT claim as false if the counterexample is reproduced.

F3: inventory existing frozen Code de la propriété intellectuelle and Code de
commerce LivreIV snapshots/typed norms. Include only actual reviewed evidence
or explicitly identified assumed-reviewed demo encodings. No automatic reviewer
creation. Report eligible documents/norms, cyclic structures and obstructions,
including zero if actually evaluated. Missing inputs => not_run with counts null.
No network scraping or unreviewed prose is relabelled as an empirical study.

Persistence: supplied effective-date/version snapshots only; track identical
minimal norm-ID families across ordered snapshots as intervals, retaining left/
right censoring. This is obstruction lifetime, not homology of source meaning.

## N2 / H2: metamorphic autoformalization

Freeze a finite parallel EN/FR corpus before running the harness. Relations:
mechanical voice change, conditional reorder, exception synonyms, modality
synonyms, double negation, enumerated-condition reorder, supplied EN/FR pairs;
directional modality weakening, exception insertion, condition removal and
polarity reversal. Applicability is explicit. Unsupported transformations are
not silently included as passing. Composition depth cap3; breadth/work bounded.

Run deterministic extraction and fake recorded outputs with planted missing
exceptions/modality collapse. Detection target100% of applicable injected faults;
report escapes and inconclusive runs. For equivalence, use shared-atom aligned
bounded semantic comparison plus modality/target evidence. Directional duty
strength uses behavior inclusion and separately records Phase 4 modal divergence.
MVR denominator: applicable, complete comparisons; report unknown/inapplicable
counts separately. No malformed/truncated generation counts as a passing MR.
Radius is the minimum tested failing depth; no failure => right-censored at the
maximum exhaustively explored depth, never infinite robustness. Compare exception
position/EN↔FR vs lexical synonyms descriptively; no significance claim from tiny
fixtures. The hypothesized ordering may fail and must be written up.

## N5 / H5: review planning simulation

Finite noiseless questions with explicit answer partitions of caller-supplied
interpretation worlds. World verdicts must be recomputed from formulas/traces;
caller-supplied truth labels alone cannot authorize a terminal verdict. Costs are
positive integers, defaults1 per question and explicitly assumptions. Minimax
objective minimizes worst-case **cost**, not necessarily number of questions.
Never invent priors or auto-answer a review question.

Freeze eight finite Boolean-world tasks (including irrelevant queries, unequal
costs, insufficient questions and no-gain cases), with random orders from seeds
0–7. Compare adaptive minimax with the supplied fixed order and each seeded random
order, stopping as soon as surviving verdicts agree. Report per-task and per-world
cost/question distributions, worst-case values, ties, losses, unknowns and
unresolvable cases. H5 question-count superiority is tested on equal-cost tasks
only; unequal-cost tasks assess the actual objective separately. Simulated
reviewers are deterministic world oracles, not people. No default probability
weights, expected costs or personal reviewer attributes are inferred.

## N3 / H3: behavioral geometry and simulated annotation pipeline

Exact distance uses one shared uniform finite Boolean universe over aligned
atoms, shared assumptions and horizon. Empty feasible universe => unknown.
Use shared samples for every pair, seed 23, report95% simultaneous Hoeffding
intervals with union bound over pairs. Sampled distances are estimates, not exact
proofs. Exact metric axioms tested on behavioral-equivalence classes.

Compute small Vietoris–Rips H0/H1 over F2, including triangles that kill1-cycles;
max16 points, max2-skeleton, max65536 reduction operations. Infinite bars are
right-censored at filtration end. Largest finite H0 lifetime is the predeclared
cluster separation predictor (not count of alternatives or legal schools).

Pipeline demonstration: eight synthetic family/study units, geometry features
(largest finite H0 lifetime, alternative count, mean pairwise distance), and
simulated complete annotation submissions processed by existing annotation-assess.
No credentials/signatures are fabricated; identities explicitly denote simulated
non-expert oracles, and outer records declare zero real reviewers. Primary outcome:
unit-level disagreement in the faithfulness dimension. Prediction comparison:
leave-one-unit-out ordinary least squares with intercept, mean-squared error;
constant predictors use intercept only. Compare H0 MSE against both baselines:
strictly smaller than each supports the simulation hypothesis, equality or worse
is a null/negative result. Threshold0 (no tolerance tuned post hoc); no p-values.
Labels alternate by frozen unit index and are independent of geometry by design,
so no real predictive validity is implied even by a positive simulated result.
