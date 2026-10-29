# Phase 5: research hypotheses and bounded diagnostics (0.8.0)

We hypothesize that local/global consistency, metamorphic testing, behavioral
geometry, checkable verdict artifacts and review planning can improve the
inspection of legal formalizations. The repository does not establish priority
of publication, legal accuracy or certification. The literature search, exact
protocol, mathematical deltas and a dated implementation addendum are retained
in PHASE5_NOTES.md and PHASE5_PROTOCOL*.md. Existing models and all legacy
benchmark labels remain authoritative; no reviewed fixture was changed.

All analyses remain `review_status=unreviewed`, `decision_authorized=false` and
`legal_certification=false`. Mathematical or model agreement never approves an
interpretation, factual grounding, source mapping, priority or reviewer. No
production authorization, release policy or existing HTTP route changes here.

## Results, including unavailable and negative outcomes

| Protocol item | Observed result | Scope |
| --- | --- | --- |
| F1 |32/32 planted minimal cyclic families detected; each pair SAT | Boolean relational encodings; typed pairwise detector inapplicable when no typed norm exists |
| F2 | Unqualified pair-SAT plus acyclicity is false | Fixed p, p iff q, not q counterexample; corrected full-semijoin shortcut agrees on tested systems |
| F3 | Not run; eligible counts and obstruction counts null | No eligible local reviewed/ingested Légifrance corpus; not an empirical zero |
| H2 baseline |6 violations/14 complete comparisons;2 inconclusive | Frozen0.7.0 extractor; exception-position ordering not established |
| H2 treatment |3/14 complete comparisons violated;2 inconclusive | Token-boundary repair on identical fixtures;2/2 planted fake faults detected |
| H3 | Not supported in the simulation | H0 LOOCV MSE0.20565; alternative-count0.46113; mean-distance 0.17836. H0 fails to beat both baselines |
| H4 | Wrong-model escape without an external intended-query pin | Artifact tampering: unpinned1/3, pinned0/3. Runtime mutations:2 unknown diagnostics,1 affirmative wrong-model certificate,1 no-certificate; the affirmative escape is rejected by a trusted pin |
| H5 | Improvement in some tasks, ties and insufficiency retained | Eight finite-oracle tasks; among5 equal-cost resolvable fixed-order comparisons,1 strict question-count improvement and4 ties |

The complete per-case/per-world vectors, unknown counts and seeded random-order
distributions are in `benchmarks/phase5-*-results.json`. There are no real model
inference measurements, legal study participants or authenticated expert labels.
The geometry pipeline labels identify SIMULATED_NONEXPERT oracles, not people;
independence flags there are simulation inputs, not verified credentials.

## Reproduction and interfaces

Install the existing dev/verification extras. `geometry` is an empty optional
install group because the bounded topology implementation has no extra library;
the calculation still needs explicit `--topology`/`topology=true`. No TDA or Lean
project is present. No mechanized Python correctness claim is made.

```bash
python -m pytest -q
python -m src.cli benchmark --backend exhaustive
python -m src.cli benchmark --backend z3
python -m src.cli experiment-benchmark --backend exhaustive
python -m src.cli experiment-benchmark --backend z3
python -m src.cli family-benchmark
python -m src.cli faithfulness
python -m src.certificate_study
python -m src.obstruction_study
python -m src.review_simulation
python -m src.geometry_study
python -m src.cli metamorphic --input benchmarks/metamorphic-v1.json --provider extractor
```

| Command | Versioned input | Result |
| --- | --- | --- |
| `certify --input Q.json` | Explicit property/horizon/assumptions/observations/binding | Finite-model certificate and independent replay |
| `check-certificate --input C.json` | `vle-certificate/v1` | `vle-check-result/v1` |
| `obstruction --input X.json --backend exhaustive` | `vle-obstruction/v1` or timeline wrapper | Obstruction or censored lifetime report |
| `metamorphic --input X.json --provider extractor` | `vle-metamorphic-corpus/v1` | Per-relation/modality/language vectors and radius diagnostics |
| `geometry --input X.json --horizon 1 --exact --topology` | Existing family or `vle-geometry/v1` | Exact quotient metric and optional F2 Rips bars |
| `geometry --input X.json --sample 128 --seed 23` | Same | Shared-sample estimates and simultaneous intervals |
| `plan-review --input X.json` | `vle-review-plan/v1` | Suggested decision DAG or opposite-verdict witnesses |

Use spaces between option names and values (`--horizon 1`, `--sample 128`,
`--seed 23`). Optional `--bundle FILE.zip` creates a new deterministic archive;
existing paths are never overwritten. Exit 0 means a completed analysis, not
approval; exit 1 means unknown or failed certificate checking; invalid requests
return exit 2 with an unauthorized error envelope. Bounds/unsupported fragments
are preserved. JSON schemas and CLI tests include unknown/error paths.

## N4: proof-carrying finite-model queries

The independent `src/checker/` implementation uses only the standard library and
is under 500 lines. It does not import the main reasoner or reuse its evaluator.
It implements explicit strong-Kleene finite-trace semantics and independently
replays counterexamples, observations and assumptions. Complete-trace witnesses
must address the exact horizon and all query atoms. Unknown cannot be converted
to true through Python truthiness. Learning a missing fact may legitimately
resolve unknown; the invariant is no unsupported promotion.

`violation` carries a concrete admissible trace making the property false.
`compliant_within_bound` carries a state count, admissible count and a
canonical-enumeration Merkle root (domain-separated leaf/internal hashes).
The checker independently re-enumerates all completions and requires nonempty
admissibility, avoiding vacuous compliance. `conflict` carries an unsatisfiable
family plus a witness for every single deletion, and an optional declared graph
cycle. `unknown` carries only a bound/unsupported-fragment diagnostic; checking
its syntax is not checking an affirmative verdict.

The query digest binds property, exact horizon, assumptions, observations and
source/priority/model binding metadata. Those metadata are not proved legally
correct. A producer must not call its own computed digest an independently
trusted intended model. Supply `--expected-query-sha256 DIGEST` separately for
that trust boundary. Actual reasoner mutations that force positive unknowns or
shorten horizons are downgraded to unknown; a consistently mistranslated priority
model can still yield a valid certificate for the wrong query. The published
escape rate explicitly retains that case and the no-certificate case.

Installed Z3 exposes proof ASTs (locally inspected with a proof-enabled context).
Neither those ASTs nor unsat cores are claimed checked by this independent
checker. Refutation/compliance proof production therefore falls back to bounded
enumeration, even when Z3 supplied the reasoner's diagnosis. Larger solver-only
results remain unknown when the independent replay cannot finish in scope.

Audit v1 manifests remain unchanged. `verify-bundle` recursively checks embedded
certificates as well as standalone artifacts; identical certificates are cached.
A 262144-cell materialization cap bounds enumeration memory. A shared 5000 ms certificate-checking deadline and 128 unique-certificate cap bound
this extra work. The old archive size/member/nesting guards still apply. Optional
repeated `--expected-certificate-query-sha256 DIGEST` arguments restrict the exact
set of intended queries. Self-contained hashes are not signatures or expertise.

Certificates cover the supported explicit finite-query layer, not every legacy
legal/compliance/priority algorithm. Unsupported encodings are diagnosed instead
of receiving a pretend proof. Plan optimality and legal-source faithfulness are
not independently certified by these verdict artifacts.

## N1: admissible sections and global consistency

The carrier is the ordinary assignment sheaf: assignments restrict by projection
on overlapping atoms/time cells. Each formal norm selects an admissible subset
of its carrier. A jointly compatible selection of one admissible local row per
cell glues by union; the obstruction is **absence of any such jointly compatible
selection**, not inability to glue an already compatible assignment family.
Different satisfiable pairs may require different local choices.

Cells use explicit finite formulas or the existing typed temporal translation.
Scope/target comparisons reuse existing functions; references resolve only
against a supplied index; unresolved references block the analysis. Nerve faces record common atoms or shared compatible
target/scope metadata. A graph cycle may bound a filled simplex and is not an H1
cohomology class. The set-valued constraint analysis is not an abelian sheaf
cohomology calculation. No optional linearized sheaf package exists here.

GYO acyclicity alone does not justify inference from pair SAT. A verified
running-intersection join tree performs full relational semijoins; a reconstructed
witness must replay, and default differential checks compare the global solver.
Global satisfiability is an existential formal-model result, never blanket legal
compliance. Existing normative modal conflicts remain separate and cannot be
laundered into compliance just because hard behavior constraints are satisfiable.

Minimal unsatisfiable families are searched in increasing cardinality. Every
single deletion is witnessed, which implies all proper subfamilies satisfiable
by monotonicity. A genuine higher-order conflict with no graph cycle receives the
explicit `higher_order_conflict` result. At most one minimal family is discovered
per report. Timelines track caller-selected families across strictly ordered
version/date snapshots; unknown continuity and boundary censoring remain visible.
This is tracked-family lifetime, not complete discovery or persistent homology.

Only explicit assumed-reviewed demo encodings are currently analyzed through
this research adapter; the flag does not manufacture approval. F3 needs actual
eligible frozen legal snapshots and existing review evidence. No raw statutory
prose or generated constraints are automatically relabelled as reviewed evidence.

## N2: metamorphic relations and treatment separation

Transformations operate on a small declared grammar or supplied parallel EN/FR
pairs. Inapplicable rewrites are reported, not counted as passing. Clause order,
voice and exception movement can reveal changed actor/scope extractions. Condition
lists remain atomic in the existing compiler; unsupported equivalence/unaligned
predicates remain inconclusive rather than guessed. Fixture aliases are explicit
phrase correspondences; no automatic translator or embedding alignment is trusted.

Behavior-strength and normative-output comparisons are distinct: O→P weakens
hard duty constraints, but Phase4 preserves their normative modality distinction.
No Phase4 definition was overwritten. MVR denominators show completed comparisons
alongside unknown and inapplicable counts. Equivalence-preserving transformations
compose to depth<=3; directional transformations are single-step. An unresolved
shallower comparison prevents an exact-radius claim. No missing failure becomes
infinite robustness. Live providers use the existing bounded runtime adapters,
remain unreviewed, and are excluded from CI; their actual performance is unmeasured.

The original 0.7.0 failure artifact is immutable in this branch. The token-boundary
repair is documented in the dated addendum, with the 0.8.0 result stored separately.
No benchmark, hypothesis or threshold was altered to obtain improvement.

## N5: human question planning

The state is a finite family of explicit formal worlds. World verdicts are
recomputed from formulas/traces; caller truth labels are rejected. Existing
interpretation-family parsing can generate worlds while preserving its review
findings. Each question supplies finite answer partitions, a review type, prompt
and positive cost (default 1, explicitly assumed). Answers prune hypothetical
worlds only in the plan; no answer is inserted into an actual review ledger.

The multi-answer dynamic program minimizes worst-case cumulative cost. Boolean
cases cross-check the existing factual-query minimax kernel. If no question set
can separate opposite labels, concrete witnesses/certificates are returned.
Bound hits or unknown world verdicts suppress plans and optimality claims.
Conditional terminal labels are not permission to act. The independent checker
validates the finite world-verdict artifacts, not the whole minimax algorithm.

The simulation reports all world/order distributions, no inferred priors. Random
review is eight recorded seeded orders. Equal-cost question counts and unequal-
cost objectives are distinguished. Some fixed orders are already optimal and
some question sets are insufficient; those cases are retained. Simulated oracles
are not models of real reviewer time, expertise, disagreement or fatigue.

## N3: behavioral metrics and topology

Exact distance is normalized symmetric difference under one explicit shared
finite Boolean measure. Equality means identical **bounded behavior**, not
identical legal meaning or syntax. Duplicate interpretations remain unresolved;
quotient classes do not authorize representative selection. Empty admissible
universes and incompatible per-member assumptions require explicit handling.
Permissions with identical hard behavior may have distance 0 despite different
normative effects; Phase4's normative distinction remains a separate diagnostic.

Sampling uses the same seeded valuations for all pairs, with simultaneous 95%
Hoeffding/union-bound intervals. It introduces no legal likelihood prior. The
resulting empirical Hamming metric obeys triangle inequalities on its empirical
quotient; this is not proof that estimates equal the exact legal model. Exact
checks use integer counts. Topology is small Vietoris–Rips reduction over F2 with
triangles included, not graph cycles alone. Infinite bars are right-censored.
H0/H1 features are explanatory hypotheses, not automatically discovered schools
of interpretation, non-transitive normative orders or amendment causation.

The frozen eight-unit H3 pipeline uses the existing annotation assessment,
explicit simulated non-expert submissions and leave-one-unit-out regression.
H0 beats alternative count but loses to mean distance, so the predeclared
hypothesis is not supported even in this simulation. Real predictive validity
requires independently collected, properly reviewed legal annotation data.

## Bounds and operational limits

| Component | Declared limits |
| --- | --- |
| Checker |<=64 atoms,128 states in a trace,512 AST nodes/formula,48 depth,64 assumptions; <=1m enumerated completions subject to the262144-cell materialization cap,2m evaluation work and<=60s when explicitly configured |
| Obstruction |<=12 cells,horizon<=32, bounded SAT/state/work/time; default 65536 states/5000ms. Timeline<=16 snapshots and16 tracked families |
| Metamorphic |<=64 cases,depth<=3, controlled transforms and explicit work/time limits; source<=32k, provider requests/responses bounded by existing runtime transport |
| Review planning |<=4096 worlds,12 questions,16 answers/question; positive integer costs<=100000; explicit work/time caps, unknowns never partial plans |
| Geometry/topology |<=16 points,horizon<=32,64 atoms; exact finite enumeration cap,<=65536 shared samples, bounded 2-skeleton reduction |
| Audit certificate replay |<=128 unique certificates/shared 5000 ms, existing 16 MiB/32-member archive constraints |

Dependency installation/security feeds still need networked supply-chain
artifacts in the existing CI; **analyses/tests** use no network, model server or
GPU. A fully disconnected runner needs preprovisioned dependencies/security
feeds. No production deployment, legal certification, independent expert study,
source-completeness guarantee or mechanized Python proof follows from these tests.
