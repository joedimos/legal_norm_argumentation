# Phase 5: inspected APIs and semantic deltas

Baseline: main `47e84dc0a98457cb688764250efa25043df9843f`, shipped 0.7.0.
Inspected typed norms, conflicts, references, finite-trace verifier/evaluator,
Dung argumentation, benchmark metrics, audit ZIP verification, review ledger,
evidence and observation planners, interpretation-family contracts and analysis,
independent annotation and signed-annotation modules. Phase 4 provider adapters,
alignment/comparison and fail-closed budgets are available. No TDA package or
Lean project is present. Target version is 0.8.0.

## Actual integration points

- `Formula`, `evaluate_formula`, `verify`: use explicit finite trace ASTs; no
  general legal satisfiability or deontic proof system is implied.
- `TemporalNorm.constraint`, `norm_codec.decode_norm`: translate typed norms
  only within their existing declared window/calendar restrictions.
- `detect_conflicts`, `scopes_overlap`, `ReferenceResolver`: preserve the existing
  scope and reference metadata when building overlap relations.
- `parse_family`: retains quote-bound alternatives, assumptions and review
  findings. New research adapters must not create review receipts.
- `optimal_strategy`: existing Boolean-query minimax kernel. Finite multi-answer
  interpretive questions require a compatible separate extension, not fabricated
  Boolean observations or modification of the factual planner's semantics.
- `assess_annotations`: requires explicit complete blinded packets/submissions;
  simulated pipelines must identify themselves as simulations, not expert data.
- `create_bundle`/`verify_bundle`: maintain v1 manifests. New certificates are
  opt-in artifacts; pre-certificate bundles still verify their old integrity.
- `Budget`, `BoundHit`, `envelope`: reuse work/deadline budgets and unauthorized,
  unreviewed result envelopes. A checker must not import these reasoner modules.

## Mathematical corrections fixed before experiments

1. Pairwise **satisfiability** is weaker than relational pairwise consistency
   (equal overlap projections). The acyclic chain p, (p iff q), not q has
   satisfiable pairs but no global assignment. Therefore no shortcut may infer
   a global section solely from pair SAT plus GYO acyclicity. Use full semijoin
   consistency on a verified join tree and independently replay a global witness.
   F2 is tested under its correct relational-consistency precondition; the
   unqualified pair-SAT reading is explicitly refuted by a fixed counterexample.
2. Some minimal unsatisfiable families have no graph cycle. Such an obstruction
   is reported explicitly as `higher_order_conflict` (additive new research
   status), rather than inventing a cycle or reporting compliance. A cycle in a
   nerve's one-skeleton can bound a filled simplex; it is not proof of H1.
3. The proposed behavioral distance is a pseudometric on syntax and a metric on
   equivalence classes of identical bounded behavior. Identity of indiscernibles
   applies to this quotient, not to different strings/IDs with equal behavior.
   H1 bars are topological summaries, not non-transitivity of a metric, amendment
   causation or nontrivial legal reasoning loops.
4. Obligation→permission is weaker by satisfying-behavior inclusion, while
   Phase 4 intentionally reports different modalities as incomparable normative
   outputs. The metamorphic harness must report both views, never alter Phase 4.
5. Unknown is not universally prohibited from becoming true under information
   refinement: unknown AND false is false, while learning an unknown atom true
   can resolve it true. The invariant is **no unsupported promotion**. No Lean
   proof is claimed; the independent Python checker is tested, not mechanized.
6. Certificates bind the complete explicit formal query/horizon/assumptions,
   not natural-language meaning, priority correctness, source completeness or
   expert review. Unsat cores alone are not refutations. Use independent
   enumeration rather than claiming Z3 exposes a proof format we can check.

## Literature check (28 October 2026 client date)

We hypothesize that local/global consistency diagnostics can be useful in legal
formalization; originality is unestablished. Searches for sheaf/cohomological
legal-norm consistency found adjacent institutional/contracting applications as
well as relational consistency and contextuality theory. Lack of a search hit
cannot establish priority. Primary grounding:

- Abramsky & Brandenburger (2011), https://arxiv.org/abs/1102.0264v7:
  global-section obstructions in contextuality; not an empirical legal claim.
- Beeri, Fagin, Maier & Yannakakis (1983),
  https://research.ibm.com/publications/on-the-desirability-of-acyclic-database-schemes:
  acyclic relational consistency, requiring the appropriate consistency notion.
- Author-hosted adjacent institutional discussion:
  https://cox-associates.com/index_htm_files/Online_Supplement.pdf.
  This prevents any inference that adjacent institutional applications are absent.

F3 requires frozen, eligible Légifrance snapshots. Inventory existing local data
before analysis; if none are eligible, report `not_run_missing_reviewed_data`
with null empirical counts, not a manufactured zero-obstruction result.
