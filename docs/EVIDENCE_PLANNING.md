# Bounded evidence acquisition

`src/evidence_planning.py` turns a contingent temporal proposition into an
adaptive observation plan. It extends phase 3's witness suggestions with exact
minimax planning over a deliberately small, fully enumerated model.

This is a finite decision problem, not a claim of research novelty. It assumes
that each permitted query reveals the true Boolean value of one cell, at a
fixed positive cost. It does not model noise, adversarial answers, uncertain
legal interpretation, causal interventions, priors, expected costs, or the
physical/legal availability of evidence. A human must establish the assumptions,
query permissions and costs before acting on a plan.

## Run

```bash
python -m src.cli plan-evidence --input examples/evidence-plan.json
python -m src.cli plan-evidence --input examples/evidence-plan.json \
  --bundle evidence-plan.zip --revision "$(git rev-parse HEAD)"
python -m src.cli verify-bundle --input evidence-plan.zip
```

A request has `schema: vle-evidence-plan/v1`, a temporal `property`, and `horizon`.
Optional `assumptions` and `observations` have the same semantics as phase 3:
complete Boolean traces at exactly that horizon, consistent with all observations
and assumptions. Missing/null observations and missing suffix steps are unknown.
Contradictions in the model produce `inconsistent_evidence`.

`queries` is an optional array of `{step, atom, cost}`. Omit it to allow every
unknown cell at unit cost; supply an empty array to allow no queries. Explicit
query atoms are included in the vocabulary. Costs must be positive integers up
to 100,000. Duplicate cells, already observed Boolean cells, and out-of-range
steps are rejected. Assumptions may determine a cell without a direct observation;
the planner then skips a query that cannot split the current feasible set.

## Semantic contract

Let S be a nonempty set of feasible completions and let q reveal a Boolean cell
at cost c(q). Let S0 and S1 contain its false and true completions. The optimum is:

- V(S) = 0 when the property has the same truth value throughout S.
- Otherwise V(S) = minimum over permitted nonconstant queries q of
  `c(q) + max(V(S0), V(S1))`.
- A mixed state that no permitted query can resolve has infinite cost.

Positive costs and strictly smaller successor sets give an acyclic recursion.
Memoization stores completion subsets as bitsets. Stable query ordering resolves
cost ties deterministically. The algorithm exhausts every permitted split before
claiming `optimal_in_scope: true`. Its claim covers the declared finite model and
objective, not real-world evidence collection or a better legal interpretation.

The output is a decision DAG with a root ID and nodes. Query nodes expose step,
atom, price, false/true successor IDs, compatible-completion count, and optimal
remaining worst-case cost. Leaf nodes give `definitely_true` or `definitely_false`.
Follow the branch matching the observed answer. A query inferred to be constant
from preceding answers is not asked again.

| Status | Interpretation |
| --- | --- |
| `planned` | A complete finite strategy was proved optimal within the declared scope |
| `already_decided` | All feasible completions already agree; cost is zero |
| `unresolvable` | Two opposite-verdict completions agree on every available query |
| `inconsistent_evidence` | No feasible completion exists |
| `unknown` | The completion, state or time bound was reached; no plan or optimum is released |

`unresolvable` returns the two indistinguishable completions so the result can be
checked independently. This is an information limitation of the permitted queries;
it does not establish that the legal question itself is irresolvable.

A successful plan may terminate in a negative verdict. CLI exit 0 for `planned`
or `already_decided` means planning succeeded, not that behavior is compliant.
Other valid statuses exit 1; invalid requests exit 2. `legal_certification` is
always false. The existing HTTP authorization and review gates are unchanged.

## Weighted adaptive example

The supplied formula is `(selector AND p) OR (NOT selector AND q)`. Observing
selector costs 1, while p and q cost 10 each. The optimal plan first asks selector,
then asks only the selected branch. Its worst-case cost is 11. Asking both costly
cells would cost 20; asking one before selector can cost 21 in the worst case.
The exact plan is derived from all eight completions, not from a sample.

## Bounds and verification

Default limits are 1,024 total completions, 10,000 memoized states, and a shared
5-second planning timeout. Configurable maxima are 4,096 completions, 100,000
states, and 60 seconds. The completion cap applies before assumptions filter the
space: the algorithm never claims that assumptions make an oversized unenumerated
space safe to search. At most 12 query cells and the existing 64 atoms/128 steps
are supported. Fixed observations are shared across completions in memory.
Limits apply during enumeration, planning, witness construction and plan emission.
A limit discards any partial plan and leaves `optimal_in_scope: false`.

The request's canonical SHA-256 binds formulas, evidence, query costs and budgets.
The existing deterministic audit-bundle format retains request, plan and runtime
metadata, with overwrite protection and optional trusted-manifest verification.
Re-execution is the current independent optimality check; the output is not a
standalone machine-checked proof certificate.

Tests replay every path in the weighted example; check all 16 Boolean functions
of two atoms against their decision complexities under unequal costs; exercise
restricted queries, temporal cells, fixed evidence and assumptions; check
indistinguishability witnesses; and verify timeout/state limits and audit export.

## Research enabled by this extension

A controlled study can now compare exact bounded minimax planning against random,
fixed-order, witness-difference and expert-chosen questions. Report worst-case
cost, realized cost, questions asked, disagreement with expert annotation, and
unresolved/limit cases separately. Do not substitute query cost for expert effort
without validation. Use held-out sources and explicitly documented atom mappings.

A second study can vary assumptions, prices and query availability while keeping
the source interpretation fixed, measuring plan and outcome stability. For
alternate legal interpretations, run separate models; current planning resolves
observational uncertainty inside one declared interpretation.
