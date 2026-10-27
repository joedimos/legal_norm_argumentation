# Statutory alignment: interpretation families and factual clarification

This first implementation milestone follows the proposal based on Delage et al.,
*Statutory AI: Aligning Large Language Models With Legal Norms*,
https://arxiv.org/abs/2608.28593 and the publisher version at
https://doi.org/10.1007/978-3-032-31319-5_25.

The paper uses thematic classification and statute-guided critique/revision. This
extension supplies an independent computational layer for uncertainty about the
facts and disagreement between formal interpretations. It does not yet generate,
revise, or automatically formalize LLM responses, and does not claim to reproduce
the paper's empirical results. It establishes a tested research interface for the
subsequent grounded revision loop.

## Reproduce

```bash
python -m pip install -e '.[dev,verification]'
python -m src.cli family --input examples/interpretation-family.json
python -m src.cli family --input examples/factual-family.json
python -m src.cli family-benchmark
python -m src.cli family --input examples/interpretation-family.json \
  --bundle family-evidence.zip --revision "$(git rev-parse HEAD)"
python -m src.cli verify-bundle --input family-evidence.zip
```

The examples exit 1 because they are unreviewed. `family-benchmark` exits 0 only
when all synthetic expected *analysis* labels match. It does not approve the
examples. Invalid input exits 2. A regular `family` call exits 0 only for a
nonvacuous `robust_satisfied` result with no model/grounding review findings.
Deployment authorization and legal certification remain false in every result.

## Contract and provenance

A `vle-family/v1` request contains:

- `context`: explicit jurisdiction, canonical as-of date, actor and capacity.
- `documents`: immutable-in-request snapshots with ID, version, source text,
  SHA-256 and language. Kinds are normative, observation or draft.
- `interpretations`: unique IDs, an existing temporal-AST property, optional
  assumptions, quoted normative source spans, and a mapping rationale.
- `horizon`: the exact finite number of observation steps.
- Optional partial `observations`, evidence `grounding`, factual `queries`,
  content-bound `reviews`, and resource budgets.

Normative sources need an explicitly stated effective period (either endpoint
may be explicitly null to mean unbounded). Jurisdiction mismatches, sources
outside the as-of date, missing period metadata and unresolved declared references
block a reviewed result. One snapshot version per document ID is allowed; a
source update must be selected explicitly. Offsets use Unicode character indices,
with half-open bounds. Quotes and hashes must match the supplied text exactly.

The reference graph checks the closure of *declared* references only. It cannot
establish that all references or relevant law were discovered. A language label is
not a translation-equivalence check. This milestone does not resolve jurisdictions,
authority hierarchies, lexical aliases, or ambiguous actor capacities automatically.

Every known Boolean observation needs a grounding record with exactly matching
step, atom and value plus quoted observation evidence. A draft is never trusted
as that evidence. An intact quotation does not prove that it entails the fact:
the grounding mapping requires its own review. Source facts supplied to the
computation are explicit caller assumptions, not independently established truths.

## Content-bound reviews

There are two review scopes. The family fingerprint binds context, horizon,
normative snapshots and every interpretation including its rationale/assumptions.
The grounding fingerprint additionally binds observations, their mappings, and
observation/draft snapshots. Changes invalidate the corresponding approval;
family changes also invalidate grounding approval. Missing, stale, rejected or
revision-requested records keep top-level status `review_required`.

An explicit lab review can be recorded as follows:

```bash
python -m src.cli family-review --input family.json --scope family \
  --reviewer 'Reviewer identity' --notes 'Actual interpretation review rationale' > family-reviewed.json
python -m src.cli family-review --input family-reviewed.json --scope grounding \
  --reviewer 'Grounding reviewer' --notes 'Actual evidence review rationale' > grounded-reviewed.json
```

Receipts record caller-declared identities, UTC timestamps, notes, decisions and
fingerprints. They are neither signatures nor professional credentials, and the
CLI does not authenticate the reviewer. Do not fabricate approvals to run a demo.
This research interface does not bypass or replace the service's authenticated
review and separation-of-duties controls. Production wiring is a later milestone.
A matching receipt cannot override a source-kind, validity or reference finding.

Exploratory `analysis` is retained even when review gates fail. The top-level
status then remains `review_required`; `deployment_authorized` is always false.
A consumer must not treat a nested exploratory result as an authorized decision.

## Family semantics

For each interpretation I, enumerate complete Boolean traces of exactly horizon
h that satisfy its assumptions and agree with the supplied fixed observations.
A world is `(interpretation ID, complete trace)`, labelled by that member's
property. The vocabulary is the union of all formula, assumption, observation and
query atoms. All atoms are opaque identifiers. Member assumptions are not pooled:
contradictions between alternatives do not cause logical explosion.

Each interpretation must have at least one feasible completion. If any has none,
return `inconsistent_family` and name it. It is never silently removed to make
another interpretation appear robust. Encode conditional activation as a guard in
the property when an inapplicable duty should evaluate vacuously; do not use
member assumptions to conceal an inconvenient interpretation.

| Analysis status | Meaning within the declared finite model |
| --- | --- |
| `robust_satisfied` | Every member has a feasible world and all worlds satisfy their member property |
| `robust_violated` | Every member has a feasible world and all worlds violate their member property |
| `interpretation_disagreement` | Two members give opposite labels under the identical complete trace |
| `clarification_needed` | A complete factual query strategy exists; worlds have mixed labels |
| `unresolvable_queries` | Opposite-label worlds agree on every permitted query |
| `inconsistent_family` | At least one member has no feasible completion |
| `unknown` | A computation budget prevents a complete result |

Identical complete facts with opposite interpretation labels constitute a direct
indistinguishability witness: no factual query over those atoms can select a
uniform verdict. The output retains both IDs, identical traces and opposite
truth labels, directing the workflow to interpretation review. This does not
claim that a legal dispute is impossible for humans to resolve.

Where possible, the planner minimizes worst-case total factual-query cost over
the *family's worlds*, using the shared minimax kernel. Queries cannot reveal an
interpretation ID; their answers depend only on a factual step/atom. No noisy
answers, probability weighting or expected-cost objective is implemented. Plans
are meaningful only if the model and query permissions/prices are justified.
Leaves can establish either truth or falsity. A successful plan is not an
assertion that the response is permissible.

## Bounds and failure behavior

At most 16 interpretations, 32 snapshots, 64 distinct atoms, 128 steps, 64
assumptions per member and 12 query cells are accepted. Snapshot text is limited
to 500,000 characters each; the CLI also enforces its existing 16 MiB input cap.
Defaults: 1,024 complete factual assignments, 8,192 feasible member-world pairs,
10,000 planning states and a shared 5-second analysis timeout. Maximum configurable
caps are 4,096 assignments, 65,536 worlds, 100,000 states and 60 seconds.

The assignment cap is checked before filtering by assumptions. Budget exhaustion
releases no partial plan and makes no optimum or family-wide truth claim. Fixed
facts are shared between compact completion tuples. Canonical request hashes and
the existing audit archives retain source, request, result and runtime evidence;
external trust in the archive's origin still needs a separately trusted digest.

## Validation and annotation status

`benchmarks/families-v1.json` contains 15 hand-labelled semantic fixtures. The
French source text is a synthetic research policy, not a French legal provision.
All examples are deliberately marked `synthetic_unreviewed`, contain no review
receipts, and remain blocked at top-level review. Benchmark scoring uses the
explicit exploratory analysis label. No independent expert annotation or
real-world statutory accuracy is claimed.

Tests check provenance corruption, reviewed-content changes, actor/source-date
scope, missing grounding, draft self-grounding, inconsistent family members,
identical-fact disagreement witnesses, factual plans, and budget failures. They
compare all pairs from a small formula set against a complete-trace oracle and
cross-check uniform family verdicts against the independent Z3 backend. Existing
single-interpretation planner tests remain enabled after extracting a shared
minimax kernel.

## Next milestones

1. Independent expert annotation: use `FAMILY_ANNOTATION_PROTOCOL.md` to curate
   real, rights-cleared source-to-model examples with preserved disagreements.
2. Response-grounding contracts: bind candidate text spans and intended actions
   to reviewed predicates, retaining alternate mappings. Never upgrade a generated
   assertion into an observed fact simply because an LLM stated it.
3. Bounded critique/revision: feed structured counterexamples to a model adapter,
   then re-ground and recheck each revision with explicit stopping budgets.
4. Service integration: authenticated registry/review storage, historical receipt
   snapshots and authorization gates, with separate LEGATO and VLE reporting.
5. Reproduce and extend the paper: run its baseline alongside the new stages,
   measuring all routed/unrouted prompts, false refusals, usefulness, grounding
   failures, latency and human effort under matched budgets.

These are future milestones, not capabilities asserted by the current interface.

## Response-grounding follow-on

The [response-alignment contract](RESPONSE_ALIGNMENT.md) adds separately reviewed
`behavior` mappings and a `behavior` receipt scope. These expand the world set to
interpretation/mapping/trace products, with explicit response-action namespaces.
The provided-revision controller is implemented; live generation/model adapters
and independent legal annotation remain future work.
