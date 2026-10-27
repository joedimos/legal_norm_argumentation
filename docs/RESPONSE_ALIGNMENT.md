# Response grounding and bounded statutory revision

This milestone adds a response-to-behavior contract and a bounded controller to
the interpretation-family foundation. The controller evaluates **provided**
candidate revisions; it does not call a model, generate text or automatically
approve a mapping. It can be connected to a model adapter later without weakening
the current review boundary. It does not reproduce the paper's empirical results.

## Use

```bash
python -m src.cli statutory-revise --input examples/statutory-revision.json
python -m src.cli statutory-revise --input examples/statutory-revision.json \
  --bundle revision-evidence.zip --revision "$(git rev-parse HEAD)"
python -m src.cli verify-bundle --input revision-evidence.zip
```

The example is deliberately unreviewed and exits 1 with `review_required`, without
consuming its next candidate. There are no fictional expert approvals. Python
callers can use `bind_candidate` to create a family request for inspection and
`record_family_review(..., scope="behavior", ...)` to record an actual explicit
lab review. Receipts remain self-reported; they are not authenticated credentials.

A sequence uses `vle-revision/v1`, a base `family`, `candidates` (1-5), and optional
`max_cycles` (1-5, default 2), counting the initial candidate. Each candidate has:

- A `document` snapshot of kind `draft`, with exact text and SHA-256.
- A `behavior` contract binding that draft ID to `action_atoms` and 1-8 alternative
  mappings. Each mapping has an ID, constraints, quoted draft spans and rationale.
- Optional candidate-specific `grounding` and `behavior` review receipts. It may
  not overwrite the base family's normative approval.

## Claims are not observations

A response action belongs to the reserved `response:` namespace, for example
`response:disclose`. Behavior constraints may mention only declared response
action atoms. They are hypotheses about what the response proposes or enables,
not assertions that the action happened in the world. They cannot assert external
facts such as `fact:authorization_documented`.

When a behavior contract is present, response action atoms cannot be supplied as
trusted observations or queried as factual evidence. External facts need separate
observation evidence and review. A generated statement about authorization must
not become an observed authorization just because an LLM stated it. This enforces
an explicit modeling distinction; human review must still establish the correct
meaning of each atom and the faithfulness of a response-action mapping.

The family fingerprint continues to bind normative sources and interpretations.
A separate behavior fingerprint binds the candidate, alternate mappings, family
and grounding context. Editing text, action constraints, rationales or context
invalidates that receipt. Draft changes also invalidate grounding receipts under
the existing conservative grounding fingerprint. A matching receipt cannot erase
source-kind, missing-grounding, validity or reference findings.

## Product semantics

The computation ranges over `(normative interpretation, behavior mapping,
complete trace)` worlds. A trace must satisfy both the member's assumptions and
the selected hypothetical behavior constraints. Every interpretation/mapping pair
must remain feasible; an inconsistent mapping cannot be silently discarded.
`scope.model_ids` binds compound model IDs back to both source IDs.

- Opposite normative verdicts under identical complete facts still produce an
  `interpretation_disagreement` witness.
- Opposite verdicts from alternate mappings with the same factual projection can
  produce `grounding_disagreement`.
- An underspecified response action that remains ambiguous after all available
  facts produces `behavior_uncertainty` rather than a question asking the model
  to certify its own intended behavior.
- Disagreements involving both dimensions can be marked `model_disagreement`.

These results require review, not factual invention or automatic preference for
a convenient interpretation. The same completion, world, state and timeout caps
apply. The default factual-query set excludes all reserved response atoms.

## Controller and stopping rules

Every candidate is bound and checked anew. A `robust_violated` result supplies a
structured counterexample as feedback and allows examination of the next supplied
revision. A `robust_satisfied` result stops as `model_satisfied`. Review gaps,
disagreement, uncertain behavior, missing facts, inconsistent models and resource
limits stop immediately for the corresponding review/clarification step.

Revisiting the same candidate text stops as `stalled`; changing only its mapping
cannot silently turn an unchanged answer into a successful revision. No candidate
within the declared budget yields `revision_limit`. Results retain each attempted
candidate hash, feedback, full family result and the selected candidate ID and SHA-256, if any.
The archive retains the complete input sequence and runtime metadata. The family
timeout is per cycle, not a shared timeout for the entire sequence (up to five
cycles). Only attempted candidates are interpreted; unused revisions remain input
artifacts, not approved results.

CLI exit 0 for `model_satisfied` means the supplied model conditions passed after
current lab receipts. Deployment authorization and legal certification remain
false. A passing bounded behavior model is not proof that free-form text is safe,
that a source-to-model mapping is faithful, or that a provision applies legally.

## Subsequent integration

A generation adapter should receive concise structured findings, source IDs and
counterexamples, then return a new unreviewed candidate and mapping proposals.
It must not generate its own reviewer approval or promote its assertions to facts.
A controlled study can compare provided expert revisions, model revisions with
expert mapping review, and automated mapping proposals against independently
annotated data. Report failures at the routing, retrieval, mapping, verification
and revision stages separately. Live model calls and that evaluation are not
implemented by this milestone.
