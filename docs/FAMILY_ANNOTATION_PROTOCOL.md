# Independent annotation protocol for statutory interpretation families

The shipped family corpus is synthetic and unreviewed. This protocol specifies
how to construct an independent legal dataset; it does not assert that such
annotation has already happened.

## Unit of annotation

A unit includes the authorized source snapshot and version, jurisdiction/as-of
scope, actor/capacity, request and candidate answer, normative spans including
conditions/exceptions, one or more plausible formal interpretations, predicate
mappings, factual evidence, and an explicitly bounded behavior/trace model.
Separate normative applicability, linguistic faithfulness, factual grounding,
harmful assistance, usefulness, and refusal appropriateness.

## Collection and review

1. Select a narrow French IP/competition-law task and obtain permission or a
   documented reuse basis for every source and response. Save official versions
   and all declared definitions/exceptions/cross-references. Do not assume the
   present collector covers every relevant provision.
2. Split by source provision and related amendment lineage before developing
   prompts/models. Keep near-duplicate cases and translations together to avoid
   leakage. Freeze a held-out portion before tuning thresholds or examples.
3. Have two qualified annotators work independently without model verdicts or
   benchmark predictions. Record identity/qualification under the study's consent
   and access rules; the CLI receipt alone does not establish either.
4. Annotate alternate interpretations rather than forcing agreement. Record why
   an interpretation is plausible, which context it assumes, and how each atom
   corresponds to source/answer spans. Do not label a solver-consistent mapping
   as legally faithful just because it is satisfiable.
5. Adjudicate discrepancies with a third reviewer or retain an explicit unresolved
   family. Preserve original annotations, adjudication and version histories.
6. Mark factual evidence independently of generated assertions. Inspect whether
   quotations support the predicate, the actor and the observation time. Distinguish
   missing evidence from evidence that a condition is false.
7. Freeze labels, provenance and exclusions before running the candidate system.
   Report unrouteable requests, missing corpus coverage, abstentions and resource
   limits in the denominator, rather than silently discarding them.

## Evaluation

Report source retrieval/coverage, span/condition/exception preservation, unsupported
citations, grounding errors, interpretation disagreements, harmful assistance,
false refusals, usefulness and expert overturns separately. Include inter-annotator
agreement and the adjudication distribution. Use paired comparisons on identical
cases with confidence intervals and declared stopping rules. Compare the paper's
statutory baseline, versioned retrieval, verification, interpretation families and
factual planning as separate ablations under matched inference budgets.

No percentage result from the synthetic fixtures should be advertised as legal
accuracy, generalization, or an independent replication of arXiv:2608.28593.

The executable packet, submission and adjudication workflow is described in
[LIVE_GENERATION_ANNOTATION.md](LIVE_GENERATION_ANNOTATION.md). It does not substitute
for recruiting independent legal experts or verifying their qualifications.

For vLLM, runtime diagnostics, blank assignments and signed independent judgments,
see [INFERENCE_AND_SIGNED_REVIEW.md](INFERENCE_AND_SIGNED_REVIEW.md).
