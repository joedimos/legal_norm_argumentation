# Evidence-Bound Legal Norm Argumentation

## A framework for faithful autoformalization, bounded normative reasoning, and governed model inference

**Joseph Dimos**

Research software manuscript | 29 October 2026 | Version 0.9.0

## Abstract

Automatically converting legal language into executable rules creates two distinct verification problems: whether a representation preserves the source's meaning, and whether a system behaves consistently with that representation. This paper presents legal_norm_argumentation, a research and service framework that keeps those problems separate while connecting them through versioned evidence. The implementation combines source-bound extraction, explicit interpretation alternatives, defeasible activation, structured conflict analysis, preference-aware argumentation, finite-trace temporal verification, and counterexample-guided response revision. A product model over normative interpretations, response-behavior mappings, and factual completions preserves uncertainty rather than choosing a convenient interpretation. A finite minimax planner identifies informative factual questions without treating interpretive choices as observable facts. Operational components include authenticated review, signed independent annotations, append-only persistence, durable inference claims, runtime configuration binding, source ingestion, and reproducible audit archives. The expanded implementation adds redundant formalizations, explicit DDL losses, assumption corrections, structured arguments, independently checkable finite-model certificates, local/global consistency diagnostics, metamorphic tests, interpretation geometry and review planning. Automated acquisition targets French IP-code articles and U.S. IP opinions. Local evaluation reports 461 passing tests and four PostgreSQL-specific skips, together with complete agreement on the expected labels of 62 semantic, 19 experiment, and 15 interpretation-family regression cases. These results concern implementation behavior on synthetic fixtures; they do not establish real-world legal accuracy. The paper specifies conditional correctness arguments, the deployment trust model, current limitations, and a falsifiable empirical research agenda.

**Keywords:** legal autoformalization; normative reasoning; argumentation; finite-trace verification; interpretation uncertainty; statutory alignment; reproducibility.

## 1. Introduction

Legal and organizational norms constrain AI systems at several levels. A statute may govern an actor's conduct; a policy may regulate an organization's use of a model; a contract may impose a deadline; and an ethical principle may influence which trade-offs require human review. These sources differ in authority, scope, temporal effect and interpretive uncertainty. A system that maps all of them directly into Boolean rules risks discarding precisely the distinctions that determine legal applicability.

The LEGATO research problem begins before automated reasoning. A well-formed rule can omit a condition, reverse an exception, conflate permission with obligation, or cite an outdated provision. Conversely, a faithful representation can be used incorrectly by a reasoner or connected to unsupported factual claims. We therefore distinguish source faithfulness, formal consequence, factual grounding, response characterization and operational authorization. Success at one layer is not evidence that all the others succeeded.

The framework studied here addresses this separation as an engineering and knowledge-representation problem. It maintains traceability from source snapshots to formal alternatives, requires explicit review of material interpretation choices, and reports when the available evidence cannot determine a conclusion. Bounded verification supplies counterexamples to supplied models. It does not decide what the law ought to mean. Human review supplies accountable interpretations; it does not turn an inconsistent formal model into a valid one.

This paper has four contributions. First, it describes a unified evidence model joining extraction, interpretation, reasoning and governance. Second, it formalizes analysis over interpretation families and alternative response mappings, with factual clarification treated as a restricted decision problem. Third, it explains how that analysis constrains live generation and durable service execution. Fourth, it provides a reproducible synthetic evaluation and a study design for the independent legal evaluation that remains outstanding. The integration is the claimed contribution; priority over all prior approaches or empirical superiority is not established.

## 2. Background and related work

Dung's abstract argumentation framework separates the structure of attack from the internal language of arguments [1]. This is useful when conflicting norms must be assessed through explicit defeat relations. Our implementation uses grounded, preferred and stable semantics and an explicit acyclic preference relation. It does not equate argument acceptability with a court's interpretation or treat an empty collection of stable extensions as universal acceptance.

Finite-trace temporal logic provides a precise language for bounded behavioral questions. De Giacomo and Vardi distinguish finite-trace interpretations of temporal specifications from infinite-run semantics [2]. Our implementation uses a restricted LTLf syntax, not LDLf. A verifier checks complete traces at an exact declared horizon, using independent exhaustive and optional SMT encodings. Z3 supplies the symbolic backend [3]. The SMT result is conditional on the encoding, supplied assumptions and bound; it is not a certificate that a natural-language provision was formalized faithfully.

LegalRuleML provides a standardized vocabulary for representing legal rules and their contextual features [4]. It motivates attention to modality, authority, temporal context and provenance. The present JSON contracts are a project-specific operational format. No LegalRuleML conformance, complete legal ontology, or lossless interchange implementation is claimed. Such interoperability is a future adapter and equivalence-testing task.

Statutory AI uses legal material to guide the critique and revision of model outputs [5]. Its two-stage approach routes a prompt to statutory themes and then conditions revision on selected legal provisions. Our use of the paper is methodological: statute-conditioned revision creates a useful control loop, but faithful source-to-rule and response-to-behavior mappings remain separate questions. We add explicit alternatives, factual evidence constraints and review checkpoints. We do not reproduce the paper's empirical findings, adopt its penal-law dataset as our gold corpus, or infer that its routing categories cover other legal domains.

Governance literature also distinguishes technical measurements from contextual risk management. NIST AI RMF 1.0 is a voluntary, use-case-agnostic risk-management framework [6]. We use this distinction to explain why runtime tests, external legal review and release decisions are different evidence categories. The project is not certified against that framework. Ed25519 signatures [7] establish attribution to enrolled keys; they do not establish the expertise, independence or correctness of the person holding a key.

An et al. [11] use redundant formalizations and semantic-equivalence checks in a neurosymbolic policy-verification pipeline. Horner et al. [12] segment legal text into atomic snippets, extract DDL rules and apply a dedicated refinement stage, emphasizing completeness-aware evaluation. These motivate the complementary redundancy and DDL layers here. Their reported model performance is not reproduced by this project's synthetic software tests, and agreement among models cannot establish legal source faithfulness.

## 3. System architecture and scope

The project has a deterministic source-processing path, a bounded research path and a service-governance layer. The source path ingests a versioned document, proposes norm candidates, audits extraction, constructs typed interpretations, resolves supported priorities and evaluates a fact/action context. The research path accepts explicit temporal formulas, interpretation families and candidate response mappings. The service layer persists sources, reviews, selections, evaluations and workflow events and exposes role-constrained APIs and browser interfaces.

[[ARCHITECTURE]]

**Figure 1.** Evidence and control boundaries. Source-to-rule review, factual grounding and response-mapping review are distinct inputs. Verification results feed a reviewed revision loop, while operational authorization remains a separate policy decision.

| Layer | Main responsibility | Principal implementation |
| --- | --- | --- |
| Acquisition | Official snapshots, version hashes, bounded collection | legifrance.py; legifrance_piste.py |
| Autoformalization | Candidate spans, modality, conditions, exceptions, alternatives | norm_extraction.py; formalization.py |
| Normative reasoning | Activation, conflicts, priorities and argument acceptability | defeasible.py; conflicts.py; preferences.py |
| Temporal analysis | Exact-horizon interpretation and bounded verification | temporal.py; temporal_deontic.py; model_checking.py |
| Uncertainty research | Completions, amendments, interpretation products and factual questions | experiments.py; interpretation_family.py; observation_strategy.py |
| Model control | Unreviewed draft proposals, reviewed revisions and runtime adapters | live_generation.py; response_alignment.py; model_runtime.py |
| Governance | Identity roles, signed judgments, durable events and audit exports | signed_annotation.py; service/; audit_bundle.py |

The default extractor combines explicit English/French modal and clause patterns with the project's NLP facilities. It is an inspectable baseline, not a demonstrated state-of-the-art legal parser. The service's extraction path and the interpretation-family research path are also not the same reasoner. The service evaluates its typed single-context norm model; the research path evaluates explicitly supplied finite-trace formulas. Connecting both paths under identical semantics requires a reviewed translation, not an implicit claim of equivalence.

Légifrance acquisition targets French intellectual-property and competition-law corpora. The service supports the official PISTE transport with configured credentials and an explicitly selected bounded HTML route. URL restrictions, article bounds, version hashes and visible transport failures support controlled ingestion. They do not establish that the chosen corpus contains every applicable legal provision or that every cross-reference has been resolved.

### 3.1 Automated IP acquisition

A supervised importer acquires dated current IP-code manifests from production PISTE and hydrates article text using official IDs. CourtListener v4 discovery targets copyright, patent, trademark and trade-secret queries, then retrieves individual full opinions. Opinion IDs are distinct from the cluster IDs used in website URLs [10]. Each document retains source identity, raw/content hashes, language, jurisdiction and version provenance. Search snippets never substitute for full text. Opinion type is retained, while controlling authority remains unassessed.

Acquisition has durable cursors, conditional database leases, fenced atomic batch commits, provider-wide 429 cooldown and periodic reconciliation of older U.S. cases. Crashes resume the declared query cycle. Missing/oversized documents and unsupported responses block their job rather than silently counting as a completed import. Current French acquisition does not cover all French judgments, and U.S. keyword results are not a census of applicable IP law. All generated norm candidates enter unreviewed. Live provider access has not been demonstrated in this evaluation.

## 4. Source-bound representations and defeasible activation

A norm candidate records its source identifier, exact character span and quotation, modal category, actor, action, condition, exception, references and ambiguity indicators. Formalization produces a set of typed interpretations. Selection of a material alternative requires a recorded identifier and rationale. A singleton interpretation is not automatically trusted when ambiguity flags remain.

Let a represented norm be n = (id, m, s, a, o, C, E, A, P), where m is modality; s, a and o are subject, action and object; C and E are condition and exception predicates; A is authority and temporal metadata; and P is provenance. The implementation permits obligation, prohibition, permission and recommendation distinctions. Compound natural-language predicates remain atomic unless a decomposition has been supplied and reviewed.

For context x, predicate evaluation returns true, false or unknown. A norm becomes inactive when its represented temporal scope or actor excludes the case, or a condition is false. Unknown relevant conditions or exceptions yield undetermined activation. A true exception defeats the norm. Otherwise, the norm is active and may still be defeated by supported priority reasoning. This operational ordering is explicit; it is not a complete theory of legal interpretation.

An important consequence is that missing evidence does not establish an exception's absence. If a condition requires documented authorization and that predicate is unknown, neither the model's assertion nor the absence of a document in the supplied corpus makes it true. Similarly, a permission is not compiled into a duty merely because it concerns the same action as an obligation.

Conflict detection compares represented modality and overlapping scopes. Priority analysis may use explicit authority ranks, strict conjunctive scope inclusion or enactment dates between equally ranked sources. These are conditional operational rules. They are not universal legal maxims that resolve all conflicts across jurisdictions. If the metadata does not justify an ordering, the conflict remains unresolved and human interpretation is required.

## 5. Argumentation and finite-trace semantics

### 5.1 Preference-aware argumentation

Let AF = (Arg, Att) be a finite directed attack graph. A set S is conflict-free if no pair of members attacks one another. It defends argument a if every attacker of a is attacked by a member of S. Grounded semantics uses the least fixed point of the characteristic defense operator. Preferred semantics selects maximal admissible sets; stable semantics additionally attacks every argument outside the extension [1].

The explicit strict preference relation is acyclic. Its transitive closure determines which attacks are suppressed. For an attack a -> b, preference for b over a blocks that attack; incomparability preserves it. This construction makes the preference assumption visible and testable. It does not infer a preference from model confidence or from which interpretation makes the desired behavior permissible.

**Equation 1.** Defeat relation under a supplied strict preference.

EQUATION: Def = { (a,b) ∈ Att : ¬(b > a) }.

The project bounds preferred/stable extension enumeration to 16 arguments in the research interface. Where stable extensions do not exist, the system records that fact and does not license vacuous skeptical acceptance. Separate metadata supports explaining suppressed attacks and retained defeats.

### 5.2 Complete and partial temporal traces

Let Sigma be a finite set of atoms and h a positive horizon. A complete trace is sigma = (v0, ..., v(h-1)), where each valuation maps Sigma to Boolean values. Formulas use constants, atoms, negation, conjunction, disjunction, strong/weak next, eventually, always and strong until. Strong next is false at the final state, weak next is true there, and strong until requires its right-hand witness before the finite end.

**Equation 2.** Exact-horizon validity under assumptions and observations.

EQUATION: Valid<sub>h</sub>(φ | A,O) ⇔ Ω<sub>h</sub>(A,O) ≠ ∅ ∧ ∀σ ∈ Ω<sub>h</sub>(A,O): σ ⊨ φ.

The nonempty condition is essential. Inconsistent assumptions are reported separately and cannot manufacture a proof of compliance. The exhaustive backend enumerates unfixed Boolean cells; the SMT backend encodes the same horizon and constraints. Counterexamples are replayed with the independent formula evaluator before being used as evidence.

Partial observations use a conservative strong-Kleene extension. A missing atom is unknown. This is not supervaluation: for an unobserved p, p OR NOT p can evaluate as unknown even though every Boolean completion satisfies it. Completion analysis therefore remains a separate procedure rather than being inferred from a single three-valued evaluation.

**Proposition 1 (conditional partial-evaluation soundness).** For the implemented operators at a fixed finite horizon, a true strong-Kleene result implies truth in every complete Boolean trace extending the observations; a false result implies falsity in every such extension. The converse need not hold.

**Proof sketch.** Structural induction uses the information-order preservation of negation and the strong-Kleene Boolean operators. Each temporal operator is a finite composition of those operators at fixed positions, with deterministic end-of-trace constants. The excluded-middle example establishes failure of the converse. This is a property of the supplied formal language, not of its legal interpretation.

### 5.3 An operational deontic fragment

TemporalNorm represents achievement and maintenance windows with inclusive endpoints. Applicability is sampled at the declared activation point. For a reviewed guard g and action predicate a, an achievement obligation requires an action witness in the window; a maintenance obligation requires it throughout. Prohibition requires absence throughout the window. Permission contributes no hard duty constraint, although permission/prohibition conflicts remain normatively relevant.

**Equation 3.** Representative window requirements.

EQUATION: Achievement: g<sub>s</sub> ⇒ (a<sub>s</sub> ∨ ... ∨ a<sub>e</sub>). Maintenance: g<sub>s</sub> ⇒ (a<sub>s</sub> ∧ ... ∧ a<sub>e</sub>).

This fragment does not axiomatize all deontic logic. It excludes general recurring obligations, unrestricted quantification, a complete contrary-to-duty calculus and retrospective exception cancellation. A textual deadline must first be translated into a reviewed discrete window. These restrictions make the implemented semantics reproducible and expose where a richer legal model is needed.

## 6. Interpretation families, grounding and factual clarification

### 6.1 Product worlds

A legal source may support several plausible formal interpretations, and a proposed answer may support several behavior mappings. The family contract therefore retains both. Normative, observation and draft documents have separate kinds, exact quotes and content hashes. Normative snapshots carry jurisdiction and effective-period declarations, including an explicit declaration when a bound is unknown or unbounded. Unresolved references and inappropriate evidence kinds produce review findings.

Let I be the interpretation family and B the candidate's behavior-mapping alternatives. For interpretation i, property phi_i and assumptions A_i are supplied. Mapping b contributes hypothetical response constraints M_b. Observations O constrain external facts. The feasible product is:

**Equation 4.** Interpretation-family product semantics.

EQUATION: W = { (i,b,σ) : i ∈ I, b ∈ B, σ ∈ Ω<sub>h</sub>(A<sub>i</sub> ∧ M<sub>b</sub>, O) }.

Every represented (i,b) pair must have a feasible completion. An inconsistent pair is reported, not removed to make the surviving family look compliant. The family is robustly satisfied only when all feasible product worlds satisfy their corresponding phi_i; robust violation requires the opposite universal result. Mixed outcomes retain witnesses.

Opposite verdicts under the same complete trace can expose interpretation disagreement. Alternative response mappings with the same factual projection can expose grounding disagreement. An underspecified response remains behavior uncertainty. These are not automatically resolved by asking an external Boolean question, because neither legal interpretation nor the meaning of a draft is a directly observable fact.

### 6.2 Response actions are not observations

Candidate actions use a reserved `response:` namespace. Their constraints are hypotheses about what an answer proposes or enables, not observations that an event occurred. Mapping constraints may not assert external fact atoms, and response atoms cannot be inserted as trusted observations or queried as factual evidence when a behavior contract is present.

For example, consider a synthetic policy prohibiting disclosure unless authorization is documented. Let d denote proposed disclosure and u denote independently documented authorization. A draft saying 'I am authorized to disclose' does not establish u. If u is unknown, the family may require factual clarification. If d is itself ambiguous, it requires mapping review. If two legal readings disagree about what authorization means, that disagreement remains interpretive. The example illustrates the contracts; it is not a statement of French law.

Family, grounding and behavior fingerprints bind different semantic scopes. Editing source text, effective dates, mappings, assumptions or context invalidates the relevant review. Draft changes conservatively invalidate associated grounding receipts. The CLI's basic review receipts are self-reported laboratory records. Authenticated service reviews and signed independent annotations provide additional accountability, but neither removes source-meaning findings.

### 6.3 Exact factual question planning

For a finite world set S and allowed noiseless Boolean queries q with positive costs c(q), the planner partitions S into S_q^false and S_q^true. A leaf has zero cost when all worlds agree on the verdict. Otherwise the minimax recurrence is:

**Equation 5.** Worst-case clarification cost.

EQUATION: V(S) = min<sub>q</sub> [ c(q) + max( V(S | q=F), V(S | q=T) ) ].

A query is considered only when both children are nonempty. If no allowed decision tree separates opposite verdicts, the value is infinite and the result is unresolvable using the permitted observations. The implementation memoizes world subsets as bitsets and emits a decision DAG with stable query ordering. It can return opposing worlds that agree on every permitted query answer, providing a concrete explanation of non-resolvability.

**Proposition 2 (finite minimax optimality).** If enumeration and dynamic programming finish within their declared limits, the emitted strategy has minimum worst-case summed query cost among adaptive trees using the supplied noiseless Boolean queries.

**Proof sketch.** Every useful first query partitions a state into strict subsets. Induction on subset size makes the recursively computed child values optimal. Every admissible tree has some first query and incurs its cost plus the worse optimal continuation. Minimizing over those roots gives the recurrence. Resource exhaustion invalidates the completed-strategy claim and returns unknown. The proposition does not cover noisy evidence, unavailable queries, misleading witnesses or legal interpretation choices.

## 7. Amendments and counterexample-guided model revision

Amendment experiments compare sets of satisfying complete traces under shared assumptions and observations. Two checks test implication in both directions. A strict strengthening removes behaviors admitted before without adding new ones; a weakening reverses that inclusion. Counterexamples in both directions establish incomparability. Inconsistent evidence, solver unavailability and resource exhaustion remain separate results.

This comparison is behavioral inclusion in a bounded formal model. It cannot show that two legal texts have equivalent meaning. In particular, an omitted exception can produce a stronger formula while constituting a worse formalization. The inclusion label is therefore evidence for legal debugging, not an interpretation-quality score.

The response controller accepts up to five candidate cycles. Each candidate binds an exact draft snapshot to one or more reviewed behavior mappings. Robust violation supplies a counterexample for the next revision. Robust satisfaction stops the sequence. Missing review, factual clarification, inconsistent models, interpretation disagreement, uncertain behavior and resource limits all stop for the corresponding human or evidence step. Repeated candidate text stops as stalled; merely changing a mapping cannot silently rehabilitate the same answer.

Live Ollama and vLLM adapters return unreviewed text and mapping proposals. One explicit checkpoint makes at most one provider call, with non-streaming JSON, bounded requested output, response-size limits, no automatic redirect/retry and strict completion validation. vLLM responses must identify the requested model, return one assistant completion and terminate normally without tool calls or refusal payloads. Provider errors are classified without reflecting secrets or untrusted response bodies.

The adapter does not create reviewer approval, alter the normative family or promote model assertions to factual observations. Initial source/grounding approval is required. A generated answer may satisfy its proposed model while remaining unreviewed or legally mistaken. The controller's selected-candidate hash prevents ambiguity about which exact text passed the supplied model conditions.

## 8. Independent annotation and evidence governance

A frozen study packet groups units by source/amendment lineage and assigns train, development or test splits before tuning. Reused normative snapshots cannot cross splits under renamed groups. Packets remove review receipts and reject schema fields for solver predictions or model identity. Study designers must still detect paraphrase/translation leakage and remove textual clues that software cannot reliably recognize.

Two qualified reviewers should work separately on applicability, faithfulness, conditions/exceptions, grounding, response behavior, harmful assistance, usefulness and refusal appropriateness. The labels supported, unsupported, uncertain and not applicable describe evidence for separate propositions. Supported harmful assistance is adverse; it must not be averaged into a positive safety score. Alternatives are preserved rather than forced into false agreement.

The assessment reports observed agreement and nominal Cohen's kappa by dimension [9]. With observed agreement p_o and chance agreement p_e, kappa is (p_o - p_e)/(1 - p_e); it is undefined when that denominator is zero. Coarse label agreement does not establish interpretive agreement; different alternative interpretations still require adjudication. A third reviewer can resolve or explicitly retain a disagreement, while original submissions remain intact. These are descriptive sample measures, not confidence bounds or estimates of real-world legal accuracy.

Signed judgments bind the complete payload, packet hash, role and UTC timestamp using a domain-separated Ed25519 envelope [7]. A separately pinned registry supplies public keys, approved roles/jurisdictions, validity periods, revocation and custodian-defined independence groups. The service excludes generator/developer groups and repeated signer keys. Determinate interpretation and behavior judgments require matching source quotations; supported factual grounding requires observation evidence. Quotes establish integrity, not entailment.

Operational trust remains external. A registry custodian checks identity and qualifications, retains evidence under appropriate access controls and distributes the trust anchor independently of submissions. A signature verifies possession of an enrolled key, not the person's expertise or actual independence. The implementation retains these limitations explicitly and does not convert collected annotations into automatic family-review receipts or legal certification.

### 8.1 Complementary formalization and diagnostic layers

Redundant formalization retains independent unreviewed candidates and their model/prompt provenance. Predicate alignment uses explicit supported mappings and exposes unresolved atoms. Bounded cross-checks partition equivalent candidates and return stronger/weaker/incomparable witnesses. Consensus is a review aid; candidates can agree on the same omitted exception. Structured-argument analysis distinguishes strict/defeasible rules and undermining, rebutting and undercutting, lifting explicit priorities before invoking Dung semantics. It does not invent authority from model confidence.

DDL interchange records unrepresentable fields as explicit losses. Atomic snippet segmentation and second-stage refinement remain reviewable proposals. Completeness-aware evaluation reports a vector rather than allowing a correct simple fragment to hide omitted obligations. Assumption analysis surfaces bounded minimal correction/unsatisfiable subsets and source-bound review prompts. Relaxing an assumption restores satisfiability in a supplied model, not legal permission to change the source.

The Phase 5 constraint layer distinguishes pairwise inconsistency from higher-order unsatisfiability. Minimal cyclic examples can have every proper subfamily satisfiable. However, acyclicity and pairwise satisfiability alone do not imply global satisfiability: p, p iff q and not q form a counterexample. The implemented acyclic shortcut therefore requires a verified join tree and relational semijoin consistency, cross-checked against exact global enumeration. Reports preserve higher-order acyclic conflict as a separate outcome. This is a bounded set-valued constraint analysis, not a claim that linear cohomology decides legal consistency.

A small independent standard-library checker replays finite-trace violation witnesses and exhaustive compliant-within-bound certificates. Enumeration counts and Merkle roots bind the visited states. Conflict certificates provide an unsatisfiable family and witnesses for single deletions. A certificate can faithfully prove the wrong encoded question; independent intended-query binding is necessary. The checker is differential- and mutation-tested, not mechanized or a proof of the Python implementation.

Metamorphic testing applies deterministic source transformations and compares resulting behaviors, including bilingual fixtures. Behavioral geometry measures symmetric differences of compliant behavior sets: it is a pseudometric on syntax and a metric on bounded behavioral equivalence classes. Exact and seeded sampled analyses remain distinct. Rips H0/H1 diagnostics do not establish legal schools, cohomological obstruction or normative non-transitivity. Interpretive review planning computes bounded minimax question order over supplied finite worlds/costs; it suggests questions and never auto-answers them or invents priors.

## 9. Production service and authorization policy

### 9.1 Durable execution and runtime binding

The FastAPI service uses PostgreSQL in production, Alembic migrations, API-key roles, separation of duties, database-backed rate limits and idempotency. Source/review/selection/run/audit/workflow tables are append-only under PostgreSQL triggers. The browser console is an operations client over these APIs rather than a parallel reasoner. Complete prompts and workflow evidence are restricted to the owner and authorized review/audit roles.

Inference is queued, not executed inside a request handler. A worker commits a unique claim before contacting the provider. A claimed job is never automatically retried. A process crash after remote execution but before result persistence leaves uncertainty about the remote outcome; an administrator must investigate and record interruption. Explicit retries are separate audited jobs. Late results are retained without overwriting an interruption.

[[STATE_MACHINE]]

**Figure 2.** Durable inference lifecycle. Configuration drift and revoked reviews can terminate a claimed job before any provider call. Human review determines whether a completed draft may participate in a subsequent revision. Administrative interruption never promises cancellation of remote compute.

**Proposition 3 (at-most-once initiation per workflow).** With the documented PostgreSQL transaction/uniqueness assumptions and without out-of-band table mutation, two workers cannot both initiate a provider call for the same workflow claim.

**Argument.** A unique claim slot is committed before dispatch. Workers select only workflows lacking a claim/terminal event; concurrent races are additionally constrained by row locking and uniqueness. No automatic claim retry exists. This does not imply exactly-once provider computation, end-to-end delivery, or safe repetition after an ambiguous external failure. PostgreSQL locking semantics support this mechanism [8].

Version 0.6.2 additionally binds queued work to provider, model, endpoint, optional model-revision identifier and configured build revision at submission. The worker refuses silent runtime drift with a zero-call diagnostic; an explicit investigated retry binds the current runtime. Model revision remains a custodian-supplied identifier, not automatically verified weights provenance. Credential rotation does not place secrets in the binding.

Private queue diagnostics expose counts, oldest queued age and stale claims without loading full prompt/result JSON into list responses. Owner and kind/time indexes support these queries. A stale claim is an observation, not a reason to retry automatically. Duplicate configured API tokens are rejected so nominally different principals cannot share an indistinguishable authentication credential.

### 9.2 Reviewed operation versus strict release

The deployment has two explicit policies. The default reviewed policy permits organizational report authorization after current source review, interpretation resolution and determinate reasoning. Missing external release/case evidence remains advisory and visible. Such responses identify reviewed_operation, preserve override_used, and record the chosen policy in the audit event. They do not claim that external validation succeeded.

Strict policy additionally requires a pinned, unexpired release-evidence graph and a favorable independently signed exact-case assessment. The case target binds the source, formalization, current reviews, facts/actions and explicit actor, date, jurisdiction and capacity. Registry changes and case-target changes are checked again on high-stakes evaluations, including idempotent replays. Historical receipts remain historical.

**Equation 6.** Organizational authorization policies, with H denoting a high-stakes report.

EQUATION: Reviewed: H ∧ R. Strict: H ∧ R ∧ D ∧ J.

Here R denotes current research readiness, D deployment evidence and J the independently signed case predicate. Non-high-stakes research evaluations return results without production authorization. The choice between policies is deployment-owned; callers cannot downgrade it per request. Neither predicate is a theorem of legal correctness. In particular, choosing reviewed operation relaxes an organizational gate; it does not supply the missing empirical evidence.

### 9.3 Deployment and recovery

Reference deployment uses a non-root, read-only application container, separate worker and PostgreSQL, loopback-bound host exposure and externally configured TLS/network admission. PISTE/model/trust settings are passed explicitly. The service rejects missing production credentials when the relevant source transport is enabled and refuses automatic schema creation. The current schema revision is checked, not only database connectivity.

Executable probes cover authenticated ingestion/review/evaluation/audit export, persisted-evaluation load, both configured legal corpora and isolated database restoration. Restore verification compares governed-row hashes and the audit chain against a nonempty frozen baseline. CI performs a real server-version-matched PostgreSQL dump/restore drill. Deployment-specific backup custody, alert routing, RPO/RTO, sustained traffic and incident response remain operational responsibilities.

Audit archives use deterministic JSON entries and content digests. Audit events form a chained history; external anchoring is needed for stronger tamper accountability. The archive's own hashes cannot authenticate themselves. No DELETE or UPDATE API is supplied for governed evidence, and any legally required retention/deletion process needs a separately governed administrative procedure.

### 9.4 Scoped release-candidate preparation

Version 0.9.0 introduces an opt-in strict supervised French/U.S. IP profile while preserving the legacy deployment contract. A v2 pinned evidence graph binds the exact build/model revision, signed held-out annotations, predictions and judicial review protocol, all three direct/symbolic/gold disagreements, recent full-text provider receipts, and real staging smoke/load/restore reports. Quality accounting retains abstentions and false compliance per language, modality, exception presence and source kind. Custodians must verify the underlying experiment and reviewer qualifications; the software checks attestation integrity and arithmetic.

The profile requires a supported imported source type, upstream provenance and matching jurisdiction in addition to the existing strict predicates. Judicial holdings, dicta, quotations, separate opinions, precedential status, dates and source changes require explicit independent assessment. A source receipt establishes retrieval, not binding authority. Read-only readiness diagnostics report missing prerequisites and remain unauthorized even when supplied prerequisites pass. The default zero-observed-false-compliance budget is a configurable organizational policy, not a zero-error population guarantee.

Distributions contain the versioned schemas, examples, regression corpora and migration scripts. Benchmark defaults and forward-only migration commands work outside a checkout. CI rebuilds a wheel from the sdist under a fixed build timestamp and checks identical bytes in the same environment, then executes the installed package from an unrelated directory. This establishes a bounded reproducibility check, not universal reproducible builds or a signed supply-chain guarantee. Real credentials, expert annotation, staging qualification, licensing and enforced repository protection remain external release actions.

## 10. Evaluation and reproducibility

### 10.1 Evidence actually obtained

The evaluated runtime-code snapshot is version 0.9.0, commit `e54d2c81b6107a62fdf21a25b4d4ba24093af7de`, tree 2d2d78075eb18f534901f8eaeccb8340b379da71. The exact code/resource hashes and regression commands are retained in RC_EVALUATION.json and the paper evaluation record; the earlier 0.6.2 record remains archived. Python 3.12.14 completed 461 tests with four PostgreSQL-specific skips and one existing Starlette/httpx warning. Ruff, Bandit, schema/YAML validation, a wheel-from-sdist byte comparison and installed-package/migration checks passed locally. The repository's CI additionally exercises Python 3.13, PostgreSQL and container/dependency gates. Deployment-specific qualification and live expert/model evaluation were not performed. The current paper is a separate documentation update to that runtime snapshot.

| Evaluation component | Configuration | Observed result | Supported interpretation |
| --- | --- | --- | --- |
| Local automated tests | Python 3.12; current source snapshot | 461 passed; 4 skipped | Covered code behavior agrees with tests |
| Semantic regression corpus | Exhaustive backend | 62 / 62 expected labels | Synthetic temporal/verification/reasoning behavior |
| Semantic regression corpus | Z3 backend | 62 / 62 expected labels | Cross-backend agreement on this corpus |
| Completion/amendment experiments | Exhaustive backend | 19 / 19 expected labels | Bounded uncertainty and inclusion behavior |
| Completion/amendment experiments | Z3 backend | 19 / 19 expected labels | Cross-backend agreement on these experiments |
| Interpretation-family corpus | Finite product enumeration | 15 / 15 expected analysis labels | Synthetic family semantics, explicitly unreviewed |
| Argumentation invariant check | Every directed graph on three arguments, including self-attacks | 512 graphs checked | Small-graph semantic invariants |

The two backend runs reuse the same corpora; they are not independent legal samples. Only verification cases that use the configurable backend switch to SMT; temporal evaluation, argumentation and shared reasoning cases retain their corresponding implementations. Thus 62/62 is a corpus outcome, not 62 independent SMT proofs. The 512 graphs are internal cases within a test, not 512 additional pytest tests. Family fixture labels concern the nested computational analysis even when the top-level outcome remains review_required. No shipped fixture pretends to have independent expert approval.

Tests also cover malformed formulas, horizon boundaries, nonvacuous assumptions, resource limits, replayed counterexamples, preference cycles, stale reviews, draft self-grounding, behavior ambiguity, source-span tampering, signature revocation, jurisdiction restrictions, cache policy changes, worker interruption and runtime drift. HTTP inference tests use local deterministic provider fixtures. They test the protocol and control path, not the quality of a trained language model.

The repository's CI additionally runs Python 3.12/3.13 matrices, semantic benchmarks, PostgreSQL migration/immutability/concurrent-claim checks, real isolated restore verification, container smoke tests, dependency/security scans and final-image inventory generation. The paper's local results do not by themselves assert the outcome of a future CI run or your deployment's acceptance tests. CI status is a versioned engineering artifact, not a legal evaluation result.

### 10.2 Resource and semantic limits

The temporal AST caps horizon at 128 states, nodes at 512 and depth at 48. The general verifier caps atoms at 64 and reports unknown when declared resource limits prevent a conclusion. Interpretation-family analysis supports up to 16 interpretations, up to eight behavior alternatives, 12 factual queries, and explicit completion/world/planning bounds. Defaults include 1,024 completions, 8,192 worlds and 10,000 planning states. Timeout is shared within a family analysis; a revision sequence can perform several separately bounded cycles.

These caps are safeguards, not evidence of useful throughput on large legal models. Enumerating k unknown Boolean cells can require 2^k completions, and adaptive planning can visit many world subsets. SMT may avoid some explicit enumeration but does not remove worst-case complexity or guarantee termination inside a time budget. Corpus-scale retrieval, temporal abstraction and symbolic family analysis remain scaling research questions.

### 10.3 Reproduction procedure

Install the project's dev, verification and annotation extras in a supported Python environment; run pytest and the configured lint/security gates. Execute `benchmark --backend exhaustive`, `benchmark --backend z3`, both experiment-benchmark variants and family-benchmark through `python -m src.cli`. Preserve the JSON results rather than only their headline success counts. The paper source and build script are committed so the PDF can be regenerated without an external document service.

For model studies, additionally freeze the source versions, prompt templates, weights revision or digest, server release, tokenizer/chat template, inference parameters, review policy, candidate hashes and split lineage. A model's served name is not an immutable weights identifier. For independent annotation, preserve the original separate submissions, enrollment records, adjudication and external trust anchors. For operational tests, retain configuration, workload, database baseline and restore evidence for the actual deployment.

### 10.4 Frozen Phase 5 results, including unavailable and negative outcomes

| Protocol | Observation | Interpretation |
| --- | --- | --- |
| F1 | 32/32 planted minimal cyclic families detected; all pairs satisfiable | Synthetic Boolean constraint examples only |
| F2 | Unqualified acyclic pair-satisfiability claim refuted | Corrected join-tree/semijoin shortcut tested against exact checking |
| F3 | Not run: no eligible reviewed Légifrance corpus | Counts are unavailable, not an empirical zero |
| H2 | Baseline 6/14 violations plus 2 inconclusive; treatment 3/14 plus 2 inconclusive | Token-boundary repair retains original fixture results; 2/2 planted fake faults detected |
| H3 | H0 MSE 0.20565; mean distance 0.17836; alternative count 0.46113 | H0 fails to beat both baselines; no real reviewer predictive validity |
| H4 | Wrong-model artifact escape 1/3 unpinned, 0/3 with trusted query pin | Runtime mutants also include unknown outcomes and a no-certificate case |
| H5 | Eight simulated tasks; 1 improvement and 4 ties among five equal-cost resolvable fixed-order comparisons | Distributions and insufficient-question cases retained; not a real reviewer study |

The protocol, dated addendum, per-case outputs and negative results remain in the Phase 5 artifacts. No thresholds were retuned to manufacture empirical support. Subsequent automatic acquisition does not turn unreviewed data into eligible reviewed data for F3. The current release-profile tests use explicitly synthetic/unreviewed graphs to exercise software branches, never supplied production evidence.

## 11. Threats to validity and research agenda

All Phase 4/5 results above remain synthetic and unreviewed. Consensus, round-trips, bounded repairs, diagnostic geometry and independent replay do not establish source faithfulness, demonstrated originality or legal certification. There is no Lean project and no mechanized correctness claim. The wrong-model/query-binding limitation and the negative/unavailable research outcomes are retained rather than omitted from the release narrative.

Construct validity is the principal limitation. Synthetic fixtures establish that the implementation follows selected formal examples; they do not establish faithful interpretation of French intellectual-property or competition law. Rule-based extraction may miss implicit norms, scope-bearing definitions, judicial interpretation or interactions among provisions. A source quote and content hash can be exact while the associated legal predicate is wrong.

Internal validity also depends on the modeling boundary. Response-action mappings may miss indirect assistance, pragmatic implications or future tool effects. Boolean facts abstract away evidence quality, uncertainty and correlated testimony. Independent expert review can expose these problems but is itself subject to disagreement and contextual limits. Signatures make submissions attributable; they cannot eliminate shared assumptions or organizational conflicts.

External validity is presently unmeasured. No qualified independent expert corpus, live configured-model evaluation, field deployment study or matched-budget comparison is reported for this project. Public-source availability does not establish complete legal coverage. Operational probes and the CI restore exercise cannot substitute for organization-specific security review, legally appropriate retention and reliable backup custody.

A first empirical study should compare a statute-conditioned revision baseline with versioned source retrieval, single-model verification, interpretation-family analysis and factual clarification as separate ablations. Use the same requests, source scope and inference budgets; include unrouteable cases, abstentions and resource limits in denominators. Freeze legal-source/amendment lineages before prompt development. Two independent experts and third-party adjudication should label representational faithfulness, omitted conditions/exceptions, unsupported facts, harmful assistance, usefulness and refusal appropriateness separately.

A second study should measure whether explicit alternative interpretations reduce expert overturns and unintended consequences relative to selecting one formalization. Predeclare what constitutes a material disagreement and report unresolved families. A third should compare minimax clarification with uniform or heuristic questioning under the same admissible queries, then extend to noisy evidence with explicit error models. The current exact planner provides a small-instance oracle, not a solution to that noisy problem.

Further formal work could establish a verified translation from the typed norm model to temporal formulas, characterize abstraction soundness, develop richer contrary-to-duty and recurring-obligation fragments, and support quantified domains. Incremental amendment analysis should connect source lineage to changed interpretations and prioritize re-review. Standards adapters should be evaluated through round-trip and behavioral equivalence properties rather than format validity alone.

## 12. Conclusion

legal_norm_argumentation supplies an inspectable path from normative source snapshots to bounded reasoning, reviewed model revision and governed service execution. Its central design is to keep legal interpretation, factual evidence, behavior hypotheses and deployment authorization distinguishable while binding each to reproducible artifacts. The current evidence supports software-level claims on bounded synthetic tasks and tested service paths. Establishing legal faithfulness and dependable field performance requires the independent annotation and deployment studies specified above. Making those obligations visible is part of the framework's contribution, rather than a gap that can be removed by relabeling a passing test.

## Reproducibility, authorship and disclosure

Repository: https://github.com/joedimos/legal_norm_argumentation. Runtime-code snapshot: `e54d2c81b6107a62fdf21a25b4d4ba24093af7de`; version 0.9.0. The accompanying paper update is a separate documentation commit. The archived 0.6.2 evaluation remains available. This is an academic-style research-software manuscript, not a claim of acceptance by LEGATO, VLE, JURIX or any publisher. The paper and software were developed with ChatGPT/Codex assistance; the manuscript is grounded in repository inspection, executed regression tests and the cited primary references. No independent legal study participants, institutional affiliations, funding awards, ethics approvals or expert endorsements are asserted. The author should review the manuscript and arrange any required consent, permissions and institutional review before an empirical study or submission.

## References

[1] Dung, P. M. (1995). On the acceptability of arguments and its fundamental role in nonmonotonic reasoning, logic programming and n-person games. Artificial Intelligence, 77(2), 321-357. https://doi.org/10.1016/0004-3702(94)00041-X

[2] De Giacomo, G., and Vardi, M. Y. (2013). Linear Temporal Logic and Linear Dynamic Logic on Finite Traces. Proceedings of IJCAI 2013, 854-859. https://www.ijcai.org/Proceedings/13/Papers/132.pdf

[3] de Moura, L., and Bjorner, N. (2008). Z3: An Efficient SMT Solver. TACAS 2008, LNCS 4963, 337-340. https://doi.org/10.1007/978-3-540-78800-3_24

[4] OASIS (2021). LegalRuleML Core Specification Version 1.0. OASIS Standard, 30 August 2021. https://docs.oasis-open.org/legalruleml/legalruleml-core-spec/v1.0/os/legalruleml-core-spec-v1.0-os.html

[5] Delage, C., Canu, S., Decombas, M., and Foureur, J. (2026). Statutory AI: Aligning Large Language Models With Legal Norms. Artificial Intelligence for Digital Transformations, CCIS 3023, 375-388. https://doi.org/10.1007/978-3-032-31319-5_25. Related manuscript: https://arxiv.org/abs/2608.28593

[6] Tabassi, E. (2023). Artificial Intelligence Risk Management Framework (AI RMF 1.0). NIST AI 100-1. https://doi.org/10.6028/NIST.AI.100-1

[7] Josefsson, S., and Liusvaara, I. (2017). Edwards-Curve Digital Signature Algorithm (EdDSA). RFC 8032. https://www.rfc-editor.org/rfc/rfc8032

[8] PostgreSQL Global Development Group. PostgreSQL 18 Documentation: Explicit Locking. https://www.postgresql.org/docs/18/explicit-locking.html

[9] Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales. Educational and Psychological Measurement, 20(1), 37-46. https://doi.org/10.1177/001316446002000104

[10] Free Law Project. CourtListener Case Law APIs, REST API v4. Official documentation, accessed 29 October 2026. https://wiki.free.law/c/courtlistener/help/api/rest/v4/case-law

[11] An, C., Bayless, S., Buliani, S., et al. (2026). A Neurosymbolic Approach to Natural Language Formalization and Verification. arXiv:2511.09008v2, revised 13 July 2026. https://arxiv.org/abs/2511.09008

[12] Horner, E., Mateis, C., Governatori, G., and Ciabattoni, A. (2025). Toward Robust Legal Text Formalization into Defeasible Deontic Logic using LLMs. arXiv:2506.08899v3, revised 31 December 2025. https://arxiv.org/abs/2506.08899


## Native-backend implementation addendum (0.9.3)

The optional PyReason backend now computes a bounded relational/temporal interval
fragment and supplies exact derived facts to typed norm activation, priority and
compliance. An independent finite oracle checks every projected query; disagreement
or a bound hit blocks the native result. A reusable bounded worker and a non-root
read-only native image support deployment. Separate native-engine jobs exercise
162 synthetic activation contexts, temporal expiry/chaining, relational joins,
soft intervals and seeded/unseeded cycles. Mixed 1/8/32/64-norm timings preserve
startup and all observations separately. The core is faster on small flat examples;
no speedup claim is made. These are implementation checks on synthetic, unreviewed
data, not independent legal validation. The earlier evaluation counts in this
manuscript remain historical snapshots, not counts for this addendum.

Native organizational release additionally requires runtime-specific evidence,
independently signed held-out IP judgments, representative persisted load and all
existing exact-case legal review gates. Content hashes do not establish experimental
truth. No independent real-IP qualification, legal accuracy, certification or
Python formal-verification claim follows from these additions. The supported
fragment deliberately excludes arbitrary upstream annotations/neural functions.
