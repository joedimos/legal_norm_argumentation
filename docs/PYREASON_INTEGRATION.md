# PyReason graph-diagnostic integration contract

For the required normative activation backend introduced in 0.9.2, see
[PYREASON_NORMATIVE_BACKEND.md](PYREASON_NORMATIVE_BACKEND.md). The contract below
describes the legacy graph-annotation path.

Upstream inspected: https://github.com/lab-v2/pyreason source tree
`633e4074c2cd99c6cf06b835b12aa0f4d8f5cc08` (2026-10-29).
The inspected setup.py identifies version 3.8.0. Fact/Rule keyword constructors,
reset/reset_rules/reset_settings, persistent/store_interpretation_changes settings,
reason(timesteps=...) and Interpretation.get_dict() inform this adapter.
A real PyReason 3.8.0 smoke run on Python 3.12 verified input [1,1], an
attack-derived [1,1] defeat, and unknown [0,1] preservation. Cold JIT required
an extended 300-second budget; the corrected warm run passed within 60 seconds.
This small synthetic/unreviewed check is not full-engine qualification.

Install explicitly with `pip install -e '.[pyreason]'`, or use
`pip install -r requirements-pyreason.txt` from the repository root. Both paths
select 3.8.0 on the toolkit's Python 3.12/3.13 range. Other backend versions are
rejected before their import-time initialization. The optional dependency is
never imported by the verified core. Existing `OptimizedAAF.run_pyreason()` callers
retain their interface. PyReason uses interval annotations, not calibrated
probabilities. These diagnostics do not implement defeasible legal priorities,
deontic modalities, temporal legal deadlines, or Dung extension semantics.

## Semantics and safety

Legacy `accepted` is the supplied annotation; it is not a computed legal
acceptance. `defeated`, `reinstated` and `strengthened` are positive graph-rule
signals and can coexist. Reinstatement does not retract defeat. No signal
changes review status, authorizes a decision, or enters the compliance evaluator.
Explicit unknown inputs and missing trace predicates are `[0,1]`; the connector
no longer invents a confidence interval when callers omit it. Existing
LegalArgument defaults remain unchanged for compatibility.

Safe, deterministic node symbols prevent source identifiers from becoming rule
syntax. Each invocation starts a subprocess, resets upstream settings and state,
and uses persistent annotations over the bounded horizon. Persistent mode is
appropriate only for this static graph diagnostic; it is not legal persistence.
Facts cover steps 0 through the requested horizon inclusively. Early engine
termination is not a fixed-point proof. `converged` remains false in every result.

Limits: 256 arguments, 4,096 edges per relation, horizon 1–100, output 2 MB,
wall-clock timeout 1–300 seconds (default 60, including import/JIT). Resource
failures, invalid intervals and malformed/incomplete receipts return unknown with
empty annotations; timeout reports its bound. The subprocess isolates global
state and enforces a deadline, but is not an operating-system security sandbox or
a hard RAM quota. Deploy within existing container resource limits.

Reports include canonical input SHA-256, bounds, engine version on observed runs,
unreviewed status, and false authorization/certification fields. The input hash
binds normalized graph annotations, horizon and rule-set hash. A separate
argument-map hash binds source identifiers to normalized symbols. Together these
distinguish renamed graphs while retaining structural reproducibility. Worker
receipts must match the request hash and preserve supplied annotations exactly.
Identical inputs produce the same hashes. Runtime failures can differ across machines; they
never produce partial positive reports. Never treat a successful diagnostic as
semantic equivalence to the core reasoner.

## Validation

Offline synthetic/unreviewed fixtures check constructor calls, upstream-shaped
trace extraction, unknown preservation, identifier isolation, invalid intervals,
timeouts, incomplete output, deterministic encoding and no convergence claim.
Required CI performs no network or real-engine inference. Broader real-engine differential evaluation remains qualification work.
The original smoke receipt is in RC_0_9_1_EVALUATION.json; the public-wrapper
follow-up is in PYREASON_QUALIFICATION.json. The latter covers an attack chain,
support, unknown non-propagation, and a second isolated graph. It does not assert
equivalence to Dung semantics. The legacy originality/protection scores remain
keyword heuristics; their metadata now explicitly says they are not PyReason
results or independent legal judgments.

The installed-wheel verifier runs the actual adapter and child worker from
outside the checkout with an explicitly synthetic provider. This checks packaging
and process wiring without installing or running the real engine in CI.

The public entry points accept an optional deadline without changing positional
arguments: `aaf.run_pyreason(2, timeout_seconds=300)`,
`reasoner.analyze_legal_case(text, run_pyreason=True, pyreason_timeout_seconds=300)`,
and `annotator.annotate_originality(text, True, pyreason_timeout_seconds=300)`.
Use the longer budget only for controlled cold-start qualification; first import
may initialize upstream JIT caches and requires a writable upstream cache location.
An unwritable cache, missing dependency or deadline hit returns unknown. There is
no automatic retry, fallback verdict, or dependency download.
