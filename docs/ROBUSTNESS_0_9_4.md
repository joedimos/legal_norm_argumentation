# Boundary hardening — 0.9.4

This maintenance revision addresses reproduced failure cases. It adds no legal
accuracy or certification claim, changes no reviewed benchmark cases, and keeps
existing commands, HTTP routes and valid result formats compatible.

## Worker I/O

Previously, JSON decoding used Python's default last-key-wins behavior: a receipt
containing both `"status":"unknown"` and `"status":"observed"` could be accepted.
Both transports and the private worker now use strict decoding. Duplicate keys,
NaN/Infinity, numeric exponent overflow, invalid UTF-8, excessive nesting and
non-object envelopes fail closed. Reads themselves are capped, rather than
relying on a prior file-size check. Input and output caps are two million bytes;
JSON nesting is limited to 64 levels. Source identifiers still never become
executable rule text. Existing canonical hashes for valid inputs are unchanged.

The producer caps receipts before writing them. Oversized input/output produces
an unknown result with a bound marker. Transport deadlines cover validation and
receipt parsing as well as native execution. A late valid receipt cannot become
a positive result, and capacity is released on every failure.

## Worker lifetime and readiness

The parent retires reusable workers before the child's idle deadline, with a
one-second safety margin. Both processes share the same idle/request limits.
Cleanup detaches private state first and removes temporary files even when the
child exits during shutdown or a reap fails. Requests are not replayed. Shutdown
reaping remains bounded at five seconds; computation is never authorized because
of cleanup or retry behavior.

Startup verifies the probe against the installed native fingerprint and checks
the aggregate deadline again after graph inference. Qualification also checks
adapter-fingerprint stability and records worker timeout bounds explicitly.
Neither incomplete nor late startup can make readiness positive.

## Evidence accounting

A correctly pinned, signed artifact can still contain malformed measurements.
The native release gate previously checked a latency population's length without
requiring it to be an actual numeric array. It now validates native and core
populations independently: finite nonnegative observations, integer counts and
workload sizes, exact sample counts, and summaries consistent with the raw data.
The established lower-median p50 and nearest-rank p95 conventions are preserved.
The reported speed comparison must match the observations. Startup and measured
samples must fit inside the recorded aggregate elapsed time and timeout.

Synthetic signing keys in regression tests demonstrate rejection even when hashes
and signatures are internally consistent. They are not independent expert data.
These checks establish accounting consistency, not experimental truth or reviewer
credentials. The existing source, legal-review, signature, scope and release gates
remain required. Historical well-formed 0.9.3 performance data still validates;
old build/runtime attestations cannot authorize a changed deployment.

## Graph edge cases

Observation-only programs are accepted by the graph interface, but PyReason 3.8.0
requires at least one rule. The adapter now installs a tautological identity rule
for that case. It adds no information and preserves exact true/false, soft and
unknown intervals. An independent finite oracle still checks all queried answers.
Real-engine tests exercise unary and binary observation-only programs.

Invalid timeout arguments now produce serializable unknown graph reports with a
null timeout bound, rather than echoing nonfinite or non-JSON Python values.
Standalone computations remain unreviewed and unauthorized.

## Verification

The regression suite includes corrupt receipts, hard byte caps, late replies,
idle retirement, child-exit/reap races, invalid timeout serialization, malformed
signed measurement artifacts and historical-report compatibility. Separate native
jobs use the actual pinned CPU engine; default tests use labelled offline fixtures.
CI retains per-module coverage floors, installed-package checks, unchanged
semantic benchmarks and the non-root/read-only/network-disabled native image test.
All new legal-shaped examples are synthetic and unreviewed.

Local validation: `python -m pytest -q` passes 641 tests with 14 expected skips
(ten separately selected native tests and four PostgreSQL checks). With
`VLE_RUN_NATIVE_PYREASON=1`, all ten actual-engine tests pass. The existing
exhaustive/Z3 benchmarks remain 62/62 semantic and 19/19 experiment cases;
interpretation families remain 15/15, and the 12-case faithfulness report is
reproduced. `python -m src.api_contract`, Ruff, Bandit and the pinned native
requirements audit pass. CI repeats validation for the exact committed build,
including default installs without PyReason and the hardened native image.
