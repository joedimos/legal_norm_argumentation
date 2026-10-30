# Native integration qualification protocol — 2026-10-29

This is an engineering protocol, not a legal study or a claim of novelty.
All generated cases remain synthetic and unreviewed. An initial single-norm
latency smoke run is exploratory; the qualification workload below was specified
before examining mixed-workload results. No speedup is assumed.

1. Identify the installed PyReason package by actual package source hashes,
   dependency versions, Python version and adapter contract hash. A stand-in
   module must fail this probe.
2. Compare native activation on all 162 assignments of four three-valued
   condition/exception facts and two negation choices, in batches of at most 32.
3. Execute recorded native graph programs covering delay, expiry, same-tick
   chaining, relational joins/new edges, soft intervals, exact false and true.
   Compare all requested answers to a separate finite interval-closure oracle.
4. Demonstrate a derived fact changing a normative verdict through the public
   evaluation pipeline. Conflicting asserted/derived observations must block.
5. Measure 1-, 8-, 32- and 64-norm workloads cyclically. Record every latency,
   the comparison-core latency, startup separately, p50/p95/max, and whether
   native is faster. Warm sample timing follows an explicit startup probe;
   native deployments must warm before readiness. Report slower results too.
6. Use explicit iteration/time caps. Every check must complete; mismatch,
   truncation, timeout or missing native engine prevents qualification success.
7. Native CI runs without provider/model/network calls during analysis. It is
   a local symbolic engine, not a live LLM. Recorded/fake model tests remain
   separate. Native image tests run as non-root with a read-only root filesystem.
8. Synthetic software success cannot satisfy legal release qualification.
   Release also requires independently signed, held-out IP judgments, externally
   pinned custodial evidence, and the existing exact-case review/release gates.
   No reviewed corpus or expert judgment is fabricated or relabelled here.
