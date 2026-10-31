# Provider boundaries — 0.9.5

This maintenance revision addresses reproduced failures in HTTP and JSON input
handling. It changes no reviewed benchmark fixtures, reasoning semantics, public
command names, HTTP routes or schemas. It grants no legal certification or
review approval.

## Reproduced failures

A loopback server delivering a JSON response one byte every 40 milliseconds
completed successfully after approximately 0.77 seconds despite a configured
0.2-second timeout. Socket inactivity timeouts are insufficient to bound an
entire operation. The upstream [urllib3 timeout documentation](https://urllib3.readthedocs.io/en/stable/reference/urllib3.util.html#urllib3.util.Timeout)
explicitly discusses slow streaming and DNS limitations.

The shared model parser also accepted `{"n":1e10000}` as an infinite float. This
bypassed its explicit rejection of the nonstandard NaN/Infinity tokens. Later
canonicalization could reject that value, but it should never enter analysis.

## Network lifetime and data bounds

`BoundedHTTPClient` sends each request to `src.http_worker` in a new process.
The parent controls the aggregate budget, kills and reaps a timed-out child, and
never replays that request. This covers otherwise blocking DNS, connection/TLS,
header parsing, streaming bodies and decompression. Process startup, parent
staging, receipt validation and acceptance checks consume the budget as well.
Model and legal-import wrappers recheck after JSON decoding; a late valid body
cannot become a successful result. This is an application deadline, subject to
OS scheduling and process-reaping overhead, not a real-time OS guarantee.

| Boundary | Limit |
| --- | --- |
| Active one-shot children | 4 per caller process; excess fails immediately |
| Request body | 16 MiB before base64 encoding |
| Private request envelope | 24 MiB |
| Response | 1 MiB for models; 8,000,000 bytes for U.S. IP imports; 40,000,000 bytes for French IP imports |
| Private receipt | 65,536 bytes |
| HTTP budget | Finite, positive, at most 120 seconds; connector settings retain their existing tighter limits |
| Import-batch budget | Existing configurable aggregate limit, up to 600 seconds, across multiple requests and waits |
| JSON depth | 64 levels |

Byte bounds apply to reads themselves. Legal responses are checked after
bounded decompression; model responses must use identity encoding. Non-success
responses are not drained. Redirects and transport retries are disabled.
Response receipts are bound to the canonical request hash and the body's exact
length and SHA-256; malformed or mismatched receipts fail closed. These hashes
detect corruption between local processes, not provider authenticity or legal
truth. TLS certificate verification remains enabled.

Request bodies and credentials are passed through a private temporary directory
(0700), with the request file restricted to 0600. Worker stdout/stderr are
suppressed. Only an allowlist of response headers and sanitized failure codes
crosses the process boundary. Temporary state is removed and the capacity slot
released on every ordinary failure or timeout. Hard host termination can still
leave temporary files; deployment volume lifecycle and access controls matter.
There is no persistent HTTP connection pool, so each request incurs process and
connection setup overhead. A caller must configure its budget accordingly.

## Durable behavior

Automatic imports use the existing allowlisted PISTE and CourtListener connectors.
A timeout discards the whole hydration batch; no partial source versions, success
timestamp or advanced cursor are committed. The durable scheduler records a
failure and resumes the same checkpoint after its cooldown. A 429 retains the
existing provider-wide Retry-After handling. The existing PISTE connector may
refresh an expired token once on 401 for its read-only methods; the transport
itself does not retry.

A model deadline produces `generation_failed` with `transport_timeout`. A claimed
generation remains terminal and is not automatically rerun, including after a
worker restart. An explicit audited retry remains a separate workflow. Killing
a local client does not prove that the remote provider stopped inference; this
is why automatic generation replay remains forbidden. Capacity and malformed
receipt failures likewise cannot produce a reviewed candidate or authorized
verdict. Imported sources and model candidates stay unreviewed.

## Shared JSON boundary

`src.json_boundary.strict_json` rejects duplicate keys, NaN/Infinity, floating
point exponent overflow, invalid UTF-8, lone Unicode surrogates and excessive
nesting. It accepts UTF-8 with an optional BOM and preserves valid Unicode and
canonical hashes. UTF-16/32 byte streams are not accepted at these UTF-8
boundaries. Existing `src.model_runtime.strict_json` imports remain supported.
Research CLI input, inline/file factual JSON, model responses, active legal-data
responses and native-worker envelopes use this parser. The native adapter
fingerprint includes the shared parser, so changed deployments require fresh
qualification evidence. Historical valid audit archives remain verifiable;
this change does not alter their canonical representation.

## Scope and verification

The hard network lifetime applies to the automatic IP import transport and model
provider/diagnostic paths. Legacy HTML/PISTE collectors and operational load
probes retain their existing transports. This revision does not claim every
network path has the new process boundary. It changes no legal review gates and
does not qualify real provider credentials, live model quality, independent legal
annotations or a production deployment.

Regression tests use synthetic/unreviewed JSON, local loopback HTTP servers and
injected failures. They cover slow headers and slow-drip bodies, no redirect or
retry, gzip and byte limits, corrupt receipts, killed/reaped children, released
capacity, late decoding, malformed JSON, atomic import recovery and terminal
generation failures. The research/import/native offline guard also blocks the
parent transport so a child cannot bypass its socket monkeypatch. CI runs the
same tests without a live provider and preserves an 80% coverage floor for each
provider boundary module. Existing semantic, experiment, family and faithfulness
benchmarks are rerun unchanged. Results establish implementation behavior on
these fixtures only.

Local validation on Python 3.12.14: `python -m pytest -q` passed 721 tests with 14
expected skips (ten separately selected native-engine tests and four PostgreSQL
checks). `VLE_RUN_NATIVE_PYREASON=1 python -m pytest -q tests/test_pyreason_native.py`
passed all ten actual-engine tests. The provider-boundary coverage run exceeded
90% in every selected module (HTTP parent/worker, JSON, model runtime and legal
transport). Ruff, Bandit, `python -m src.api_contract` and
`pip-audit -r requirements-prod.txt` passed. Existing benchmarks remained 62/62
semantic cases on each exhaustive/Z3 backend, 19/19 experiment cases on each,
15/15 interpretation families and the unchanged 12-case faithfulness report.
CI repeats these checks for the committed build, including Python 3.13, installed
package resources, PostgreSQL and native container qualification. Local Docker
and real-provider qualification were not performed.
