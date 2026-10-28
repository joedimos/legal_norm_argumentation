# Inference readiness and attributable independent legal judgments

The inference path supports native Ollama and vLLM's OpenAI-compatible Chat
Completions endpoint. The signed legal-review path authenticates registered public
keys against a **separately pinned** registry. Neither path supplies legal experts,
proves the truth of their judgments, or authorizes deployment.

## Configure and diagnose a model

Set these variables in the shell running the CLI (an `.env` file is not loaded by
this CLI). Request fields override environment values.

```bash
export VLE_MODEL_PROVIDER=vllm
export VLE_MODEL_NAME=YOUR_EXACT_SERVED_MODEL_ID
export VLE_MODEL_ENDPOINT=http://localhost:8000/v1/chat/completions
# Set VLE_MODEL_API_KEY through your secret manager if the server requires it.
python -m src.cli model-doctor --input examples/model-doctor.json \
  --bundle model-diagnostic.zip
```

`examples/model-doctor.json` uses environment configuration. Ollama defaults to
`http://localhost:11434/api/chat`; vLLM defaults to
`http://localhost:8000/v1/chat/completions`. For a remote server, use HTTPS, or
forward its loopback listener over an SSH connection you control. Do not expose
an unauthenticated inference service publicly. No runtime or model is downloaded
or launched by these commands.

The diagnostic reads `/api/tags` or `/v1/models` at the same origin. It reports
`model_available`, `model_not_listed` or `runtime_unavailable`. It never generates
text, sends the legal corpus or asserts that inference works. An Ollama inventory
digest is recorded when advertised; vLLM model IDs are not weights digests. Retain
server versions, weights hashes, tokenizer/chat template and launch configuration
separately for reproducibility.

For actual inference, set up a reviewed family as described in
[LIVE_GENERATION_ANNOTATION.md](LIVE_GENERATION_ANNOTATION.md), then run:

```bash
python -m src.cli live-generate --input examples/vllm-generation.json \
  --bundle generation-evidence.zip
```

That example omits `model` and `endpoint`, using the environment. It contains a
synthetic **unreviewed** family, so it consumes zero calls until actual source
review is supplied. Never create fictional receipts to bypass that requirement.
A successful inference still yields an unreviewed draft/mapping; inspect and
review it before resuming a revision sequence.

vLLM requests use JSON-object mode, `n=1`, `stream=false`, `temperature=0`, and
`max_tokens`; Ollama uses its native JSON mode and `num_predict`. Both go through
the same semantic validation. vLLM requires one assistant message, a `stop` finish
reason, no tool calls/refusal, and an exact returned model ID. No API call executes
model-proposed tools. No fallback converts a truncated completion into a draft.
JSON-object mode does not establish schema or legal correctness.

Transport rejects redirects and compressed responses, caps response bytes at
1 MiB, and reports sanitized error codes for authentication, permissions,
rate limits, timeout, unavailable runtime and malformed output. It does not retry
or switch providers. Timeout bounds socket operations; it does not promise remote
cancellation or a hard inference wall-clock limit. Temperature zero is not a
reproducibility guarantee. The existing candidate-review/budget stop rules remain.

Protocol references: [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/),
[vLLM server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/),
[Ollama inventory](https://docs.ollama.com/api/tags).

## Collect real legal judgments

Prepare a frozen blinded packet using `annotation-prepare`. For each actual
reviewer, make a separate blank submission:

```bash
python -m src.cli annotation-template --input packet.json \
  --reviewer ACTUAL_PSEUDONYMOUS_ID --qualification 'Actual relevant qualifications' \
  > reviewer-submission.json
```

`packet.json` must be the packet itself, without bundle metadata added to stdout.
If needed extract `request.json`/`result.json` from the verified audit ZIP rather
than copying an edited JSON object. Null labels and independence attestations in
the template intentionally fail assessment until the reviewer completes them.
The reviewer must work without access to the other submission or solver verdicts.
Preserve alternative interpretations and the original separate judgments.

Optional `evidence_refs` in plain judgments is an array of exact
`{document_id,start,end,quote}` spans into the frozen unit documents, using
half-open Unicode character offsets. Every supplied reference is validated.
**Signed** judgments must explicitly contain this array. A determinate
applicability/faithfulness/conditions judgment needs a normative quote; a
determinate response-behavior judgment needs a draft quote; a `supported` factual
grounding judgment needs an observation quote. Quotation integrity does not
establish that the cited passage entails the judgment. Uncertain judgments may
have an empty array and remain uncertain after signing.

## Sign on each reviewer's machine

```bash
pip install -e '.[annotation]'
python -m src.cli annotation-keygen --private-key-file reviewer.reviewer.key \
  --public-key-file reviewer.pub
python -m src.cli annotation-sign --input reviewer-submission.json \
  --role annotator --packet-sha256 FROZEN_PACKET_SHA256 \
  --private-key-file reviewer.reviewer.key > signed-review.json
```

Key generation refuses overwrite and writes the private key with owner-only
permissions. Keep it off shared storage and out of the repository. The coordinator
needs only the public key. Private keys are raw unencrypted Ed25519 seeds encoded
as lowercase hex; secure storage, backups and endpoint protection are the owner's
responsibility. The signer signs a domain-separated canonical JSON envelope with
role, frozen packet hash, full payload and UTC timestamp. It does not approve or
complete a judgment. Do not have the coordinator hold both reviewers' private keys.

## Pin a reviewer registry outside the submission

A study custodian must verify identity, expertise, public-key possession and
conflicts through the study's actual enrollment process. Create a
`vle-reviewer-registry/v1` with `reviewers` and nonempty
`excluded_independence_groups` naming the generating/development organizations.
For each reviewer register exactly:

| Field | Content |
| --- | --- |
| id | Matches the submission's pseudonymous reviewer ID |
| public_key | 32-byte Ed25519 public key as 64 lowercase hex characters |
| roles | Nonempty subset of `annotator`, `adjudicator` |
| jurisdictions | Actual approved jurisdiction codes, such as `FR` |
| independence_group | Custodian-defined organization/conflict group |
| valid_from, valid_until | Inclusive canonical `YYYY-MM-DD` dates |
| revoked | Explicit Boolean; revoked keys cannot verify new assessments |
| credential_check | `checked_by`, UTC `checked_at`, and `evidence_sha256` |

The evidence digest refers to externally retained enrollment/qualification
records; the CLI does not retrieve or independently validate those records.
Keep personally identifying evidence under appropriate study access controls.
Use different identities, keys and independence groups for both annotators and
any adjudicator. A registered group excluded from this study cannot participate.
A registry is not a certificate authority or proof of genuine expertise.

Freeze the registry's canonical SHA-256 using `digest(canonical_json(registry))`
from `src.audit_bundle`. Distribute that digest, and the frozen packet digest,
through a trusted channel separate from the signed submission. Obtain the
current registry and pin after revocation/credential changes; an old trusted
snapshot cannot reveal a later revocation.

A `vle-signed-assessment/v1` input has `packet`, two signed `submissions`, and
optional signed `adjudication`. Adjudication uses role `adjudicator`; its plain
payload follows the existing adjudication schema. Its `submission_sha256s` bind
the ordered **plain payloads**, not the signature envelopes. The envelope also
binds the entire frozen packet. A third registered reviewer must sign it.

```bash
python -m src.cli annotation-verify --input signed-assessment.json \
  --registry trusted-reviewers.json \
  --expected-registry-sha256 TRUSTED_REGISTRY_SHA256 \
  --expected-packet-sha256 FROZEN_PACKET_SHA256 \
  --bundle independently-signed-evidence.zip
```

Verification rejects signature tampering, unknown/revoked keys, wrong role or
jurisdiction, future signatures, keys outside signing/current validity periods,
reused keys/identities, prohibited groups and nonindependent registered groups.
It preserves signatures, public registry, original judgments, disagreements,
adjudication and descriptive metrics in the audit bundle. Original study labels
are never replaced by solver predictions.

`key_signatures_verified` means attribution to enrolled keys under the supplied
trust anchor. `qualifications_verified`, `independence_verified`,
`reviewers_authenticated` (real-world people), `deployment_authorized` and
`legal_certification` remain false: software cannot verify expertise or actual
independence from a signature. `annotations_collected` includes uncertain/adverse
judgments; it is not legal approval. Signed judgments never mint `family-review`
receipts or silently authorize inference. Human source/mapping review and
organizational release decisions remain explicit separate steps.

The crypto backend is optional; research CI installs and tests it. Base inference
and unsigned annotation work without it. Cryptographic primitive reference:
[PyCA Ed25519](https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ed25519/).
