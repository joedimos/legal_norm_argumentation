# Légifrance ingestion

VLE–LEGATO can ingest public French statutory material directly from **Légifrance**, the official public service for dissemination of French law. The implementation is intentionally narrow: it does not accept arbitrary URLs and it does not treat downloaded text as automatically faithful formal law.

## Supported corpora

Two fixed legislative corpora are exposed:

- `ip` — **Code de la propriété intellectuelle** (`LEGITEXT000006069414`), including copyright and industrial-property provisions exposed by the legislative code page.
- `competition` — **Code de commerce, Livre IV: De la liberté des prix et de la concurrence** (`LEGITEXT000005634379`, `LEGISCTA000006113741`).

The collector discovers article links only under `https://www.legifrance.gouv.fr/codes/article_lc/`. Redirects are revalidated against the same HTTPS host and `/codes/` namespace, preventing the ingestion endpoint from becoming a general-purpose URL fetcher.

## Web console

Run the production service and open `http://localhost:8000/`:

```bash
vle-legato-api
```

or:

```bash
uvicorn src.service.app_with_ui:app --host 0.0.0.0 --port 8000
```

The console supports:

- corpus selection for IP or competition law;
- bounded article batches and offsets;
- authenticated collection using an operator API key;
- source/version registry browsing;
- candidate counts and human-review status;
- source provenance and extracted norm inspection.

The browser stores the API key in `sessionStorage` only. No key is embedded in the application bundle and no third-party JavaScript is loaded.

## API

Public catalog:

```http
GET /v1/legifrance/corpora
```

Authenticated source registry (`reader`, `operator`, `reviewer` or `auditor`):

```http
GET /v1/sources?prefix=legifrance:&limit=100
Authorization: Bearer <token>
```

Authenticated ingestion (`operator`):

```http
POST /v1/legifrance/scrape
Authorization: Bearer <token>
Content-Type: application/json

{"corpus":"competition","limit":20,"offset":0}
```

A changed article creates a new source version. An unchanged article is not duplicated. Both outcomes are written to the existing audit chain. Légifrance provenance is stored in `extraction_audit.provenance` with the provider, corpus, article identifier, official URL and retrieval mode.

## Unattended synchronization

The maintenance command can be scheduled by cron, systemd or Kubernetes CronJob:

```bash
# One bounded page from each corpus
vle-legato-maintenance --sync-legifrance all --limit 50

# Walk every discovered article in both corpora, page by page
vle-legato-maintenance --sync-legifrance all --limit 50 --full

# Resume from an article offset
vle-legato-maintenance --sync-legifrance ip --limit 50 --offset 500 --full
```

Each page is fetched before its database transaction starts, then committed independently. A partially interrupted full synchronization can therefore be safely rerun: unchanged text is recognized by `(source_key, content_hash)` and is not duplicated.

## Safety bounds

Environment controls:

```text
VLE_LEGIFRANCE_ENABLED=true
VLE_LEGIFRANCE_TIMEOUT_SECONDS=12
VLE_LEGIFRANCE_REQUEST_DELAY_SECONDS=0.2
VLE_LEGIFRANCE_MAX_ARTICLES_PER_RUN=50
VLE_LEGIFRANCE_MAX_FULL_SYNC_ARTICLES=2500
```

Additional protections include:

- HTTPS-only fixed host validation;
- fixed `/codes/` namespace validation;
- article-only link acceptance during discovery;
- manual redirect validation rather than unrestricted redirect following;
- retry/backoff for transient `429` and `5xx` responses;
- response-size limits;
- per-request article limits and full-sync total bounds;
- hash-based versioning and database uniqueness constraints;
- existing API-key rate limiting for interactive ingestion.

## French norm extraction

The rule-based extraction layer now recognizes French obligation, prohibition and permission forms such as:

- `doit`, `doivent`, `est tenu de` → obligation;
- `ne doit pas`, `est interdit`, `sont prohibées` → prohibition;
- `peut`, `peuvent`, `est autorisé` → permission;
- `lorsque`, `si`, `dès lors que` → conditions;
- `sauf`, `sauf si`, `à l'exception de`, `sous réserve de` → exceptions.

These candidates remain **reviewable hypotheses**, not certified legal formalizations. The existing human-review, ambiguity, interpretation-selection, priority, temporal/deontic reasoning and audit gates continue to apply.

## Operational note

Public Légifrance HTML is an external dependency. The collector fails visibly if it cannot discover article links or extract enough legal text, rather than silently treating a layout failure as an empty corpus. CI uses deterministic fixtures; live collection should additionally be monitored in deployment because the external page structure can change independently of this repository.
