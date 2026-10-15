# Légifrance ingestion

VLE–LEGATO can ingest public French statutory material from **Légifrance**, the official public service for dissemination of French law. The implementation is intentionally narrow: it does not accept arbitrary URLs and it does not treat downloaded text as automatically faithful formal law.

## Supported corpora

Two fixed legislative corpora are exposed:

- `ip` — **Code de la propriété intellectuelle** (`LEGITEXT000006069414`).
- `competition` — **Code de commerce, Livre IV: De la liberté des prix et de la concurrence** (`LEGITEXT000005634379`, section `LEGISCTA000006113741`).

## Provider strategy

Production deployments should use the official **PISTE Légifrance API**. PISTE exposes the LEGI consultation service through OAuth2 client credentials. The application calls `/consult/legiPart` for a fixed LEGI text identifier and an effective date, then reads structured `sections` and `articles` rather than scraping site navigation.

For competition law, the structured response is restricted to the known Livre IV section CID. If an API version returns a flattened tree, article `pathTitle` values are used as a conservative fallback and must identify both Livre IV and the competition-law title.

`VLE_LEGIFRANCE_SOURCE=auto` selects PISTE whenever both credentials are configured. Otherwise it falls back to the bounded public-HTML collector. HTML mode remains available for development and compatibility but should not be relied on as the primary production transport because public website requests may be rate-limited or refused independently of the public availability of the legal data.

The HTML fallback accepts article links only under `https://www.legifrance.gouv.fr/codes/article_lc/`. Redirects are revalidated against the same HTTPS host and `/codes/` namespace, preventing the ingestion endpoint from becoming a general-purpose URL fetcher.

## PISTE configuration

Register an application on PISTE and inject the credentials through the deployment secret manager. Do not commit them.

```text
VLE_LEGIFRANCE_SOURCE=auto
VLE_LEGIFRANCE_PISTE_ENVIRONMENT=production
VLE_LEGIFRANCE_PISTE_CLIENT_ID=<secret>
VLE_LEGIFRANCE_PISTE_CLIENT_SECRET=<secret>
```

Use `VLE_LEGIFRANCE_SOURCE=piste` to require the official API and fail configuration when credentials are absent. `html` explicitly selects the bounded HTML fallback. `sandbox` can be selected for the PISTE environment during integration testing.

Credentials are represented as Pydantic `SecretStr` settings. They are used only in the OAuth exchange and are never added to API responses, provenance, source text, audit events or UI state.

## Web console

Run the production service and open `http://localhost:8000/`:

```bash
vle-legato-api
```

or:

```bash
uvicorn src.service.app_with_ui:app --host 0.0.0.0 --port 8000
```

The console supports corpus selection, bounded article batches and offsets, authenticated collection, source/version browsing, candidate/review status, provenance and extracted norm inspection. The public corpus endpoint also reports the active provider (`piste` or `html`) without exposing credentials.

The browser stores the VLE API key in `sessionStorage` only. No key is embedded in the application bundle and no third-party JavaScript is loaded.

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

A changed article creates a new source version. An unchanged article is not duplicated. Both outcomes are written to the existing audit chain. Légifrance provenance is stored in `extraction_audit.provenance` with provider, corpus, article identifier, official URL and retrieval mode (`piste_api` or `public_html`).

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

```text
VLE_LEGIFRANCE_ENABLED=true
VLE_LEGIFRANCE_SOURCE=auto
VLE_LEGIFRANCE_TIMEOUT_SECONDS=12
VLE_LEGIFRANCE_REQUEST_DELAY_SECONDS=0.2
VLE_LEGIFRANCE_MAX_ARTICLES_PER_RUN=50
VLE_LEGIFRANCE_MAX_FULL_SYNC_ARTICLES=2500
```

Protections include fixed official hosts/endpoints, fixed corpus identifiers, OAuth credentials kept out of evidence, response-size limits, transient-error retry/backoff, per-run/full-sync article bounds, HTML redirect revalidation, hash-based source versioning, database uniqueness constraints, and the existing API-key role/rate-limit layer.

## French norm extraction

The rule-based extraction layer recognizes French obligation, prohibition and permission forms such as:

- `doit`, `doivent`, `est tenu de` → obligation;
- `ne doit pas`, `ne peut`, `est interdit`, `sont prohibées`, `illicite` → prohibition;
- `peut`, `peuvent`, `a le droit de`, `est autorisé` → permission;
- `lorsque`, `si`, `dès lors que` → conditions;
- `sauf`, `sauf si`, `à l'exception de`, `sous réserve de` → exceptions.

These candidates remain **reviewable hypotheses**, not certified legal formalizations. The existing human-review, ambiguity, interpretation-selection, priority, temporal/deontic reasoning and audit gates continue to apply.

## Operational note

PISTE is the preferred production transport because it is the machine-oriented official interface. Public HTML remains an external presentation surface and may change or reject automated requests. The collector fails visibly rather than silently treating a transport/layout failure as an empty corpus. CI uses deterministic structured and HTML fixtures so tests do not depend on external availability; live provider health should still be monitored in deployment.
