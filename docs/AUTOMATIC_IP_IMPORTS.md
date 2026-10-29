# Automated IP-source import (0.8.1)

The `importer` service acquires and refreshes French and American intellectual-
property source documents automatically. It uses the same PostgreSQL source
registry, extraction pipeline and append-only audit chain as the existing API.
No text-file uploads, manually selected article offsets, browser credentials or
live model are required for routine acquisition after one-time provider setup.

## Sources and scope

| Source | Automatic acquisition | Identity and provenance |
| --- | --- | --- |
| Légifrance / DILA PISTE | Dated manifest for the Code de la propriété intellectuelle; hydrate every selected article using its official ID | Existing `legifrance:ip:<article-number>` keys, official article/version ID, consultation date, raw payload and manifest hashes, FR/fr metadata |
| Free Law Project / CourtListener v4 | Opinion search for copyright, patents, trademarks and trade secrets, then retrieve complete opinions | `courtlistener:opinion:<opinion-id>`, cluster/court IDs, filing date, supplied citations, opinion type, raw payload hash, text-field and query/window provenance |

The French manifest excludes explicitly non-current articles and records that
count; it does not silently import only articles whose preview contains text.
It consults `/consult/legiPart` followed by `/consult/getArticle`, retaining the
cycle's date and ordered IDs across batches. Future scans discover changes and
new IDs. Acquisition currently covers the IP code, not all French IP judgments.
The legacy IP/competition manual routes remain available; automatic jobs focus
on IP and do not schedule the competition corpus.

CourtListener discovery uses `type=o` and `order_by=dateFiled asc`, following its
opaque cursor. Search snippets are never treated as case text. Opinion IDs come
from the nested `opinions` array; website opinion URLs contain **cluster IDs**.
The preferred field is `html_with_citations`, followed by explicit HTML/plain-text
fallbacks. Dissent/concurrence/type metadata is retained rather than pretending
every opinion is the court's controlling holding. No PACER/RECAP purchase,
Pray-and-Pay request, arbitrary download URL or external binary fetch is used.

Court metadata is fetched from `/courts/{id}/` and checked against the provider's
jurisdiction types. International, testing, committee and attorney-general advisory records are explicitly
excluded with ID/reason/hash evidence; unknown court types block the batch for
inspection. Court/API identifiers and pagination destinations are validated
before credentials are used. Keyword relevance is a retrieval signal, not an
expert IP classification. Coverage depends on provider holdings and the declared
queries; this is not a complete census of every American IP case.

## One-time setup; automatic operation thereafter

Create a PISTE application subscribed to the production Légifrance API, and a
CourtListener API token in your account. Inject these values using your normal
secret manager or uncommitted deployment environment:

```dotenv
VLE_LEGIFRANCE_PISTE_CLIENT_ID=<production-client-id>
VLE_LEGIFRANCE_PISTE_CLIENT_SECRET=<production-client-secret>
VLE_COURTLISTENER_API_TOKEN=<courtlistener-token>
VLE_AUTO_IMPORT_ENABLED=true
VLE_COURTLISTENER_ENABLED=true
```

The existing PostgreSQL/API-key/build-revision deployment configuration still
applies. `docker compose up -d --build` runs migrations and starts the importer
alongside the API and inference worker. It runs independently of model inference
and continues with inference disabled. Compose enables both source providers by
default. No secret is sent to the browser or stored in cursor/audit payloads.

Missing provider credentials produce visible, retrying setup errors, not fake
successful imports. This implementation session has no provider credentials;
live account access and deployment were **not** demonstrated. Accounts/keys
cannot be generated or assumed by the code. Offline tests use synthetic,
unreviewed responses and never call providers.

For an installed non-Compose service, start `vle-legato-importer` once as a normal
supervised service. It continuously processes due batches. Diagnostic commands:

```bash
alembic upgrade head
vle-legato-importer --status
vle-legato-importer --once
# Equivalent without reinstalling the console entry point:
python -m src.service.automatic_imports --status
```

`--once` processes one due bounded batch, **not the whole historical corpus**.
It returns nonzero for retry/lease failures. Continuous operation is the default.
A process heartbeat establishes liveness only; inspect import status to establish
provider setup, freshness and successful collection.

## Scheduling, recovery and bounds

Defaults are configurable in `.env.example` and Compose:

- One French code job, refreshed 86400 seconds after a completed scan.
- Two jobs per U.S. IP topic: a recent 90-day filing-date window, refreshed daily;
  and a historical backfill starting 1700-01-01, reconciled weekly after completion.
  Recent jobs run independently while a historical scan is still incomplete.
- Three source items per batch, 15-second poll interval, 120-second run deadline
  and 180-second lease. The lease must exceed the run bound by more than 10 seconds.
- CourtListener requests are spaced 13 seconds apart by default. Actual token
  quotas vary. HTTP 429 `Retry-After` is persisted as a provider-wide cooldown;
  later topic jobs do not immediately retry with the same token. No long in-process
  retry sleeps are used. Transient failures use bounded exponential backoff.
- At most 10000 French manifest entries, 100 search results/page, 1000 pending
  opinions/page, 100000 search pages/cycle, 200000 text characters/document,
  and 8 MB API responses (40 MB for the French manifest transport).

Dates, pending opinion IDs, cursors and manifest offsets are durable. A restart
resumes the same cycle; it does not silently start at page one. Configuration
changes that alter scope reset/audit the cursor; credential rotation does not.
The manifest pins article IDs/date; CourtListener's live index is not a frozen
snapshot. Backdated additions or modifications to old cases may be missed by a
recent window and are sought during periodic full reconciliation. Disappeared
records, retractions and repealed articles do not delete past source versions or
automatically revoke prior reviews. Their legal applicability needs review.

A unique conditional database lease prevents duplicate claims. A stale worker
cannot commit after a newer lease takes over. Network requests happen outside
transactions; source versions, ingestion audit and the updated cursor commit in
one fenced transaction. A failed batch advances nothing. SQLite development
mode establishes a physical outer write transaction before nested savepoints;
PostgreSQL is required for production. Migration `0004_automatic_imports` adds a
mutable operational cursor table, separate from immutable review/source evidence.

Malformed pagination, unsupported API shapes, empty/missing full texts and size/
time bounds are failures with retained progress and visible retries, never
successful complete imports. A permanently bad document can block its job until
fixed; it is not silently skipped. Other jobs continue, subject to provider-wide
cooldowns. Quotas can make large backfills slow; no fixed historical completion
or source-completeness guarantee is made.

## Observability and review boundary

The authenticated read-only `GET /v1/imports/status` reports configured jobs,
missing setup, due dates, lease state, last successful **batch**, failures and
last reports. The web console adds automatic-import status and French IP/U.S.
opinion source filters. Existing source endpoints expose the full acquired text,
provenance and extraction candidates. Retry/error reports omit raw exception
messages and secrets. Acquisition audit receipts link source hashes and cursor
hashes; changing text creates a new source version, never overwrites the old one.

Imported text is real provider data when live acquisition runs. Extracted norms
are unreviewed proposals. None of these jobs executes the legal decision workflow,
selects an interpretation, creates expert annotations, signs a review, or sets
`decision_authorized` true. A judicial opinion can quote a party, describe a
rejected rule or dissent; lexical extraction alone does not establish its legal
force. Automatically acquired sources do not satisfy Phase 5 F3's reviewed-data
eligibility gate merely by being imported. Existing human review gates remain.

## Provider interfaces checked

Checked 29 October 2026 against these primary sources:

- DILA API overview: https://www.legifrance.gouv.fr/contenu/pied-de-page/open-data-et-api
- DILA connection and consultation examples: https://www.legifrance.gouv.fr/contenu/pied-de-page/foire-aux-questions-api
- Free Law Project: https://free.law/
- CourtListener authentication/API: https://wiki.free.law/c/courtlistener/help/api/rest/v4/rest-api-v47
- Search and cursor schema: https://wiki.free.law/c/courtlistener/help/api/rest/v4/search
- Full opinion/cluster schema: https://wiki.free.law/c/courtlistener/help/api/rest/v4/case-law
- Jurisdiction constants and supported sorts: `freelawproject/courtlistener`,
  `cl/search/models.py` and `cl/search/forms.py` (upstream source).

No actual source data, reviewer attestations or citations were fabricated to
claim live validation. Existing reviewed and synthetic research corpora are
unchanged; new import tests use explicitly synthetic transport fixtures.
