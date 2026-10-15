from __future__ import annotations

import argparse
import json
import uuid

from src.legifrance import CORPORA, LegifranceClient, LegifranceError

from .config import Settings, get_settings
from .database import Database
from .legifrance_ingestion import ingest_legifrance_batch
from .repository import cleanup_ephemeral_state, verify_audit_chain


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLE-LEGATO maintenance and authoritative-source synchronization")
    parser.add_argument(
        "--sync-legifrance",
        choices=["none", "ip", "competition", "all"],
        default="none",
        help="Synchronize a configured Légifrance corpus before maintenance checks.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Articles per Légifrance page/batch.")
    parser.add_argument("--offset", type=int, default=0, help="Starting article offset for a corpus.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Continue through all discovered articles, subject to the configured full-sync bound.",
    )
    return parser.parse_args()


def _sync_legifrance(database: Database, settings: Settings, args: argparse.Namespace) -> list[dict]:
    if args.sync_legifrance == "none":
        return []
    if not settings.legifrance_enabled:
        raise SystemExit("Légifrance synchronization requested but VLE_LEGIFRANCE_ENABLED is false")

    limit = args.limit or settings.legifrance_max_articles_per_run
    if not 1 <= limit <= settings.legifrance_max_articles_per_run:
        raise SystemExit(
            f"--limit must be between 1 and {settings.legifrance_max_articles_per_run} for this deployment"
        )
    if args.offset < 0:
        raise SystemExit("--offset must be non-negative")

    corpora = list(CORPORA) if args.sync_legifrance == "all" else [args.sync_legifrance]
    client = LegifranceClient(
        timeout_seconds=settings.legifrance_timeout_seconds,
        request_delay_seconds=settings.legifrance_request_delay_seconds,
    )
    summaries: list[dict] = []

    for corpus in corpora:
        offset = args.offset
        processed = 0
        while True:
            remaining = settings.legifrance_max_full_sync_articles - processed
            if remaining <= 0:
                raise SystemExit(
                    f"full-sync safety bound reached for {corpus}: "
                    f"{settings.legifrance_max_full_sync_articles} articles"
                )
            page_limit = min(limit, remaining)
            try:
                batch = client.fetch_corpus(corpus, limit=page_limit, offset=offset)
            except (LegifranceError, ValueError) as exc:
                raise SystemExit(f"Légifrance synchronization failed for {corpus}: {exc}") from exc

            request_id = f"maintenance-legifrance-{uuid.uuid4()}"
            with database.session() as session:
                summary = ingest_legifrance_batch(
                    session,
                    batch,
                    actor="maintenance:legifrance",
                    request_id=request_id,
                )
            summaries.append(summary)
            processed += batch.selected

            if not args.full or batch.selected == 0 or offset + batch.selected >= batch.discovered:
                break
            offset += batch.selected

    return summaries


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    database = Database(settings)
    try:
        sync = _sync_legifrance(database, settings, args)
        with database.session() as session:
            cleanup = cleanup_ephemeral_state(session)
            chain = verify_audit_chain(session)
        print(json.dumps({"legifrance_sync": sync, "cleanup": cleanup, "audit_chain": chain}, sort_keys=True))
        if not chain["valid"]:
            raise SystemExit(2)
    finally:
        database.dispose()


if __name__ == "__main__":
    main()
