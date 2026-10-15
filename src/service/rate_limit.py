from __future__ import annotations

from datetime import UTC, datetime

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .database import Database
from .db_models import RateLimitBucket


class DatabaseRateLimiter:
    """Cross-worker fixed-window rate limiting persisted in the primary database."""

    def __init__(self, database: Database, limit_per_minute: int) -> None:
        self.database = database
        self.limit = limit_per_minute

    def consume(self, key_hash: str) -> None:
        window_start = datetime.now(UTC).replace(second=0, microsecond=0)
        count = 0
        with self.database.session() as session:
            dialect = session.get_bind().dialect.name
            if dialect == "postgresql":
                stmt = (
                    pg_insert(RateLimitBucket)
                    .values(key_hash=key_hash, window_start=window_start, count=1)
                    .on_conflict_do_update(
                        index_elements=[RateLimitBucket.key_hash, RateLimitBucket.window_start],
                        set_={"count": RateLimitBucket.count + 1},
                    )
                    .returning(RateLimitBucket.count)
                )
                count = int(session.execute(stmt).scalar_one())
            elif dialect == "sqlite":
                stmt = (
                    sqlite_insert(RateLimitBucket)
                    .values(key_hash=key_hash, window_start=window_start, count=1)
                    .on_conflict_do_update(
                        index_elements=[RateLimitBucket.key_hash, RateLimitBucket.window_start],
                        set_={"count": RateLimitBucket.count + 1},
                    )
                    .returning(RateLimitBucket.count)
                )
                count = int(session.execute(stmt).scalar_one())
            else:
                record = session.execute(
                    select(RateLimitBucket)
                    .where(RateLimitBucket.key_hash == key_hash, RateLimitBucket.window_start == window_start)
                    .with_for_update()
                ).scalar_one_or_none()
                if record is None:
                    record = RateLimitBucket(key_hash=key_hash, window_start=window_start, count=1)
                    session.add(record)
                    count = 1
                else:
                    record.count += 1
                    count = record.count
        if count > self.limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="rate limit exceeded",
                headers={"Retry-After": "60"},
            )
