from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .config import Settings
from .db_models import Base


class Database:
    def __init__(self, settings: Settings) -> None:
        kwargs: dict = {"pool_pre_ping": True}
        if settings.database_url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        else:
            kwargs.update(
                pool_size=settings.database_pool_size,
                max_overflow=settings.database_max_overflow,
                pool_recycle=settings.database_pool_recycle_seconds,
            )
        self.engine: Engine = create_engine(settings.database_url, future=True, **kwargs)
        self._session_factory = sessionmaker(bind=self.engine, autoflush=False, expire_on_commit=False, future=True)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def ping(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def ready(self, *, require_migrations: bool = True) -> bool:
        if not self.ping():
            return False
        try:
            tables = set(inspect(self.engine).get_table_names())
        except Exception:
            return False
        required = {
            "source_documents",
            "candidate_reviews",
            "interpretation_selections",
            "lifecycle_runs",
            "audit_events",
            "idempotency_records",
            "rate_limit_buckets",
            "workflow_requests",
            "workflow_events",
            "import_cursors",
        }
        if not required.issubset(tables):
            return False
        if require_migrations and "alembic_version" not in tables:
            return False
        if require_migrations:
            try:
                with self.engine.connect() as connection:
                    if connection.execute(text("SELECT version_num FROM alembic_version")).scalar() != "0004_automatic_imports":
                        return False
            except Exception:
                return False
        return True

    def create_schema_for_tests(self) -> None:
        Base.metadata.create_all(self.engine)

    def dispose(self) -> None:
        self.engine.dispose()
