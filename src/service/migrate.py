"""Forward-only migration entry point, including installations outside a checkout."""

import argparse
import json
import os

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine

from src.resources import resource_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Read migration heads without changing the database")
    args = parser.parse_args()
    config = Config()
    config.set_main_option("script_location", str(resource_path("migrations")))
    url = os.environ.get("VLE_DATABASE_URL")
    if not url:
        parser.error("set VLE_DATABASE_URL through the deployment secret manager")
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url.removeprefix("postgresql://")
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    try:
        if not args.check:
            command.upgrade(config, "head")
        engine = create_engine(url)
        try:
            with engine.connect() as connection:
                actual = sorted(MigrationContext.configure(connection).get_current_heads())
            expected = sorted(ScriptDirectory.from_config(config).get_heads())
            passed = actual == expected
        finally:
            engine.dispose()
        print(json.dumps({"schema": "vle-migration-check/v1", "passed": passed,
                          "current_heads": actual, "expected_heads": expected}, sort_keys=True))
        return 0 if passed else 1
    except Exception:
        # Exception strings can include a database URL containing credentials.
        print(json.dumps({"schema": "vle-migration-check/v1", "passed": False, "reason": "migration_or_connection_failed"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
