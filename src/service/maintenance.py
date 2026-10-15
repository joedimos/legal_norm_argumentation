from __future__ import annotations

import json

from .config import get_settings
from .database import Database
from .repository import cleanup_ephemeral_state, verify_audit_chain


def main() -> None:
    settings = get_settings()
    database = Database(settings)
    try:
        with database.session() as session:
            cleanup = cleanup_ephemeral_state(session)
            chain = verify_audit_chain(session)
        print(json.dumps({"cleanup": cleanup, "audit_chain": chain}, sort_keys=True))
        if not chain["valid"]:
            raise SystemExit(2)
    finally:
        database.dispose()


if __name__ == "__main__":
    main()
