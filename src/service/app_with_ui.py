from __future__ import annotations

import os

import uvicorn

from .factory import create_app as create_app


def main() -> None:
    """Launch the integrated service from the installed console script."""
    # The interactive console script defaults to loopback. Container and deployment
    # manifests bind explicitly as needed, and VLE_HOST remains an operator override.
    host = os.getenv("VLE_HOST", "127.0.0.1")
    port = int(os.getenv("VLE_PORT", "8000"))
    uvicorn.run(
        "src.service.app_with_ui:app",
        host=host,
        port=port,
        server_header=False,
        timeout_keep_alive=5,
        limit_concurrency=200,
    )


app = create_app()
