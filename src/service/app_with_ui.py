from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI

from .app import create_app as create_core_app
from .config import Settings
from .legifrance_routes import router as legifrance_router


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the production API plus the integrated Légifrance console."""
    app = create_core_app(settings)
    app.include_router(legifrance_router)
    return app


def main() -> None:
    """Launch the integrated service from the installed console script."""
    host = os.getenv("VLE_HOST", "0.0.0.0")
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
