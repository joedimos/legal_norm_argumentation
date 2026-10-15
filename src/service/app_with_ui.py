from __future__ import annotations

from fastapi import FastAPI

from .app import create_app as create_core_app
from .config import Settings
from .legifrance_routes import router as legifrance_router


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the production API plus the integrated Légifrance console."""
    app = create_core_app(settings)
    app.include_router(legifrance_router)
    return app


app = create_app()
