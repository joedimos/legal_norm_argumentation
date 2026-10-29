"""Integrated application factory without module-level database initialization."""

from fastapi import FastAPI

from .app import create_app as create_core_app
from .config import Settings
from .legifrance_routes import router as legifrance_router


def create_app(settings: Settings | None = None) -> FastAPI:
    app = create_core_app(settings)
    app.include_router(legifrance_router)
    return app
