"""Production service layer for the VLE-LEGATO normative lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

    from .config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the integrated production service without eager import cycles."""
    from .app_with_ui import create_app as factory

    return factory(settings)


__all__ = ["create_app"]
