from __future__ import annotations

from typing import Protocol

from src.legifrance import LegifranceClient, LegifranceError, ScrapeBatch
from src.legifrance_piste import PisteLegifranceClient

from .config import Settings


class LegifranceProvider(Protocol):
    def fetch_corpus(self, corpus: str, *, limit: int = 20, offset: int = 0) -> ScrapeBatch: ...


def build_legifrance_client(settings: Settings) -> LegifranceProvider:
    """Choose the configured official-source transport without exposing credentials."""
    mode = settings.legifrance_source
    if mode in {"auto", "piste"} and settings.has_piste_credentials:
        assert settings.legifrance_piste_client_id is not None
        assert settings.legifrance_piste_client_secret is not None
        return PisteLegifranceClient(
            client_id=settings.legifrance_piste_client_id.get_secret_value(),
            client_secret=settings.legifrance_piste_client_secret.get_secret_value(),
            environment=settings.legifrance_piste_environment,
            timeout_seconds=settings.legifrance_timeout_seconds,
        )
    if mode == "piste":
        raise LegifranceError("PISTE provider is configured but OAuth client credentials are missing")
    return LegifranceClient(
        timeout_seconds=settings.legifrance_timeout_seconds,
        request_delay_seconds=settings.legifrance_request_delay_seconds,
    )


def provider_name(settings: Settings) -> str:
    if settings.legifrance_source in {"auto", "piste"} and settings.has_piste_credentials:
        return "piste"
    return "html"
