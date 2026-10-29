from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import APIKeyEntry, Settings

bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    subject: str
    roles: frozenset[str]
    key_hash: str

    def has_any(self, roles: set[str]) -> bool:
        return "admin" in self.roles or bool(self.roles & roles)


class APIKeyStore:
    def __init__(self, entries: list[APIKeyEntry]) -> None:
        self._entries: list[tuple[str, str, frozenset[str]]] = []
        for entry in entries:
            token = entry.token.get_secret_value()
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            self._entries.append((digest, entry.subject, frozenset(entry.roles)))

    def authenticate(self, token: str) -> Principal | None:
        candidate = hashlib.sha256(token.encode("utf-8")).hexdigest()
        for digest, subject, roles in self._entries:
            if hmac.compare_digest(candidate, digest):
                return Principal(subject=subject, roles=roles, key_hash=digest)
        return None


def get_principal(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> Principal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")
    store: APIKeyStore = request.app.state.api_key_store
    principal = store.authenticate(credentials.credentials)
    if principal is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid bearer token")
    request.app.state.rate_limiter.consume(principal.key_hash)
    return principal


def require_roles(*roles: str) -> Callable[[Principal], Principal]:
    required = set(roles)

    def dependency(principal: Principal = Depends(get_principal)) -> Principal:
        if not principal.has_any(required):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="insufficient role")
        return principal

    dependency.vle_required_roles = tuple(sorted(required))
    return dependency


def verify_metrics_token(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> None:
    settings: Settings = request.app.state.settings
    configured = settings.metrics_bearer_token
    if configured is None:
        if settings.environment == "production":
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="metrics authentication is not configured")
        return
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing metrics bearer token")
    if not hmac.compare_digest(credentials.credentials, configured.get_secret_value()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid metrics bearer token")
