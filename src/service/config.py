from __future__ import annotations

import json
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class APIKeyEntry(BaseModel):
    token: SecretStr
    subject: str = Field(min_length=1, max_length=128)
    roles: set[Literal["reader", "operator", "reviewer", "auditor", "admin"]]

    @field_validator("token")
    @classmethod
    def token_length(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 32:
            raise ValueError("API keys must contain at least 32 characters")
        return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VLE_", env_file=".env", extra="ignore")

    environment: Literal["development", "test", "production"] = "development"
    service_name: str = "legal-norm-argumentation"
    service_version: str = "0.4.0"
    build_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    database_url: str = "sqlite+pysqlite:///./vle_legato.db"
    database_pool_size: int = Field(default=10, ge=1, le=100)
    database_max_overflow: int = Field(default=20, ge=0, le=200)
    database_pool_recycle_seconds: int = Field(default=1800, ge=60, le=86400)
    api_keys_json: str = "[]"
    metrics_bearer_token: SecretStr | None = None
    cors_origins_csv: str = ""
    max_request_bytes: int = Field(default=1_048_576, ge=1024, le=10_485_760)
    rate_limit_per_minute: int = Field(default=120, ge=1, le=10000)
    idempotency_ttl_seconds: int = Field(default=86400, ge=60, le=604800)
    require_separation_of_duties: bool = True
    allow_docs: bool = False
    auto_create_schema: bool = False
    log_level: str = "INFO"
    otlp_endpoint: str | None = None
    otlp_headers: str | None = None

    @field_validator("database_url")
    @classmethod
    def normalize_database_url(cls, value: str) -> str:
        if value.startswith("postgresql://"):
            return "postgresql+psycopg://" + value.removeprefix("postgresql://")
        return value

    @property
    def cors_origins(self) -> list[str]:
        return [part.strip() for part in self.cors_origins_csv.split(",") if part.strip()]

    @property
    def api_keys(self) -> list[APIKeyEntry]:
        try:
            raw = json.loads(self.api_keys_json)
        except json.JSONDecodeError as exc:
            raise ValueError("VLE_API_KEYS_JSON must contain valid JSON") from exc
        if not isinstance(raw, list):
            raise ValueError("VLE_API_KEYS_JSON must be a JSON array")
        return [APIKeyEntry.model_validate(item) for item in raw]

    @model_validator(mode="after")
    def validate_production_settings(self) -> Settings:
        if self.environment != "production":
            return self
        if not self.database_url.startswith("postgresql+psycopg://"):
            raise ValueError("production requires PostgreSQL via VLE_DATABASE_URL")
        if self.allow_docs:
            raise ValueError("interactive API docs must be disabled in production")
        if self.auto_create_schema:
            raise ValueError("schema auto-creation is forbidden in production; use Alembic migrations")
        if "*" in self.cors_origins:
            raise ValueError("wildcard CORS is forbidden in production")
        entries = self.api_keys
        if not entries:
            raise ValueError("production requires at least one API key")
        subjects = {entry.subject for entry in entries}
        if self.require_separation_of_duties and len(subjects) < 2:
            raise ValueError("production separation of duties requires at least two API-key subjects")
        roles = {role for entry in entries for role in entry.roles}
        if not ({"operator", "admin"} & roles):
            raise ValueError("production requires an operator or admin principal")
        if not ({"reviewer", "admin"} & roles):
            raise ValueError("production requires a reviewer or admin principal")
        if self.metrics_bearer_token is None or len(self.metrics_bearer_token.get_secret_value()) < 32:
            raise ValueError("production requires a metrics bearer token of at least 32 characters")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
