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
    service_version: str = "0.9.1"
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
    inference_enabled: bool = False
    model_provider: Literal["ollama", "vllm"] = "ollama"
    model_name: str | None = Field(default=None, max_length=256)
    model_endpoint: str | None = None
    model_revision: str | None = Field(default=None, min_length=1, max_length=256)
    workflow_stale_after_seconds: int = Field(default=300, ge=30, le=86400)
    model_api_key: SecretStr | None = None
    reviewer_registry_path: str | None = None
    reviewer_registry_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    release_enforcement: Literal["reviewed", "strict"] = "reviewed"
    release_profile: Literal["legacy", "ip-supervised-v1"] = "legacy"
    release_source_max_age_seconds: int = Field(default=172800, ge=3600, le=604800)
    release_evidence_path: str | None = None
    release_evidence_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    release_max_p95_ms: float = Field(default=1000, gt=0)
    release_min_load_requests: int = Field(default=100, ge=1)
    release_min_cases: int = Field(default=100, ge=1)
    release_min_extraction_f1: float = Field(default=0.9, ge=0, le=1)
    release_min_compliance_accuracy: float = Field(default=0.9, ge=0, le=1)
    release_max_false_compliance_rate: float = Field(default=0.0, ge=0, le=1)
    legifrance_enabled: bool = True
    legifrance_source: Literal["auto", "piste", "html"] = "auto"
    legifrance_piste_environment: Literal["production", "sandbox"] = "production"
    legifrance_piste_client_id: SecretStr | None = None
    legifrance_piste_client_secret: SecretStr | None = None
    legifrance_timeout_seconds: float = Field(default=12.0, ge=2.0, le=60.0)
    legifrance_request_delay_seconds: float = Field(default=0.2, ge=0.0, le=5.0)
    legifrance_max_articles_per_run: int = Field(default=50, ge=1, le=100)
    legifrance_max_full_sync_articles: int = Field(default=2500, ge=1, le=10_000)

    # Source acquisition is separate from model inference and human review.
    auto_import_enabled: bool = True
    courtlistener_enabled: bool = False
    courtlistener_api_token: SecretStr | None = None
    courtlistener_request_delay_seconds: float = Field(default=13, ge=0, le=60)
    import_batch_size: int = Field(default=3, ge=1, le=25)
    import_poll_seconds: float = Field(default=15, ge=1, le=60)
    import_refresh_seconds: int = Field(default=86400, ge=60, le=604800)
    import_reconcile_seconds: int = Field(default=604800, ge=3600, le=2592000)
    import_recent_days: int = Field(default=90, ge=1, le=3660)
    import_start_date: str = "1700-01-01"
    import_max_run_seconds: int = Field(default=120, ge=10, le=600)
    import_lease_seconds: int = Field(default=180, ge=30, le=900)

    @field_validator("import_start_date")
    @classmethod
    def validate_import_start(cls, value):
        from datetime import date
        return date.fromisoformat(value).isoformat()

    @model_validator(mode="after")
    def validate_import_lease(self):
        if self.import_lease_seconds <= self.import_max_run_seconds + 10:
            raise ValueError("import lease must exceed the run deadline by more than ten seconds")
        return self

    @field_validator("model_revision", "reviewer_registry_path", "reviewer_registry_sha256", "release_evidence_path", "release_evidence_sha256", mode="before")
    @classmethod
    def empty_optional_evidence(cls, value):
        return None if value == "" else value

    @model_validator(mode="after")
    def validate_workflow_settings(self) -> Settings:
        if self.release_profile == "ip-supervised-v1":
            if self.release_enforcement != "strict":
                raise ValueError("the IP release profile requires strict release enforcement")
            if not self.build_revision or not self.model_name or not self.model_revision:
                raise ValueError("the IP release profile requires exact build, model and model revision bindings")
            if not self.legifrance_enabled or not self.courtlistener_enabled or not self.auto_import_enabled:
                raise ValueError("the IP release profile requires automatic acquisition from both IP providers")
            if self.legifrance_piste_environment != "production":
                raise ValueError("the IP release profile requires production PISTE")
        entries = self.api_keys
        tokens = [entry.token.get_secret_value() for entry in entries]
        if len(set(tokens)) != len(tokens):
            raise ValueError("API key tokens must be unique across configured principals")
        for path, sha in ((self.reviewer_registry_path, self.reviewer_registry_sha256),
                          (self.release_evidence_path, self.release_evidence_sha256)):
            if bool(path) != bool(sha):
                raise ValueError("evidence/registry paths require separately pinned SHA-256 digests")
        if self.inference_enabled:
            from src.model_runtime import resolve_config
            request = {"provider": self.model_provider, "model": self.model_name}
            if self.model_endpoint:
                request["endpoint"] = self.model_endpoint
            resolve_config(request)
        return self

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

    @property
    def has_piste_credentials(self) -> bool:
        return bool(self.legifrance_piste_client_id and self.legifrance_piste_client_id.get_secret_value().strip()
                    and self.legifrance_piste_client_secret and self.legifrance_piste_client_secret.get_secret_value().strip())

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
        if self.legifrance_enabled and self.legifrance_source in {"auto", "piste"} and not self.has_piste_credentials:
            raise ValueError(
                "production Légifrance auto/piste mode requires PISTE credentials; "
                "set VLE_LEGIFRANCE_SOURCE=html only for an explicit HTML fallback"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
