from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class SourceCreateRequest(BaseModel):
    source_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=500_000)


class LegifranceScrapeRequest(BaseModel):
    corpus: Literal["ip", "competition"]
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0, le=100_000)


class ReviewCreateRequest(BaseModel):
    candidate_id: str = Field(min_length=1, max_length=128)
    decision: Literal["approved", "rejected", "needs_revision"]
    note: str | None = Field(default=None, max_length=4000)


class InterpretationSelectionRequest(BaseModel):
    candidate_id: str = Field(min_length=1, max_length=128)
    interpretation_id: str = Field(min_length=1, max_length=128)
    rationale: str = Field(min_length=5, max_length=4000)


class EvaluateRequest(BaseModel):
    facts: dict[str, bool] = Field(default_factory=dict)
    actor: str | None = Field(default=None, max_length=256)
    observed_actions: list[str] = Field(default_factory=list, max_length=1000)
    at_date: str | None = None
    jurisdiction: str | None = Field(default=None, min_length=1, max_length=128)
    capacity: str | None = Field(default=None, min_length=1, max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)
    high_stakes: bool = True
    legal_assessment_id: str | None = Field(default=None, max_length=36)

    @field_validator("at_date")
    @classmethod
    def canonical_date(cls, value):
        if value is not None and date.fromisoformat(value).isoformat() != value:
            raise ValueError("at_date must be a canonical YYYY-MM-DD date")
        return value

    @field_validator("facts")
    @classmethod
    def limit_facts(cls, value: dict[str, bool]) -> dict[str, bool]:
        if len(value) > 5000:
            raise ValueError("facts may contain at most 5000 entries")
        if any(len(key) > 512 for key in value):
            raise ValueError("fact names may contain at most 512 characters")
        return value

    @field_validator("metadata")
    @classmethod
    def limit_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > 100:
            raise ValueError("metadata may contain at most 100 keys")
        return value
