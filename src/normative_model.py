from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum
from typing import Any


class Modality(str, Enum):  # noqa: UP042 - preserve public Enum string behavior
    OBLIGATION = "obligation"
    PROHIBITION = "prohibition"
    PERMISSION = "permission"
    RECOMMENDATION = "recommendation"


class SourceKind(str, Enum):  # noqa: UP042 - preserve public Enum string behavior
    LAW = "law"
    REGULATION = "regulation"
    STANDARD = "standard"
    POLICY = "policy"
    ETHICAL_PRINCIPLE = "ethical_principle"
    CONTRACT = "contract"
    CASE = "case"
    OTHER = "other"


@dataclass(frozen=True)
class SourceAuthority:
    kind: SourceKind = SourceKind.OTHER
    issuer: str | None = None
    instrument: str | None = None
    jurisdiction: str | None = None
    authority_rank: int | None = None
    enacted_on: str | None = None
    effective_from: str | None = None
    effective_to: str | None = None
    version: str | None = None

    def __post_init__(self) -> None:
        for value in (self.enacted_on, self.effective_from, self.effective_to):
            if value is not None:
                parsed = date.fromisoformat(value)
                if parsed.isoformat() != value:
                    raise ValueError("dates must use canonical YYYY-MM-DD format")
        if self.effective_from and self.effective_to and self.effective_from > self.effective_to:
            raise ValueError("effective_from must not follow effective_to")
        if self.authority_rank is not None and type(self.authority_rank) is not int:
            raise ValueError("authority_rank must be an integer")

    def is_effective(self, on_date: str | None) -> bool | None:
        if on_date is None:
            return None
        current = date.fromisoformat(on_date)
        if self.effective_from and current < date.fromisoformat(self.effective_from):
            return False
        if self.effective_to and current > date.fromisoformat(self.effective_to):
            return False
        return True


@dataclass(frozen=True)
class SourceProvenance:
    source_id: str
    span: tuple[int, int]
    quote: str
    document_hash: str | None = None
    locator: str | None = None


@dataclass(frozen=True)
class Predicate:
    name: str
    negated: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("predicate name must be non-empty")

    def evaluate(self, facts: Mapping[str, bool]) -> bool | None:
        if self.name not in facts:
            return None
        value = facts[self.name]
        if type(value) is not bool:
            raise ValueError("predicate facts must be boolean")
        return not value if self.negated else value

    def canonical(self) -> str:
        return f"not {self.name}" if self.negated else self.name


@dataclass(frozen=True)
class NormInterpretation:
    id: str
    source_norm_id: str
    modality: Modality
    subject: str
    action: str
    conditions: tuple[Predicate, ...] = ()
    exceptions: tuple[Predicate, ...] = ()
    object: str | None = None
    deadline: str | None = None
    references: tuple[str, ...] = ()
    authority: SourceAuthority = field(default_factory=SourceAuthority)
    confidence: float = 1.0
    assumptions: tuple[str, ...] = ()
    provenance: SourceProvenance | None = None
    interpretation_label: str = "primary"

    def __post_init__(self) -> None:
        if not self.id or not self.source_norm_id:
            raise ValueError("interpretation and source norm ids must be non-empty")
        if not self.subject.strip() or not self.action.strip():
            raise ValueError("formal norms require non-empty subject and action")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")

    def canonical(self) -> str:
        cond = " & ".join(p.canonical() for p in self.conditions) or "true"
        exc = " | ".join(p.canonical() for p in self.exceptions) or "false"
        obj = f"({self.object})" if self.object else ""
        return f"{self.modality.value.upper()}[{self.subject}:{self.action}{obj}] IF {cond} UNLESS {exc}"

    def semantic_hash(self) -> str:
        payload = {
            "modality": self.modality.value,
            "subject": self.subject,
            "action": self.action,
            "object": self.object,
            "conditions": [p.canonical() for p in self.conditions],
            "exceptions": [p.canonical() for p in self.exceptions],
            "deadline": self.deadline,
            "references": list(self.references),
            "authority": asdict(self.authority),
            "assumptions": list(self.assumptions),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["modality"] = self.modality.value
        data["authority"]["kind"] = self.authority.kind.value
        data["canonical"] = self.canonical()
        data["semantic_hash"] = self.semantic_hash()
        return data


@dataclass(frozen=True)
class InterpretationSet:
    source_norm_id: str
    interpretations: tuple[NormInterpretation, ...]
    selected_id: str | None = None
    selection_rationale: str | None = None

    def selected(self) -> NormInterpretation | None:
        if self.selected_id is None:
            return None
        return next((item for item in self.interpretations if item.id == self.selected_id), None)

    @property
    def unresolved(self) -> bool:
        return self.selected() is None


@dataclass(frozen=True)
class FactContext:
    facts: Mapping[str, bool]
    actor: str | None = None
    observed_actions: tuple[str, ...] = ()
    at_date: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def stable_interpretation_id(source_norm_id: str, label: str, canonical: str) -> str:
    digest = hashlib.sha256(f"{source_norm_id}:{label}:{canonical}".encode()).hexdigest()[:16]
    return f"formal-{digest}"
