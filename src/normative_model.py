from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum
import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Tuple


class Modality(str, Enum):
    OBLIGATION = "obligation"
    PROHIBITION = "prohibition"
    PERMISSION = "permission"
    RECOMMENDATION = "recommendation"


class SourceKind(str, Enum):
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
    issuer: Optional[str] = None
    instrument: Optional[str] = None
    jurisdiction: Optional[str] = None
    authority_rank: Optional[int] = None
    enacted_on: Optional[str] = None
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None
    version: Optional[str] = None

    def is_effective(self, on_date: Optional[str]) -> Optional[bool]:
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
    span: Tuple[int, int]
    quote: str
    document_hash: Optional[str] = None
    locator: Optional[str] = None


@dataclass(frozen=True)
class Predicate:
    name: str
    negated: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("predicate name must be non-empty")

    def evaluate(self, facts: Mapping[str, bool]) -> Optional[bool]:
        if self.name not in facts:
            return None
        value = bool(facts[self.name])
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
    conditions: Tuple[Predicate, ...] = ()
    exceptions: Tuple[Predicate, ...] = ()
    object: Optional[str] = None
    deadline: Optional[str] = None
    references: Tuple[str, ...] = ()
    authority: SourceAuthority = field(default_factory=SourceAuthority)
    confidence: float = 1.0
    assumptions: Tuple[str, ...] = ()
    provenance: Optional[SourceProvenance] = None
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

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["modality"] = self.modality.value
        data["authority"]["kind"] = self.authority.kind.value
        data["canonical"] = self.canonical()
        data["semantic_hash"] = self.semantic_hash()
        return data


@dataclass(frozen=True)
class InterpretationSet:
    source_norm_id: str
    interpretations: Tuple[NormInterpretation, ...]
    selected_id: Optional[str] = None
    selection_rationale: Optional[str] = None

    def selected(self) -> Optional[NormInterpretation]:
        if self.selected_id is None:
            return None
        return next((item for item in self.interpretations if item.id == self.selected_id), None)

    @property
    def unresolved(self) -> bool:
        return self.selected() is None


@dataclass(frozen=True)
class FactContext:
    facts: Mapping[str, bool]
    actor: Optional[str] = None
    observed_actions: Tuple[str, ...] = ()
    at_date: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def stable_interpretation_id(source_norm_id: str, label: str, canonical: str) -> str:
    digest = hashlib.sha256(f"{source_norm_id}:{label}:{canonical}".encode("utf-8")).hexdigest()[:16]
    return f"formal-{digest}"
