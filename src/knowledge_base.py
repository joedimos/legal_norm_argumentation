from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from typing import Any, Dict, List, Mapping, Optional

from .normative_model import NormInterpretation


@dataclass(frozen=True)
class NormRevision:
    revision_id: str
    norm_id: str
    semantic_hash: str
    created_at: str
    author: str
    reason: str
    payload: Mapping[str, Any]
    supersedes: Optional[str] = None


class NormKnowledgeBase:
    """Append-only version store for governed normative knowledge."""

    def __init__(self) -> None:
        self._revisions: Dict[str, NormRevision] = {}
        self._history: Dict[str, List[str]] = {}

    def add(self, norm: NormInterpretation, author: str, reason: str) -> NormRevision:
        if not author.strip() or not reason.strip():
            raise ValueError("author and reason are required")
        previous = self.latest(norm.id)
        semantic_hash = norm.semantic_hash()
        seed = f"{norm.id}:{semantic_hash}:{len(self._history.get(norm.id, []))}"
        revision_id = "rev-" + hashlib.sha256(seed.encode()).hexdigest()[:16]
        record = NormRevision(
            revision_id=revision_id,
            norm_id=norm.id,
            semantic_hash=semantic_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
            author=author,
            reason=reason,
            payload=norm.to_dict(),
            supersedes=previous.revision_id if previous else None,
        )
        self._revisions[revision_id] = record
        self._history.setdefault(norm.id, []).append(revision_id)
        return record

    def latest(self, norm_id: str) -> Optional[NormRevision]:
        history = self._history.get(norm_id, [])
        return self._revisions[history[-1]] if history else None

    def history(self, norm_id: str) -> List[NormRevision]:
        return [self._revisions[item] for item in self._history.get(norm_id, [])]

    def changed(self, norm: NormInterpretation) -> bool:
        latest = self.latest(norm.id)
        return latest is None or latest.semantic_hash != norm.semantic_hash()

    def export(self) -> Dict[str, Any]:
        return {
            "revisions": {key: asdict(value) for key, value in self._revisions.items()},
            "history": {key: list(value) for key, value in self._history.items()},
        }
