from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class ReferenceResolution:
    reference: str
    status: str
    source_id: Optional[str] = None
    text: Optional[str] = None
    explanation: Optional[str] = None


class ReferenceResolver:
    """Deterministic resolver over a caller-supplied authoritative corpus index."""

    def __init__(self, index: Mapping[str, Mapping[str, str]]) -> None:
        self.index = {self._normalize(k): dict(v) for k, v in index.items()}

    @staticmethod
    def _normalize(reference: str) -> str:
        return " ".join(reference.lower().split())

    def resolve(self, reference: str) -> ReferenceResolution:
        item = self.index.get(self._normalize(reference))
        if item is None:
            return ReferenceResolution(reference, "unresolved", explanation="Reference not found in supplied corpus index.")
        return ReferenceResolution(reference, "resolved", source_id=item.get("source_id"), text=item.get("text"))

    def resolve_all(self, references: Sequence[str]) -> List[ReferenceResolution]:
        return [self.resolve(reference) for reference in references]
