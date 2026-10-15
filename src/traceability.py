from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Dict, List, Mapping


@dataclass(frozen=True)
class EvidenceNode:
    id: str
    kind: str
    payload_hash: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class EvidenceEdge:
    source: str
    target: str
    relation: str


class EvidenceGraph:
    """Append-only provenance graph for source→formalisation→decision evidence."""

    def __init__(self) -> None:
        self.nodes: Dict[str, EvidenceNode] = {}
        self.edges: List[EvidenceEdge] = []

    @staticmethod
    def _hash(payload: Any) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def add(self, node_id: str, kind: str, payload: Any, **metadata: Any) -> EvidenceNode:
        node = EvidenceNode(node_id, kind, self._hash(payload), dict(metadata))
        existing = self.nodes.get(node_id)
        if existing and existing.payload_hash != node.payload_hash:
            raise ValueError(f"evidence node {node_id} is immutable; create a new version id")
        self.nodes[node_id] = node
        return node

    def link(self, source: str, target: str, relation: str) -> EvidenceEdge:
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("both evidence nodes must exist before linking")
        edge = EvidenceEdge(source, target, relation)
        if edge not in self.edges:
            self.edges.append(edge)
        return edge

    def lineage(self, target: str) -> List[Dict[str, str]]:
        result: List[Dict[str, str]] = []
        frontier = [target]
        seen = {target}
        while frontier:
            current = frontier.pop()
            for edge in self.edges:
                if edge.target == current:
                    result.append(asdict(edge))
                    if edge.source not in seen:
                        seen.add(edge.source)
                        frontier.append(edge.source)
        return result

    def export(self) -> Dict[str, Any]:
        payload = {
            "nodes": {key: asdict(value) for key, value in sorted(self.nodes.items())},
            "edges": [asdict(edge) for edge in self.edges],
        }
        payload["graph_hash"] = self._hash(payload)
        return payload
