from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple, Union
import logging

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LegalArgument:
    """A reviewable legal/normative argument."""

    id: str
    premise: str = ""
    conclusion: str = ""
    confidence: Tuple[float, float] = (0.8, 1.0)
    supporting_args: FrozenSet[str] = frozenset()
    attacking_args: FrozenSet[str] = frozenset()
    sequent_type: str = "hypothesis"
    source_norm_ids: FrozenSet[str] = frozenset()

    def __post_init__(self) -> None:
        low, high = self.confidence
        if not (0.0 <= low <= high <= 1.0):
            raise ValueError("confidence must satisfy 0 <= low <= high <= 1")
        if not self.id:
            raise ValueError("argument id must be non-empty")


@dataclass(frozen=True)
class Sequent:
    antecedents: FrozenSet[str]
    consequents: FrozenSet[str]
    confidence: float
    is_contrapositive: bool = False


ArgumentInput = Union[LegalArgument, str]


class OptimizedAAF:
    """Dung-style abstract argumentation with support edges kept separate.

    The core semantics depend only on ``attacks``. Support is retained as
    metadata and for downstream structured/bipolar reasoning; it is never
    silently treated as a Dung attack.
    """

    def __init__(
        self,
        arguments: Iterable[ArgumentInput],
        attacks: Optional[Set[Tuple[str, str]]] = None,
        supports: Optional[Set[Tuple[str, str]]] = None,
    ) -> None:
        normalized: List[LegalArgument] = []
        for arg in arguments:
            if isinstance(arg, LegalArgument):
                normalized.append(arg)
            elif isinstance(arg, str):
                normalized.append(LegalArgument(id=arg, premise=arg, conclusion=arg))
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)!r}")

        self.arguments: Dict[str, LegalArgument] = {a.id: a for a in normalized}
        if len(self.arguments) != len(normalized):
            raise ValueError("argument ids must be unique")

        derived_attacks: Set[Tuple[str, str]] = set()
        derived_supports: Set[Tuple[str, str]] = set()
        for arg in normalized:
            derived_attacks.update((arg.id, target) for target in arg.attacking_args)
            derived_supports.update((arg.id, target) for target in arg.supporting_args)

        self._attacks = set(attacks) if attacks is not None else derived_attacks
        self._supports = set(supports) if supports is not None else derived_supports
        self._validate_edges(self._attacks, "attack")
        self._validate_edges(self._supports, "support")

        self.attacks_dict: Dict[str, Set[str]] = defaultdict(set)
        self.attackers_dict: Dict[str, Set[str]] = defaultdict(set)
        self.supports_dict: Dict[str, Set[str]] = defaultdict(set)
        self.supporters_dict: Dict[str, Set[str]] = defaultdict(set)

        self.attack_graph = nx.DiGraph()
        self.support_graph = nx.DiGraph()
        self.attack_graph.add_nodes_from(self.arguments)
        self.support_graph.add_nodes_from(self.arguments)

        for source, target in self._attacks:
            self.attacks_dict[source].add(target)
            self.attackers_dict[target].add(source)
            self.attack_graph.add_edge(source, target)
        for source, target in self._supports:
            self.supports_dict[source].add(target)
            self.supporters_dict[target].add(source)
            self.support_graph.add_edge(source, target)

    def _validate_edges(self, edges: Set[Tuple[str, str]], relation: str) -> None:
        known = set(self.arguments)
        dangling = [(a, b) for a, b in edges if a not in known or b not in known]
        if dangling:
            raise ValueError(f"{relation} edges reference unknown arguments: {dangling}")

    @property
    def attacks(self) -> Set[Tuple[str, str]]:
        return set(self._attacks)

    @property
    def supports(self) -> Set[Tuple[str, str]]:
        return set(self._supports)

    def is_conflict_free(self, candidate: Set[str]) -> bool:
        subset = set(candidate)
        if not subset <= set(self.arguments):
            return False
        return all((source, target) not in self._attacks for source in subset for target in subset)

    def defends(self, candidate: Set[str], argument: str) -> bool:
        if argument not in self.arguments:
            return False
        subset = set(candidate)
        if not subset <= set(self.arguments):
            return False
        return all(
            any((defender, attacker) in self._attacks for defender in subset)
            for attacker in self.attackers_dict.get(argument, set())
        )

    def characteristic(self, candidate: Set[str]) -> Set[str]:
        """Dung characteristic function F(S): arguments defended by S."""
        subset = set(candidate)
        return {arg for arg in self.arguments if self.defends(subset, arg)}

    def get_admissible_sets(self) -> List[Set[str]]:
        ids = list(self.arguments)
        admissible: List[Set[str]] = []
        for size in range(len(ids) + 1):
            for values in combinations(ids, size):
                subset = set(values)
                if self.is_conflict_free(subset) and all(self.defends(subset, a) for a in subset):
                    admissible.append(subset)
        return admissible

    def get_preferred_extensions(self) -> List[Set[str]]:
        admissible = self.get_admissible_sets()
        return [candidate for candidate in admissible if not any(candidate < other for other in admissible)]

    def get_stable_extensions(self) -> List[Set[str]]:
        ids = list(self.arguments)
        stable: List[Set[str]] = []
        for size in range(len(ids) + 1):
            for values in combinations(ids, size):
                candidate = set(values)
                if not self.is_conflict_free(candidate):
                    continue
                outside = set(ids) - candidate
                if all(any((inside, target) in self._attacks for inside in candidate) for target in outside):
                    stable.append(candidate)
        return stable

    def get_grounded_extension(self) -> Set[str]:
        current: Set[str] = set()
        while True:
            nxt = self.characteristic(current)
            if nxt == current:
                return current
            current = nxt

    def grounded_labeling(self) -> Dict[str, str]:
        accepted = self.get_grounded_extension()
        rejected = {target for source in accepted for target in self.attacks_dict.get(source, set())}
        return {
            arg: ("in" if arg in accepted else "out" if arg in rejected else "undec")
            for arg in self.arguments
        }

    def has_cycle(self) -> bool:
        return not nx.is_directed_acyclic_graph(self.attack_graph)

    def shortest_attack_path(self, start: str, goal: str) -> Optional[List[str]]:
        try:
            return nx.shortest_path(self.attack_graph, source=start, target=goal)
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return None

    def pagerank_influence(self) -> Dict[str, float]:
        if not self.arguments:
            return {}
        return nx.pagerank(self.attack_graph, alpha=0.85)

    @staticmethod
    def build_argument_graph(arguments: Sequence[LegalArgument]) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        ids = {arg.id for arg in arguments}
        for arg in arguments:
            graph.add_node(
                arg.id,
                premise=arg.premise,
                conclusion=arg.conclusion,
                confidence=arg.confidence,
            )
        for arg in arguments:
            for target in arg.attacking_args:
                if target in ids:
                    graph.add_edge(arg.id, target, relation="attacks")
            for target in arg.supporting_args:
                if target in ids:
                    graph.add_edge(arg.id, target, relation="supports")
        return graph

    def run_pyreason(self, timesteps: int = 10) -> Dict:
        """Use the optional PyReason adapter without making it a core dependency."""
        try:
            from .pyreason_connector import PyReasonConnector
        except ImportError:  # pragma: no cover - direct script fallback
            from pyreason_connector import PyReasonConnector

        payload = {
            arg_id: {
                "premise": arg.premise,
                "conclusion": arg.conclusion,
                "confidence": arg.confidence,
            }
            for arg_id, arg in self.arguments.items()
        }
        return PyReasonConnector().run_reasoning(
            arguments=payload,
            attacks=self._attacks,
            supports=self._supports,
            timesteps=timesteps,
        )

    def get_argument_status(self, arg_id: str) -> Dict:
        if arg_id not in self.arguments:
            return {"error": "Argument not found"}
        arg = self.arguments[arg_id]
        return {
            "id": arg_id,
            "premise": arg.premise,
            "conclusion": arg.conclusion,
            "confidence": arg.confidence,
            "attackers": sorted(self.attackers_dict.get(arg_id, set())),
            "attacks": sorted(self.attacks_dict.get(arg_id, set())),
            "supporters": sorted(self.supporters_dict.get(arg_id, set())),
            "supports": sorted(self.supports_dict.get(arg_id, set())),
            "grounded_label": self.grounded_labeling()[arg_id],
        }

    def export_to_dict(self) -> Dict:
        return {
            "arguments": [
                {
                    "id": arg.id,
                    "premise": arg.premise,
                    "conclusion": arg.conclusion,
                    "confidence": list(arg.confidence),
                    "source_norm_ids": sorted(arg.source_norm_ids),
                }
                for arg in self.arguments.values()
            ],
            "attacks": sorted([list(edge) for edge in self._attacks]),
            "supports": sorted([list(edge) for edge in self._supports]),
            "grounded": sorted(self.get_grounded_extension()),
            "preferred": [sorted(x) for x in self.get_preferred_extensions()],
            "stable": [sorted(x) for x in self.get_stable_extensions()],
        }
