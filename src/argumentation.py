from dataclasses import dataclass, field
from typing import Set, Tuple, Dict, List, Optional
import networkx as nx
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# -------------------------
# Dataclasses
# -------------------------
@dataclass
class LegalArgument:
    id: str
    premise: str
    conclusion: str
    confidence: Tuple[float, float]
    supporting_args: Set[str] = field(default_factory=set)
    attacking_args: Set[str] = field(default_factory=set)
    sequent_type: str = "hypothesis"  # "axiom","hypothesis","conclusion","contrapositive"

@dataclass
class Sequent:
    antecedents: Set[str]
    consequents: Set[str]
    confidence: float
    is_contrapositive: bool = False

class OptimizedAAF:
    def __init__(self, arguments: Set[str], attacks: Set[Tuple[str, str]]):
        self.arguments: Set[str] = set(arguments)
        self.attack_graph = nx.DiGraph()
        self.attacks_dict = defaultdict(set)
        self.attackers_dict = defaultdict(set)
        self.attack_graph.add_nodes_from(self.arguments)
        for a, b in attacks:
            if a in self.arguments and b in self.arguments:
                self.attacks_dict[a].add(b)
                self.attackers_dict[b].add(a)
                self.attack_graph.add_edge(a, b)

    def is_conflict_free(self, S: Set[str]) -> bool:
        subset_args = set(S).intersection(self.arguments)
        for arg in subset_args:
            if len(self.attacks_dict.get(arg, set()).intersection(subset_args)) > 0:
                return False
        return True

    def defends(self, S: Set[str], a: str) -> bool:
        if a not in self.arguments:
            return False
        subset_args = set(S).intersection(self.arguments)
        attackers_of_a = self.attackers_dict.get(a, set())
        for attacker in attackers_of_a:
            defended = False
            for defender in subset_args:
                if attacker in self.attacks_dict.get(defender, set()):
                    defended = True
                    break
            if not defended:
                return False
        return True

    def get_admissible_sets(self) -> List[Set[str]]:
        admissible = []
        args = list(self.arguments)
        n = len(args)
        # brute force power set (skip empty set)
        for mask in range(1, 1 << n):
            subset = {args[i] for i in range(n) if (mask >> i) & 1}
            if self.is_conflict_free(subset) and all(self.defends(subset, a) for a in subset):
                admissible.append(subset)
        return admissible

    def get_preferred_extensions(self) -> List[Set[str]]:
        adm = self.get_admissible_sets()
        preferred = []
        for s in adm:
            if not any(s < t for t in adm):
                preferred.append(s)
        return preferred

    def get_stable_extensions(self) -> List[Set[str]]:
        stable = []
        args = set(self.arguments)
        n = len(args)
        arg_list = list(args)
        for mask in range(1, 1 << n):
            subset = {arg_list[i] for i in range(n) if (mask >> i) & 1}
            if self.is_conflict_free(subset):
                outside = args - subset
                all_attacked = all(any(out in self.attacks_dict.get(att, set()) for att in subset) for out in outside)
                if all_attacked:
                    stable.append(subset)
        return stable

    # Graph analysis methods
    def has_cycle(self) -> bool:
        try:
            sccs = list(nx.strongly_connected_components(self.attack_graph))
            return any(len(s) > 1 for s in sccs)
        except Exception as e:
            logger.error("Error checking cycles: %s", e)
            return False

    def shortest_attack_path(self, start: str, goal: str) -> Optional[List[str]]:
        try:
            return nx.shortest_path(self.attack_graph, source=start, target=goal)
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return None
        except Exception as e:
            logger.error("Error shortest path: %s", e)
            return None

    def pagerank_influence(self) -> Dict[str, float]:
        try:
            return nx.pagerank(self.attack_graph, alpha=0.85)
        except Exception as e:
            logger.error("PageRank failed: %s", e)
            return {a: 0.0 for a in self.arguments}
