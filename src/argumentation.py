from dataclasses import dataclass, field
from typing import Set, Tuple, Dict, List, Optional, FrozenSet
import networkx as nx
from collections import defaultdict
from .pyreason_integration import PyReasonConnector
import logging

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class LegalArgument:
    """
    Represents a legal argument with a premise, conclusion, and links
    to supporting/attacking arguments.
    """
    id: str
    premise: str
    conclusion: str
    confidence: Tuple[float, float]  # (lower, upper) confidence interval
    supporting_args: FrozenSet[str] = frozenset()
    attacking_args: FrozenSet[str] = frozenset()
    sequent_type: str = "hypothesis"  # {"axiom","hypothesis","conclusion","contrapositive"}


@dataclass(frozen=True)
class Sequent:
    """
    Represents a sequent in argumentation, with antecedents leading to consequents.
    """
    antecedents: FrozenSet[str]
    consequents: FrozenSet[str]
    confidence: float
    is_contrapositive: bool = False

class OptimizedAAF:
    """
    Optimized Abstract Argumentation Framework (Dung-style) with NetworkX support.
    Provides computation of admissible, preferred, and stable extensions.
    """

    def __init__(self, arguments: Set[str], attacks: Set[Tuple[str, str]]):
        self.arguments: Set[str] = set(arguments)
        self.attack_graph = nx.DiGraph()
        self.attacks_dict: Dict[str, Set[str]] = defaultdict(set)
        self.attackers_dict: Dict[str, Set[str]] = defaultdict(set)

        self.attack_graph.add_nodes_from(self.arguments)
        for attacker, target in attacks:
            if attacker in self.arguments and target in self.arguments:
                self.attacks_dict[attacker].add(target)
                self.attackers_dict[target].add(attacker)
                self.attack_graph.add_edge(attacker, target)

    def is_conflict_free(self, S: Set[str]) -> bool:
        """Check whether a set of arguments is conflict-free."""
        subset = S & self.arguments
        return all(
            not (self.attacks_dict.get(arg, set()) & subset)
            for arg in subset
        )

    def defends(self, S: Set[str], a: str) -> bool:
        """Check whether set S defends argument a against its attackers."""
        if a not in self.arguments:
            return False
        subset = S & self.arguments
        for attacker in self.attackers_dict.get(a, set()):
            if not any(
                attacker in self.attacks_dict.get(defender, set())
                for defender in subset
            ):
                return False
        return True

    def get_admissible_sets(self) -> List[Set[str]]:
        """
        Enumerate all admissible sets: conflict-free and defending themselves.
        Note: uses brute-force powerset search, exponential in |arguments|.
        """
        admissible: List[Set[str]] = []
        args = list(self.arguments)
        n = len(args)

        for mask in range(1, 1 << n):  # skip empty set
            subset = {args[i] for i in range(n) if (mask >> i) & 1}
            if self.is_conflict_free(subset) and all(self.defends(subset, a) for a in subset):
                admissible.append(subset)
        return admissible

    def get_preferred_extensions(self) -> List[Set[str]]:
        """Return maximal (w.r.t. set inclusion) admissible sets = preferred extensions."""
        admissible = self.get_admissible_sets()
        return [s for s in admissible if not any(s < t for t in admissible)]

    def get_stable_extensions(self) -> List[Set[str]]:
        """
        Return stable extensions: conflict-free sets that attack every outside argument.
        """
        stable: List[Set[str]] = []
        args = list(self.arguments)
        n = len(args)

        for mask in range(1, 1 << n):  # skip empty set
            subset = {args[i] for i in range(n) if (mask >> i) & 1}
            if self.is_conflict_free(subset):
                outside = set(args) - subset
                if all(
                    any(out in self.attacks_dict.get(defender, set()) for defender in subset)
                    for out in outside
                ):
                    stable.append(subset)
        return stable

    def has_cycle(self) -> bool:
        """Check if the attack graph contains cycles."""
        try:
            return any(len(scc) > 1 for scc in nx.strongly_connected_components(self.attack_graph))
        except Exception as e:
            logger.error("Error checking cycles: %s", e)
            return False

    def shortest_attack_path(self, start: str, goal: str) -> Optional[List[str]]:
        """Find the shortest attack path between two arguments, if it exists."""
        try:
            return nx.shortest_path(self.attack_graph, source=start, target=goal)
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return None
        except Exception as e:
            logger.error("Error computing shortest path: %s", e)
            return None

    def pagerank_influence(self) -> Dict[str, float]:
        """Compute PageRank centrality scores for arguments in the attack graph."""
        try:
            return nx.pagerank(self.attack_graph, alpha=0.85)
        except Exception as e:
            logger.error("PageRank failed: %s", e)
            return {arg: 0.0 for arg in self.arguments}

    def build_argument_graph(arguments: List[LegalArgument]):
    G = nx.DiGraph()
    for arg in arguments:
        G.add_node(arg.id, premises=arg.premises, conclusion=arg.conclusion)
        for s in arg.supports:
            G.add_edge(arg.id, s, relation="supports")
        for a in arg.attacks:
            G.add_edge(arg.id, a, relation="attacks")
    return G
