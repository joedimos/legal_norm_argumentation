from dataclasses import dataclass
from typing import Set, Tuple, Dict, List, Optional, FrozenSet
import logging
from collections import defaultdict
import networkx as nx

logger = logging.getLogger(__name__)

try:
    from .pyreason_integration import PyReasonConnector
except ImportError:
    from pyreason_integration import PyReasonConnector


# Data classes

@dataclass(frozen=True)
class LegalArgument:
    """
    Represents a legal argument with a premise, conclusion, and links
    to supporting/attacking arguments.
    """
    id: str
    premise: str
    conclusion: str
    confidence: Tuple[float, float] = (0.8, 1.0)
    supporting_args: FrozenSet[str] = frozenset()
    attacking_args: FrozenSet[str] = frozenset()
    sequent_type: str = "hypothesis"


@dataclass(frozen=True)
class Sequent:
    """
    Represents a sequent in argumentation, with antecedents leading to consequents.
    """
    antecedents: FrozenSet[str]
    consequents: FrozenSet[str]
    confidence: float
    is_contrapositive: bool = False


# Optimized AAF + PyReason

class OptimizedAAF:
    """
    Optimized Abstract Argumentation Framework with PyReason integration.
    
    Implements Dung-style semantics (admissible, preferred, stable extensions)
    and integrates with PyReason for probabilistic reasoning over arguments.
    """
    
    def __init__(
        self,
        arguments: List[LegalArgument],
        attacks: Optional[Set[Tuple[str, str]]] = None,
        supports: Optional[Set[Tuple[str, str]]] = None,
    ):
        """
        Parameters
        ----------
        arguments : List[LegalArgument]
            Full argument objects (we'll keep id->object mapping internally).
        attacks : Optional[Set[Tuple[str, str]]]
            Optional explicit (attacker, target) pairs. If None, derived from arguments' attacking_args.
        supports : Optional[Set[Tuple[str, str]]]
            Optional explicit (arg, supports_arg) pairs. If None, derived from arguments' supporting_args.
        """
        # Keep full objects for metadata; use keys for AAF ops
        self.arguments: Dict[str, LegalArgument] = {a.id: a for a in arguments}

        # Graphs / indices
        self.attack_graph = nx.DiGraph()
        self.attacks_dict: Dict[str, Set[str]] = defaultdict(set)
        self.attackers_dict: Dict[str, Set[str]] = defaultdict(set)

        # Add nodes
        self.attack_graph.add_nodes_from(self.arguments.keys())

        # Build edges (attacks/supports)
        derived_attacks: Set[Tuple[str, str]] = set()
        derived_supports: Set[Tuple[str, str]] = set()

        for a in arguments:
            for t in a.attacking_args:
                if t in self.arguments:
                    derived_attacks.add((a.id, t))
            for s in a.supporting_args:
                if s in self.arguments:
                    derived_supports.add((a.id, s))

        self._attacks: Set[Tuple[str, str]] = attacks if attacks is not None else derived_attacks
        self._supports: Set[Tuple[str, str]] = supports if supports is not None else derived_supports

        # Populate attack indices + graph edges
        for attacker, target in self._attacks:
            if attacker in self.arguments and target in self.arguments:
                self.attacks_dict[attacker].add(target)
                self.attackers_dict[target].add(attacker)
                self.attack_graph.add_edge(attacker, target, relation="attacks")

        # Add support edges to the graph (not used by Dung semantics, but useful for analysis / PyReason)
        for src, dst in self._supports:
            if src in self.arguments and dst in self.arguments:
                # Avoid overwriting an existing attack edge attribute
                if not self.attack_graph.has_edge(src, dst):
                    self.attack_graph.add_edge(src, dst, relation="supports")

    
    # Dung-style semantics operations

    def is_conflict_free(self, S: Set[str]) -> bool:
        """Check if set S is conflict-free (no internal attacks)"""
        subset = S & set(self.arguments.keys())
        return all(not (self.attacks_dict.get(arg, set()) & subset) for arg in subset)

    def defends(self, S: Set[str], a: str) -> bool:
        """Check if set S defends argument a against all attackers"""
        if a not in self.arguments:
            return False
        subset = S & set(self.arguments.keys())
        for attacker in self.attackers_dict.get(a, set()):
            if not any(attacker in self.attacks_dict.get(defender, set()) for defender in subset):
                return False
        return True

    def get_admissible_sets(self) -> List[Set[str]]:
        """Compute all admissible sets"""
        admissible: List[Set[str]] = []
        ids = list(self.arguments.keys())
        n = len(ids)

        for mask in range(1, 1 << n):
            subset = {ids[i] for i in range(n) if (mask >> i) & 1}
            if self.is_conflict_free(subset) and all(self.defends(subset, a) for a in subset):
                admissible.append(subset)
        return admissible

    def get_preferred_extensions(self) -> List[Set[str]]:
        """Compute preferred extensions (maximal admissible sets)"""
        admissible = self.get_admissible_sets()
        return [s for s in admissible if not any(s < t for t in admissible)]

    def get_stable_extensions(self) -> List[Set[str]]:
        """Compute stable extensions"""
        stable: List[Set[str]] = []
        ids = list(self.arguments.keys())
        n = len(ids)

        for mask in range(1, 1 << n):
            subset = {ids[i] for i in range(n) if (mask >> i) & 1}
            if self.is_conflict_free(subset):
                outside = set(ids) - subset
                if all(any(out in self.attacks_dict.get(defender, set()) for defender in subset) for out in outside):
                    stable.append(subset)
        return stable

    def has_cycle(self) -> bool:
        """Check if the attack graph contains cycles"""
        try:
            return any(len(scc) > 1 for scc in nx.strongly_connected_components(self.attack_graph))
        except Exception as e:
            logger.error("Error checking cycles: %s", e)
            return False

    def shortest_attack_path(self, start: str, goal: str) -> Optional[List[str]]:
        """Find shortest attack path between two arguments"""
        try:
            # Only consider the attack subgraph for shortest attack paths
            attack_only = nx.DiGraph(
                ((u, v, d) for u, v, d in self.attack_graph.edges(data=True) if d.get("relation") == "attacks")
            )
            return nx.shortest_path(attack_only, source=start, target=goal)
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return None
        except Exception as e:
            logger.error("Error computing shortest path: %s", e)
            return None

    def pagerank_influence(self) -> Dict[str, float]:
        """Compute PageRank influence scores for arguments"""
        try:
            # PageRank on attack-only edges (typical in AAF influence analysis)
            attack_only = nx.DiGraph(
                ((u, v) for u, v, d in self.attack_graph.edges(data=True) if d.get("relation") == "attacks")
            )
            # Ensure all nodes are present even if isolated
            attack_only.add_nodes_from(self.arguments.keys())
            return nx.pagerank(attack_only, alpha=0.85)
        except Exception as e:
            logger.error("PageRank failed: %s", e)
            return {arg_id: 0.0 for arg_id in self.arguments.keys()}

    @staticmethod
    def build_argument_graph(arguments: List[LegalArgument]) -> nx.DiGraph:
        """
        Build a NetworkX DiGraph from a list of LegalArgument objects.
        
        Parameters
        ----------
        arguments : List[LegalArgument]
            List of legal arguments
            
        Returns
        -------
        nx.DiGraph
            Directed graph representation of the argumentation framework
        """
        G = nx.DiGraph()
        for arg in arguments:
            G.add_node(arg.id, premise=arg.premise, conclusion=arg.conclusion)
            for s in arg.supporting_args:
                G.add_edge(arg.id, s, relation="supports")
            for a in arg.attacking_args:
                G.add_edge(arg.id, a, relation="attacks")
        return G  # FIXED: Added missing return statement
        
    def run_pyreason(self, timesteps: int = 10) -> Dict:
        """
        Push the AAF (arguments + attacks + supports + confidences) into PyReason
        via PyReasonConnector and return its results.

        Parameters
        ----------
        timesteps : int, optional
            Number of reasoning timesteps (default: 10)

        Returns
        -------
        Dict
            PyReason reasoning results including acceptance, defeat, and convergence info
        """
        try:
            # Prepare rich argument payload for PyReason (id -> metadata)
            arg_payload: Dict[str, Dict] = {}
            for arg_id, arg in self.arguments.items():
                arg_payload[arg_id] = {
                    "premise": arg.premise,
                    "conclusion": arg.conclusion,
                    "confidence": arg.confidence,
                    "sequent_type": arg.sequent_type,
                }

            connector = PyReasonConnector()

            # Call PyReasonConnector with all parameters
            logger.info(f"Running PyReason with {len(arg_payload)} arguments, "
                       f"{len(self._attacks)} attacks, {len(self._supports)} supports")
            
            results = connector.run_reasoning(
                arguments=arg_payload,
                attacks=set(self._attacks),
                supports=set(self._supports),
                timesteps=timesteps
            )
            
            logger.info(f"PyReason completed. Results: {len(results)} entries")
            return results

        except TypeError as te:
            # Handle API mismatch
            logger.error(f"PyReasonConnector API mismatch: {te}. "
                        "Check that your PyReasonConnector.run_reasoning() signature matches.")
            return {"error": "API_mismatch", "details": str(te)}
        except Exception as e:
            logger.error(f"PyReason run failed: {e}", exc_info=True)
            return {"error": "reasoning_failed", "details": str(e)}
    
    # Additional utility methods
    
    def get_argument_status(self, arg_id: str) -> Dict:
        """
        Get comprehensive status of an argument including:
        - Its attackers and targets
        - Support relationships
        - Confidence bounds
        
        Parameters
        ----------
        arg_id : str
            Argument identifier
            
        Returns
        -------
        Dict
            Status information for the argument
        """
        if arg_id not in self.arguments:
            return {"error": "Argument not found"}
        
        arg = self.arguments[arg_id]
        return {
            "id": arg_id,
            "premise": arg.premise,
            "conclusion": arg.conclusion,
            "confidence": arg.confidence,
            "attackers": list(self.attackers_dict.get(arg_id, set())),
            "attacks": list(self.attacks_dict.get(arg_id, set())),
            "supporting": list(arg.supporting_args),
            "attacking": list(arg.attacking_args),
            "sequent_type": arg.sequent_type
        }
    
    def export_to_dict(self) -> Dict:
        """
        Export the entire AAF to a dictionary format suitable for serialization.
        
        Returns
        -------
        Dict
            Complete AAF representation
        """
        return {
            "arguments": [
                {
                    "id": arg.id,
                    "premise": arg.premise,
                    "conclusion": arg.conclusion,
                    "confidence": arg.confidence,
                    "supporting_args": list(arg.supporting_args),
                    "attacking_args": list(arg.attacking_args),
                    "sequent_type": arg.sequent_type
                }
                for arg in self.arguments.values()
            ],
            "attacks": list(self._attacks),
            "supports": list(self._supports)
        }
