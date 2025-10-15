import pyreason as pr
import networkx as nx
import gc
import logging
from typing import List, Dict, Set, Tuple, Optional
from threading import Lock

logger = logging.getLogger(__name__)

_pyreason_operation_lock = Lock()

class PyReasonConnector:
    """Fixed PyReason connector with proper API usage"""
    
    def __init__(self):
        self._initialized = False
        self.graph = None
    
    @staticmethod
    def force_reset():
        """Reset PyReason state"""
        try:
            with _pyreason_operation_lock:
                pr.reset()
                pr.reset_rules()
                gc.collect()
            logger.info("PyReason reset completed")
            return True
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            return False
    
    def run_reasoning(
        self,
        arguments: Dict[str, Dict],
        attacks: Set[Tuple[str, str]],
        supports: Optional[Set[Tuple[str, str]]] = None,
        timesteps: int = 10
    ) -> Dict:
        """
        Run PyReason with correct API usage.
        
        Parameters
        ----------
        arguments : Dict[str, Dict]
            Arguments with metadata including 'confidence' tuples
        attacks : Set[Tuple[str, str]]
            Attack relationships (attacker, target)
        supports : Set[Tuple[str, str]], optional
            Support relationships
        timesteps : int
            Number of reasoning timesteps
            
        Returns
        -------
        Dict
            Reasoning results with accepted/defeated arguments
        """
        if not arguments:
            return {'error': 'No arguments provided', 'converged': False}
        
        try:
            # STEP 1: Reset
            logger.info("Resetting PyReason...")
            self.force_reset()
            
            # STEP 2: Configure settings
            logger.info("Configuring settings...")
            if hasattr(pr, "settings"):
                pr.settings.verbose = False
                pr.settings.atom_trace = False
                pr.settings.memory_profile = False
                pr.settings.canonical = True
                pr.settings.inconsistency_check = False
                pr.settings.static_graph_facts = False
                pr.settings.store_interpretation_changes = True  # Need this for convergence check
            
            # STEP 3: Build graph (nodes and edges only, NO attributes)
            logger.info("Building graph...")
            self.graph = nx.DiGraph()
            
            valid_nodes = set()
            node_confidences = {}
            
            # Add nodes without attributes
            for arg_id, metadata in arguments.items():
                if not isinstance(arg_id, str) or not arg_id.strip():
                    continue
                
                confidence = metadata.get('confidence', (0.8, 1.0))
                try:
                    if isinstance(confidence, (tuple, list)) and len(confidence) >= 2:
                        lower, upper = float(confidence[0]), float(confidence[1])
                    else:
                        lower = upper = float(confidence)
                    
                    lower = max(0.0, min(1.0, lower))
                    upper = max(lower, min(1.0, upper))
                except Exception:
                    lower, upper = 0.8, 1.0
                
                self.graph.add_node(arg_id)  # Just add node, no attributes
                valid_nodes.add(arg_id)
                node_confidences[arg_id] = (lower, upper)
            
            if not valid_nodes:
                return {'error': 'No valid arguments', 'converged': False}
            
            # Add edges without attributes (PyReason will handle edge labels separately)
            for attacker, target in attacks:
                if attacker in valid_nodes and target in valid_nodes:
                    self.graph.add_edge(attacker, target)
            
            if supports:
                for supporter, supported in supports:
                    if supporter in valid_nodes and supported in valid_nodes:
                        self.graph.add_edge(supporter, supported)
            
            logger.info(f"Graph built: {len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges")
            
            # STEP 4: Load graph into PyReason
            logger.info("Loading graph...")
            with _pyreason_operation_lock:
                pr.load_graph(self.graph)
            
            # STEP 5: Add node facts (AFTER graph is loaded)
            logger.info("Adding node facts...")
            facts_added = 0
            with _pyreason_operation_lock:
                for arg_id, (lower, upper) in node_confidences.items():
                    try:
                        # Correct format: node, predicate, lower, upper, t_lower, t_upper
                        pr.add_fact(pr.Fact(arg_id, 'accepted', lower, upper, 0, timesteps-1))
                        facts_added += 1
                    except Exception as e:
                        logger.warning(f"Failed to add fact for {arg_id}: {e}")
            
            logger.info(f"Added {facts_added} node facts")
            
            # STEP 6: Add edge facts (AFTER graph is loaded)
            logger.info("Adding edge facts...")
            edges_added = 0
            with _pyreason_operation_lock:
                # Add attack edges
                for attacker, target in attacks:
                    if attacker in valid_nodes and target in valid_nodes:
                        try:
                            # Format: source, target, predicate, lower, upper, t_lower, t_upper
                            pr.add_fact(pr.Fact(attacker, target, 'attacks', 1.0, 1.0, 0, timesteps-1))
                            edges_added += 1
                        except Exception as e:
                            logger.warning(f"Failed to add attack edge {attacker}->{target}: {e}")
                
                # Add support edges
                if supports:
                    for supporter, supported in supports:
                        if supporter in valid_nodes and supported in valid_nodes:
                            try:
                                pr.add_fact(pr.Fact(supporter, supported, 'supports', 1.0, 1.0, 0, timesteps-1))
                                edges_added += 1
                            except Exception as e:
                                logger.warning(f"Failed to add support edge {supporter}->{supported}: {e}")
            
            logger.info(f"Added {edges_added} edge facts")
            
            # STEP 7: Add rules
            logger.info("Adding rules...")
            rules_added = 0
            with _pyreason_operation_lock:
                # Rule 1: Attack causes defeat
                try:
                    pr.add_rule(pr.Rule(
                        'defeated(Y) : [1.0, 1.0] <- attacks(X, Y), accepted(X) : [0.5, 1.0]',
                        'defeat_rule'
                    ))
                    rules_added += 1
                except Exception as e:
                    logger.error(f"Failed to add defeat rule: {e}")
                
                # Rule 2: Reinstatement
                try:
                    pr.add_rule(pr.Rule(
                        'reinstated(Y) : [1.0, 1.0] <- attacks(X, Y), defeated(X)',
                        'reinstate_rule'
                    ))
                    rules_added += 1
                except Exception as e:
                    logger.warning(f"Failed to add reinstatement rule: {e}")
                
                # Rule 3: Support strengthens
                if supports:
                    try:
                        pr.add_rule(pr.Rule(
                            'strengthened(Y) : [1.0, 1.0] <- supports(X, Y), accepted(X) : [0.7, 1.0]',
                            'support_rule'
                        ))
                        rules_added += 1
                    except Exception as e:
                        logger.warning(f"Failed to add support rule: {e}")
            
            logger.info(f"Added {rules_added} rules")
            
            # STEP 8: Run reasoning
            logger.info(f"Running reasoning for {timesteps} timesteps...")
            with _pyreason_operation_lock:
                interpretation = pr.reason(timesteps=timesteps)
            
            # STEP 9: Extract results
            logger.info("Extracting results...")
            results = self._extract_results(interpretation, valid_nodes)
            
            logger.info("PyReason reasoning completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"PyReason reasoning failed: {e}", exc_info=True)
            self.force_reset()
            return {
                'error': str(e),
                'converged': False,
                'accepted': {},
                'defeated': {}
            }
    
    def _extract_results(self, interpretation, arg_ids: Set[str]) -> Dict:
        """Extract results from PyReason interpretation"""
        results = {
            'accepted': {},
            'defeated': {},
            'reinstated': {},
            'strengthened': {},
            'converged': False
        }
        
        try:
            if interpretation is None:
                logger.warning("Interpretation is None")
                return results
            
            # Get interpretation as dictionary
            interp_dict = interpretation.get_dict()
            
            if not interp_dict:
                logger.warning("Empty interpretation")
                return results
            
            # Get final timestep
            timesteps = sorted(interp_dict.keys())
            final_t = timesteps[-1]
            final_data = interp_dict[final_t]
            
            # Extract predicates for each argument
            for arg_id in arg_ids:
                if arg_id not in final_data:
                    continue
                
                node_data = final_data[arg_id]
                
                for predicate in ['accepted', 'defeated', 'reinstated', 'strengthened']:
                    if predicate in node_data:
                        value = node_data[predicate]
                        
                        if isinstance(value, (list, tuple)) and len(value) >= 2:
                            results[predicate][arg_id] = (float(value[0]), float(value[1]))
                        elif isinstance(value, (int, float)):
                            v = float(value)
                            results[predicate][arg_id] = (v, v)
            
            # Check convergence
            if len(timesteps) >= 2:
                results['converged'] = (interp_dict[timesteps[-1]] == interp_dict[timesteps[-2]])
            
        except Exception as e:
            logger.error(f"Error extracting results: {e}", exc_info=True)
        
        return results


# Test function
def test_connector():
    """Test the fixed PyReason connector"""
    print("Testing PyReason Connector...")
    
    connector = PyReasonConnector()
    
    # Test arguments
    arguments = {
        'A': {'premise': 'P1', 'conclusion': 'C1', 'confidence': (0.9, 1.0)},
        'B': {'premise': 'P2', 'conclusion': 'C2', 'confidence': (0.8, 1.0)},
        'C': {'premise': 'P3', 'conclusion': 'C3', 'confidence': (0.7, 0.9)}
    }
    
    # A attacks B, B attacks C
    attacks = {('A', 'B'), ('B', 'C')}
    
    results = connector.run_reasoning(arguments, attacks, timesteps=5)
    
    print("\nResults:")
    print(f"Converged: {results.get('converged')}")
    print(f"Accepted: {results.get('accepted')}")
    print(f"Defeated: {results.get('defeated')}")
    print(f"Reinstated: {results.get('reinstated')}")
    
    return results


if __name__ == "__main__":
    test_connector()
