import pyreason as pr
from typing import List, Dict, Set, Tuple
import logging

logger = logging.getLogger(__name__)

class PyReasonConnector:
    
    def __init__(self):
        self.graph = None
        
    def run_reasoning(
        self,
        arguments: Dict[str, Dict],
        attacks: Set[Tuple[str, str]],
        supports: Set[Tuple[str, str]] = None,
        timesteps: int = 10
    ) -> Dict:
        """
        Run PyReason on the argument framework.
        
        Parameters
        ----------
        arguments : Dict[str, Dict]
            Argument ID -> metadata dict with 'confidence', 'premise', etc.
        attacks : Set[Tuple[str, str]]
            Set of (attacker_id, target_id) tuples
        supports : Set[Tuple[str, str]], optional
            Set of (supporter_id, supported_id) tuples
        timesteps : int
            Number of reasoning timesteps
            
        Returns
        -------
        Dict with reasoning results
        """
        try:
            # Reset PyReason state
            pr.reset()
            pr.reset_rules()
            
            # 1. Build graph
            self.graph = pr.Graph()
            
            # Add argument nodes
            for arg_id in arguments.keys():
                self.graph.add_node(arg_id)
                logger.info(f"Added node: {arg_id}")
            
            # Add attack edges
            for attacker, target in attacks:
                self.graph.add_edge(attacker, target, label='attacks')
                logger.info(f"Added attack: {attacker} -> {target}")
            
            # Add support edges if provided
            if supports:
                for supporter, supported in supports:
                    self.graph.add_edge(supporter, supported, label='supports')
                    logger.info(f"Added support: {supporter} -> {supported}")
            
            # Load graph into PyReason
            pr.load_graph(self.graph)
            
            # 2. Define rules for legal argumentation
            # Rule: An argument is defeated if it's attacked by an accepted argument
            pr.add_rule(pr.Rule(
                'defeated(Y) <- attacks(X, Y), accepted(X)',
                'argument_defeated',
                infer_edges=True
            ))
            
            # Rule: An argument is reinstated if its attacker is defeated
            pr.add_rule(pr.Rule(
                'reinstated(Y) <- attacks(X, Y), defeated(X)',
                'argument_reinstated',
                infer_edges=True
            ))
            
            # Rule: Supported arguments gain strength
            if supports:
                pr.add_rule(pr.Rule(
                    'strengthened(Y) <- supports(X, Y), accepted(X)',
                    'argument_strengthened',
                    infer_edges=True
                ))
            
            # 3. Set initial facts (argument acceptance based on confidence)
            for arg_id, metadata in arguments.items():
                confidence = metadata.get('confidence', (0.8, 1.0))
                
                # Convert to PyReason bounds format
                if isinstance(confidence, tuple):
                    bounds = list(confidence)
                else:
                    bounds = [confidence, confidence]
                
                # Add initial acceptance fact
                pr.add_fact(pr.Fact(
                    arg_id,
                    'accepted',
                    bounds,
                    timestep=0
                ))
                logger.info(f"Set initial fact: {arg_id} accepted with bounds {bounds}")
            
            # 4. Run reasoning
            logger.info(f"Running PyReason for {timesteps} timesteps...")
            interpretation = pr.reason(timesteps=timesteps)
            
            # 5. Extract results
            results = self._extract_results(interpretation, arguments.keys())
            
            logger.info("PyReason reasoning completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"PyReason reasoning failed: {e}", exc_info=True)
            return {}
    
    def _extract_results(self, interpretation, arg_ids) -> Dict:
        """Extract and format PyReason results"""
        results = {
            'accepted': {},
            'defeated': {},
            'reinstated': {},
            'convergence': {}
        }
        
        try:
            # Get final timestep
            final_timestep = max(interpretation.keys())
            
            for arg_id in arg_ids:
                # Extract acceptance bounds
                if arg_id in interpretation[final_timestep]:
                    node_data = interpretation[final_timestep][arg_id]
                    
                    if 'accepted' in node_data:
                        results['accepted'][arg_id] = node_data['accepted']
                    
                    if 'defeated' in node_data:
                        results['defeated'][arg_id] = node_data['defeated']
                    
                    if 'reinstated' in node_data:
                        results['reinstated'][arg_id] = node_data['reinstated']
            
            # Check convergence
            results['converged'] = self._check_convergence(interpretation)
            
        except Exception as e:
            logger.error(f"Error extracting results: {e}")
        
        return results
    
    def _check_convergence(self, interpretation) -> bool:
        """Check if reasoning has converged"""
        if len(interpretation) < 2:
            return False
        
        timesteps = sorted(interpretation.keys())
        last = interpretation[timesteps[-1]]
        second_last = interpretation[timesteps[-2]]
        
        # Simple convergence check: compare last two timesteps
        return last == second_last
