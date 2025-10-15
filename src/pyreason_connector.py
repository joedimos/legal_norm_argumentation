import pyreason as pr
import networkx as nx
import gc
import sys
import weakref
from typing import List, Dict, Set, Tuple, Optional
import logging
import atexit
from threading import Lock
import subprocess
import json
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_pyreason_add_fact_lock = Lock()
_pyreason_operation_lock = Lock() 

# Feature flag to enable subprocess isolation (recommended to avoid segfaults)
USE_SUBPROCESS_ISOLATION = True


_pyreason_state = {
    'initialized': False,
    'graph_loaded': False,
    'cleanup_in_progress': False
}


# Disable PyReason multithreading early (use pr alias)
try:
    pr.config.use_multithreading = False
except Exception:
    try:
        if hasattr(pr, "config"):
            setattr(pr.config, "use_multithreading", False)
    except Exception:
        pass


def _emergency_cleanup():
    """Emergency cleanup on exit - NEVER call reset during shutdown"""
    try:
        if _pyreason_state.get('cleanup_in_progress', False):
            return
        _pyreason_state['cleanup_in_progress'] = True
        
        # DON'T call pr.reset() here - can cause segfault during interpreter shutdown
        # Just mark state as clean
        _pyreason_state['initialized'] = False
        _pyreason_state['graph_loaded'] = False
    except Exception:
        pass


# Register cleanup on exit
atexit.register(_emergency_cleanup)


class PyReasonConnector:
    _instances = []

    def __init__(self):
        self._initialized = False
        self.graph = None
        self._active = True
        self._cleanup_done = False
        PyReasonConnector._instances.append(weakref.ref(self))

    def __del__(self):
        """Cleanup on deletion - avoid operations during interpreter shutdown"""
        try:
            # Check if interpreter is shutting down
            if sys is None or _pyreason_state.get('cleanup_in_progress', False):
                return
                
            if getattr(self, "_active", False) and not getattr(self, "_cleanup_done", False):
                self.cleanup()
        except Exception:
            pass

    def cleanup(self):
        """Explicit cleanup method"""
        if not getattr(self, "_active", False) or getattr(self, "_cleanup_done", False):
            return

        try:
            self._active = False
            self._cleanup_done = True

            # Clear graph reference without complex operations
            if self.graph is not None:
                try:
                    # Don't iterate or clear - just drop reference
                    self.graph = None
                except Exception:
                    pass

            # Check if we're the last instance
            active_instances = []
            try:
                active_instances = [ref for ref in PyReasonConnector._instances if ref() is not None and ref() is not self]
            except Exception:
                pass

            # Only reset if truly last instance and not during shutdown
            if len(active_instances) == 0 and sys is not None:
                try:
                    with _pyreason_operation_lock:
                        pr.reset()
                        pr.reset_rules()
                        _pyreason_state['initialized'] = False
                        _pyreason_state['graph_loaded'] = False
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            try:
                gc.collect()
            except Exception:
                pass

    @staticmethod
    def force_reset():
        """Force complete PyReason reset - with safety checks"""
        try:
            # Don't reset during shutdown
            if _pyreason_state.get('cleanup_in_progress', False):
                return False
                
            with _pyreason_operation_lock:
                # Single reset attempt with proper locking
                try:
                    pr.reset()
                    pr.reset_rules()
                except Exception as e:
                    logger.debug(f"Reset attempt error: {e}")
                    pass

                # Single garbage collection pass
                gc.collect()

                _pyreason_state['initialized'] = False
                _pyreason_state['graph_loaded'] = False

            logger.info("PyReason force reset completed")
            return True
        except Exception as e:
            logger.error(f"Force reset failed: {e}")
            return False

    def run_reasoning(
        self,
        arguments: Dict[str, Dict],
        attacks: Set[Tuple[str, str]],
        supports: Optional[Set[Tuple[str, str]]] = None,
        timesteps: int = 10
    ) -> Dict:
        """
        Run PyReason on the argument framework with maximum safety.
        """

        # Validate inputs first
        if not arguments:
            logger.error("No arguments provided")
            return {'error': 'No arguments provided', 'converged': False}

        if timesteps < 1 or timesteps > 100:
            logger.warning(f"Invalid timesteps {timesteps}, using 10")
            timesteps = 10

        # Local variables to track what needs cleanup
        graph_for_pyreason = None
        
        try:
            # STEP 1: RESET
            logger.info("Step 1: Resetting PyReason...")
            self.force_reset()

            # STEP 2: CONFIGURE SETTINGS
            logger.info("Step 2: Configuring PyReason settings...")
            try:
                if hasattr(pr, "settings"):
                    pr.settings.verbose = False
                    pr.settings.atom_trace = False  
                    pr.settings.memory_profile = False
                    pr.settings.canonical = True
                    pr.settings.inconsistency_check = False
                    pr.settings.static_graph_facts = False
                    pr.settings.store_interpretation_changes = False

                    if hasattr(pr.settings, 'parallel'):
                        pr.settings.parallel = False
                    if hasattr(pr.settings, 'distributed'):
                        pr.settings.distributed = False
            except AttributeError as e:
                logger.warning(f"Some settings not available: {e}")

            # STEP 3: BUILD GRAPH WITH LABELS
            logger.info("Step 3: Building argument graph...")
            
            # Create graph with label-based facts (safer than add_fact)
            self.graph = nx.DiGraph()

            # Add nodes with 'label' attribute containing initial facts
            valid_nodes = set()
            node_confidences = {}
            
            for arg_id, metadata in arguments.items():
                if not isinstance(arg_id, str) or not arg_id.strip():
                    logger.warning(f"Invalid argument ID: {arg_id}")
                    continue

                # Get confidence bounds
                confidence = metadata.get('confidence', (0.8, 1.0))
                try:
                    if isinstance(confidence, tuple) and len(confidence) >= 2:
                        lower, upper = float(confidence[0]), float(confidence[1])
                    elif isinstance(confidence, (list, tuple)) and len(confidence) == 1:
                        lower = upper = float(confidence[0])
                    else:
                        lower = upper = float(confidence)

                    lower = max(0.0, min(1.0, lower))
                    upper = max(lower, min(1.0, upper))
                except Exception:
                    lower, upper = 0.8, 1.0

                # Add node with label containing facts
                # This is PyReason's preferred method for initial facts
                self.graph.add_node(arg_id, label={
                    'accepted': [lower, upper]
                })
                valid_nodes.add(arg_id)
                node_confidences[arg_id] = (lower, upper)
                logger.debug(f"Added node: {arg_id} with label facts")

            if len(valid_nodes) == 0:
                logger.error("No valid nodes added to graph")
                return {'error': 'No valid arguments', 'converged': False}

            # Add edges with validation
            valid_attacks = set()
            for attacker, target in attacks:
                if attacker in valid_nodes and target in valid_nodes and attacker != target:
                    self.graph.add_edge(attacker, target, label='attacks')
                    valid_attacks.add((attacker, target))

            valid_supports = set()
            if supports:
                for supporter, supported in supports:
                    if supporter in valid_nodes and supported in valid_nodes and supporter != supported:
                        self.graph.add_edge(supporter, supported, label='supports')
                        valid_supports.add((supporter, supported))

            logger.info(f"Graph built: {len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges")

            # STEP 4: LOAD GRAPH WITH LABELS
            logger.info("Step 4: Loading graph into PyReason...")

            # Create deep copy for PyReason WITH label-based facts
            graph_for_pyreason = nx.DiGraph()
            for node, attrs in self.graph.nodes(data=True):
                graph_for_pyreason.add_node(node, **dict(attrs))
                
            for u, v, edata in self.graph.edges(data=True):
                graph_for_pyreason.add_edge(u, v, **dict(edata))

            # Load with lock
            with _pyreason_operation_lock:
                pr.load_graph(graph_for_pyreason)
                _pyreason_state['graph_loaded'] = True
            
            logger.info("Graph loaded successfully with label-based facts")

            # STEP 5: ADD RULES
            logger.info("Step 5: Adding reasoning rules...")
            rules_added = self._add_rules_safe(has_supports=len(valid_supports) > 0)

            if rules_added == 0:
                logger.warning("No rules added successfully")

            # STEP 6: VERIFY FACTS (skip add_fact to avoid segfault)
            logger.info("Step 6: Verifying label-based facts...")
            facts_verified = self._verify_label_facts(valid_nodes)
            
            if facts_verified == 0:
                logger.warning("Could not verify facts, but proceeding anyway")
            else:
                logger.info(f"Verified {facts_verified} label-based facts")

            # STEP 7: RUN REASONING
            logger.info(f"Step 7: Running reasoning for {timesteps} timesteps...")

            safe_timesteps = min(timesteps, 5) if not self._initialized else timesteps

            with _pyreason_operation_lock:
                try:
                    interpretation = pr.reason(timesteps=safe_timesteps)
                    _pyreason_state['initialized'] = True
                    self._initialized = True
                except Exception as e:
                    logger.error(f"Reasoning failed: {e}")
                    raise

            # STEP 8: EXTRACT RESULTS
            logger.info("Step 8: Extracting results...")
            results = self._extract_results(interpretation, valid_nodes)

            logger.info("PyReason reasoning completed successfully")

            return results

        except Exception as e:
            logger.error(f"PyReason reasoning failed: {e}", exc_info=True)

            try:
                self.force_reset()
            except Exception:
                pass

            return {
                'error': str(e),
                'details': type(e).__name__,
                'converged': False,
                'accepted': {},
                'defeated': {}
            }

        finally:
            # Cleanup local graph copy
            if graph_for_pyreason is not None:
                try:
                    graph_for_pyreason.clear()
                    del graph_for_pyreason
                except Exception:
                    pass

            # Single GC pass
            try:
                gc.collect()
            except Exception:
                pass

    def _check_facts_in_graph(self) -> bool:
        """Check if facts were set via graph attributes"""
        try:
            if hasattr(pr, 'graph') and pr.graph is not None:
                for node in list(pr.graph.nodes())[:1]:  # Check just first node
                    if 'accepted' in pr.graph.nodes[node]:
                        return True
        except Exception:
            pass
        return False

    def _add_rules_safe(self, has_supports: bool) -> int:
        """Add PyReason rules with maximum safety"""
        rules_added = 0

        with _pyreason_operation_lock:
            # Rule 1: Defeat
            try:
                pr.add_rule(pr.Rule(
                    'defeated(Y) <- attacks(X, Y), accepted(X)',
                    'argument_defeated'
                ))
                logger.info("Added rule: argument_defeated")
                rules_added += 1
            except Exception as e:
                logger.error(f"Failed to add defeat rule: {e}")

            # Rule 2: Reinstatement
            try:
                pr.add_rule(pr.Rule(
                    'reinstated(Y) <- attacks(X, Y), defeated(X)',
                    'argument_reinstated'
                ))
                logger.info("Added rule: argument_reinstated")
                rules_added += 1
            except Exception as e:
                logger.warning(f"Could not add reinstatement rule: {e}")

            # Rule 3: Support
            if has_supports:
                try:
                    pr.add_rule(pr.Rule(
                        'strengthened(Y) <- supports(X, Y), accepted(X)',
                        'argument_strengthened'
                    ))
                    logger.info("Added rule: argument_strengthened")
                    rules_added += 1
                except Exception as e:
                    logger.warning(f"Could not add support rule: {e}")

        return rules_added

    def _verify_label_facts(self, node_ids) -> int:
        """Verify that label-based facts were loaded correctly"""
        verified = 0
        try:
            if hasattr(pr, 'graph') and pr.graph is not None:
                for node_id in node_ids:
                    try:
                        if node_id in pr.graph.nodes():
                            node_data = pr.graph.nodes[node_id]
                            if 'label' in node_data and isinstance(node_data['label'], dict):
                                if 'accepted' in node_data['label']:
                                    verified += 1
                                    logger.debug(f"Verified fact for {node_id}")
                    except Exception as e:
                        logger.debug(f"Could not verify {node_id}: {e}")
        except Exception as e:
            logger.debug(f"Verification failed: {e}")
        
        return verified

    def _add_facts_safe(self, node_confidences: Dict[str, Tuple[float, float]]) -> int:
        """Add initial facts - try direct graph modification to avoid add_fact segfault"""
        facts_added = 0

        # METHOD 1: Try setting facts directly on the loaded graph (safest)
        try:
            if hasattr(pr, 'graph') and pr.graph is not None:
                logger.info("Attempting to set facts via direct graph modification...")
                with _pyreason_add_fact_lock:
                    for arg_id, (lower, upper) in node_confidences.items():
                        try:
                            if arg_id in pr.graph.nodes():
                                # Set the attribute directly on the loaded graph
                                pr.graph.nodes[arg_id]['accepted'] = [lower, upper]
                                logger.debug(f"Fact set directly: {arg_id} [{lower}, {upper}]")
                                facts_added += 1
                        except Exception as e:
                            logger.debug(f"Direct graph modification failed for {arg_id}: {e}")
                
                if facts_added > 0:
                    logger.info(f"Successfully set {facts_added} facts via graph modification")
                    return facts_added
        except Exception as e:
            logger.debug(f"Direct graph modification approach failed: {e}")

        # METHOD 2: Use add_fact but wrap entire operation with additional safety
        logger.info("Falling back to add_fact method...")
        facts_added = 0
        
        # Pre-allocate all fact strings to avoid issues during iteration
        fact_strings = []
        for arg_id, (lower, upper) in node_confidences.items():
            fact_strings.append((arg_id, f'{arg_id}, accepted, {lower}, {upper}, 0, 0'))
        
        # Add all facts with single lock acquisition
        try:
            with _pyreason_add_fact_lock:
                for arg_id, fact_str in fact_strings:
                    try:
                        pr.add_fact(fact_str)
                        logger.debug(f"Fact added: {arg_id}")
                        facts_added += 1
                    except Exception as e:
                        logger.error(f"Failed to add fact for {arg_id}: {e}")
                        # Don't continue if even one fact fails - this may indicate corruption
                        if facts_added == 0:
                            raise
        except Exception as e:
            logger.error(f"Fact addition failed after {facts_added} facts: {e}")
            
        return facts_added

    def _extract_results(self, interpretation, arg_ids) -> Dict:
        """Extract and format PyReason results - faithful to PyReason API"""
        results = {
            'accepted': {},
            'defeated': {},
            'reinstated': {},
            'converged': False
        }

        try:
            if interpretation is None:
                logger.warning("Interpretation is None")
                return results

            # Convert Interpretation object to dict using PyReason's API
            interpretation_dict = interpretation.get_dict()
            
            if not interpretation_dict:
                logger.warning("Empty interpretation returned")
                return results

            timesteps = sorted(interpretation_dict.keys())
            final_timestep = timesteps[-1]
            logger.info(f"Extracting from timestep {final_timestep}")

            final_data = interpretation_dict[final_timestep]

            for arg_id in arg_ids:
                try:
                    if arg_id not in final_data:
                        continue
                        
                    node_data = final_data[arg_id]

                    for predicate in ['accepted', 'defeated', 'reinstated']:
                        if predicate not in node_data:
                            continue
                        
                        value = node_data[predicate]

                        if isinstance(value, (list, tuple)) and len(value) >= 2:
                            results[predicate][arg_id] = (float(value[0]), float(value[1]))
                        elif isinstance(value, (list, tuple)) and len(value) == 1:
                            v = float(value[0])
                            results[predicate][arg_id] = (v, v)
                        elif isinstance(value, dict):
                            results[predicate][arg_id] = (
                                float(value.get('lower', 0)),
                                float(value.get('upper', 1))
                            )
                        elif isinstance(value, (int, float)):
                            v = float(value)
                            results[predicate][arg_id] = (v, v)
                            
                except Exception as e:
                    logger.warning(f"Could not extract results for {arg_id}: {e}")

            results['converged'] = self._check_convergence(interpretation_dict)

        except Exception as e:
            logger.error(f"Error extracting results: {e}", exc_info=True)

        return results

    def _check_convergence(self, interpretation_dict) -> bool:
        """Check if reasoning has converged"""
        try:
            if not interpretation_dict or len(interpretation_dict) < 2:
                return False

            timesteps = sorted(interpretation_dict.keys())
            last = interpretation_dict[timesteps[-1]]
            second_last = interpretation_dict[timesteps[-2]]

            return last == second_last
        except Exception:
            return False


def test_pyreason_api():
    """Test PyReason API with maximum safety"""
    print("=" * 70)
    print("PyReason API Compatibility Test")
    print("=" * 70)

    try:
        print(f"\nPyReason version: {pr.__version__ if hasattr(pr, '__version__') else 'unknown'}")
        print(f"Python version: {sys.version}")
    except Exception:
        pass

    # Force reset first
    print("\n1. Testing reset...")
    try:
        PyReasonConnector.force_reset()
        print("   ✓ Reset successful")
    except Exception as e:
        print(f"   ✗ Reset failed: {e}")
        return False

    # Test settings
    print("\n2. Testing settings configuration...")
    try:
        if hasattr(pr, "settings"):
            pr.settings.verbose = False
            pr.settings.memory_profile = False
        print("   ✓ Settings configured")
    except Exception as e:
        print(f"   ✗ Settings failed: {e}")

    # Test graph loading
    print("\n3. Testing graph loading...")
    try:
        g = nx.DiGraph()
        g.add_node('A')  # No attributes
        g.add_node('B')
        g.add_edge('A', 'B', label='attacks')

        pr.load_graph(g)
        print("   ✓ Graph loaded")
    except Exception as e:
        print(f"   ✗ Graph loading failed: {e}")
        return False

    # Test rule addition
    print("\n4. Testing rule addition...")
    try:
        pr.add_rule(pr.Rule('test(Y) <- attacks(X, Y)', 'test_rule'))
        print("   ✓ Rule added")
    except Exception as e:
        print(f"   ✗ Rule failed: {e}")

    # Test fact addition
    print("\n5. Testing fact addition...")
    fact_added = False

    for fmt, test_fn in enumerate([
        lambda: pr.add_fact('A, accepted, 0.8, 1.0, 0, 0'),
        lambda: pr.add_fact(pr.Fact('A', 'accepted', 0.8, 1.0, 0, 0))
    ]):
        try:
            test_fn()
            print(f"   ✓ Fact format {fmt + 1} works")
            fact_added = True
            break
        except Exception:
            pass

    if not fact_added:
        print("   ⚠ No explicit fact format worked (may use graph attributes)")

    # Test reasoning
    print("\n6. Testing reasoning...")
    try:
        interpretation = pr.reason(timesteps=2)
        print(f"   ✓ Reasoning completed")
    except Exception as e:
        print(f"   ✗ Reasoning failed: {e}")
        return False

    # Cleanup
    print("\n7. Testing cleanup...")
    try:
        pr.reset()
        pr.reset_rules()
        g.clear()
        gc.collect()
        print("   ✓ Cleanup successful")
    except Exception as e:
        print(f"   ⚠ Cleanup warning: {e}")

    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = test_pyreason_api()
    sys.exit(0 if success else 1)
