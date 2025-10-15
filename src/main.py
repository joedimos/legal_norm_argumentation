import sys
from pathlib import Path
import logging

if __name__ == "__main__":
    src_path = Path(__file__).parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import with error handling
try:
    from argumentation import OptimizedAAF, LegalArgument
    logger.info("✓ Loaded argumentation module")
except ImportError as e:
    logger.error(f"Failed to import argumentation: {e}")
    sys.exit(1)

try:
    from pyreason_connector import PyReasonConnector
    logger.info("✓ Loaded pyreason_connector module")
except ImportError as e:
    logger.error(f"Failed to import pyreason_connector: {e}")
    sys.exit(1)

# Optional: Try to load originality module (may fail if dependencies missing)
try:
    from originality import LogicBasedLegalReasoner
    ORIGINALITY_AVAILABLE = True
    logger.info("✓ Loaded originality module")
except ImportError as e:
    ORIGINALITY_AVAILABLE = False
    logger.warning(f"Originality module not available: {e}")
    logger.warning("Demos 2 and 3 will be skipped")


def demo_basic_argumentation():
    """
    Demonstrates the basic argumentation framework using OptimizedAAF
    and PyReasonConnector for simple attack/support relationships.
    """
    print("\n" + "=" * 70)
    print("DEMO 1: BASIC ARGUMENTATION FRAMEWORK")
    print("=" * 70)
    
    try:
        # Create arguments
        arg1 = LegalArgument(
            id="A1",
            premise="Contract exists",
            conclusion="Agreement is binding",
        )
        arg2 = LegalArgument(
            id="A2",
            premise="Clause is unconscionable",
            conclusion="Agreement is not binding",
            attacking_args=frozenset({"A1"}),
        )
        
        # Build argument graph using the static method
        graph = OptimizedAAF.build_argument_graph([arg1, arg2])
        
        print("\n--- Argument Graph Structure ---")
        print(f"Nodes: {list(graph.nodes())}")
        print(f"Edges: {list(graph.edges(data=True))}")
        
        # Create full argumentation framework
        aaf = OptimizedAAF(arguments=[arg1, arg2])
        
        # Run the reasoning process
        print("\n--- Running PyReason via OptimizedAAF ---")
        print("(This may take a moment...)")
        
        results = aaf.run_pyreason(timesteps=5)
        
        print("\nPyReason Results:")
        if 'error' in results:
            print(f"  ⚠️  Error: {results['error']}")
            if 'details' in results:
                print(f"  Details: {results['details']}")
        else:
            print("\n  Accepted Arguments:")
            for arg_id, bounds in results.get('accepted', {}).items():
                print(f"    {arg_id}: {bounds}")
            
            print("\n  Defeated Arguments:")
            defeated = results.get('defeated', {})
            if defeated:
                for arg_id, bounds in defeated.items():
                    print(f"    {arg_id}: {bounds}")
            else:
                print("    (none)")
            
            if results.get('converged'):
                print("\n  ✓ Reasoning converged")
            else:
                print("\n  ✗ Reasoning did not converge")
        
        return True
        
    except Exception as e:
        logger.error(f"Demo 1 failed: {e}", exc_info=True)
        print(f"\n  ⚠️  Demo 1 encountered an error: {e}")
        return False


def demo_originality_analysis():
    """
    Demonstrates the originality analysis system using LogicBasedLegalReasoner
    for copyright and intellectual property cases.
    """
    if not ORIGINALITY_AVAILABLE:
        print("\n" + "=" * 70)
        print("DEMO 2: ORIGINALITY & COPYRIGHT ANALYSIS")
        print("=" * 70)
        print("\n⚠️  Skipped - originality module not available")
        return False
    
    print("\n" + "=" * 70)
    print("DEMO 2: ORIGINALITY & COPYRIGHT ANALYSIS")
    print("=" * 70)
    
    try:
        # Create reasoner
        reasoner = LogicBasedLegalReasoner()
        
        # Sample legal case text about copyright originality
        case_text = """
        The software interface shows the author's personality through 
        free creative choices in layout and design. The developer made 
        original aesthetic decisions in color schemes and user flow.
        
        However, certain functional elements are dictated by technical 
        function and industry standards. These elements lack originality 
        as they are necessary for the software to operate correctly.
        """
        
        print("\n--- Analyzing Legal Case ---")
        print(f"Case Text Preview: {case_text[:150]}...")
        
        # Run analysis
        print("\n(Running analysis, this may take a moment...)")
        results = reasoner.analyze_legal_case(case_text, timesteps=10)
        
        # Display extracted arguments
        print("\n--- Extracted Arguments ---")
        if results['arguments']:
            for arg_id, arg_data in results['arguments'].items():
                print(f"\n  {arg_id}:")
                print(f"    Premise: {arg_data['premise']}")
                print(f"    Conclusion: {arg_data['conclusion']}")
                print(f"    Confidence: {arg_data['confidence']}")
        else:
            print("  (No arguments extracted)")
        
        # Display relationships
        print("\n--- Argument Relationships ---")
        print(f"  Attacks: {len(results['attacks'])} relationship(s)")
        for attacker, target in results['attacks'][:5]:  # Show first 5
            print(f"    {attacker} ⚔️  {target}")
        
        print(f"\n  Supports: {len(results['supports'])} relationship(s)")
        for supporter, supported in results['supports'][:5]:  # Show first 5
            print(f"    {supporter} 🤝 {supported}")
        
        # Display PyReason evaluation
        print("\n--- PyReason Evaluation ---")
        pyreason_eval = results.get('pyreason_eval', {})
        if pyreason_eval:
            print(f"  Originality Score: {pyreason_eval.get('originality_score', 0.0):.3f}")
            print(f"  Protection Score:  {pyreason_eval.get('protection_score', 0.0):.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Demo 2 failed: {e}", exc_info=True)
        print(f"\n  ⚠️  Demo 2 encountered an error: {e}")
        return False


def demo_simple_reasoning():
    """
    Simple demonstration without PyReason - just graph construction
    """
    print("\n" + "=" * 70)
    print("DEMO 3: SIMPLE ARGUMENT GRAPH (No PyReason)")
    print("=" * 70)
    
    try:
        # Create more complex argument structure
        args = [
            LegalArgument(
                id="evidence_exists",
                premise="Physical evidence was found at the scene",
                conclusion="Defendant's presence is established",
                confidence=(0.85, 0.95)
            ),
            LegalArgument(
                id="alibi",
                premise="Defendant has credible alibi witnesses",
                conclusion="Defendant was not at the scene",
                confidence=(0.70, 0.85),
                attacking_args=frozenset({"evidence_exists"})
            ),
            LegalArgument(
                id="witness_unreliable",
                premise="Alibi witnesses have credibility issues",
                conclusion="Alibi is not trustworthy",
                confidence=(0.60, 0.75),
                attacking_args=frozenset({"alibi"})
            ),
        ]
        
        print("\n--- Arguments ---")
        for arg in args:
            print(f"\n  {arg.id}:")
            print(f"    Premise: {arg.premise}")
            print(f"    Conclusion: {arg.conclusion}")
            print(f"    Confidence: {arg.confidence}")
            if arg.attacking_args:
                print(f"    Attacks: {', '.join(arg.attacking_args)}")
        
        # Build graph
        graph = OptimizedAAF.build_argument_graph(args)
        
        print("\n--- Graph Structure ---")
        print(f"  Nodes: {graph.number_of_nodes()}")
        print(f"  Edges: {graph.number_of_edges()}")
        
        print("\n  Attack Relationships:")
        for source, target, data in graph.edges(data=True):
            if data.get('type') == 'attacks':
                print(f"    {source} → {target}")
        
        print("\n✓ Graph constructed successfully (PyReason execution skipped)")
        return True
        
    except Exception as e:
        logger.error(f"Demo 3 failed: {e}", exc_info=True)
        print(f"\n  ⚠️  Demo 3 encountered an error: {e}")
        return False


def main():
    """
    Main entry point demonstrating all aspects of the legal argumentation system.
    """
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "LEGAL ARGUMENTATION SYSTEM" + " " * 27 + "║")
    print("║" + " " * 10 + "PyReason-based Legal Reasoning Framework" + " " * 17 + "║")
    print("╚" + "═" * 68 + "╝")
    
    success_count = 0
    total_demos = 3
    
    # Run demos with error isolation
    if demo_basic_argumentation():
        success_count += 1
    
    if demo_originality_analysis():
        success_count += 1
    
    if demo_simple_reasoning():
        success_count += 1
    
    # Summary
    print("\n" + "=" * 70)
    print(f"COMPLETED: {success_count}/{total_demos} demos ran successfully")
    print("=" * 70)
    
    if success_count == 0:
        print("\n⚠️  All demos failed. This may be due to PyReason compatibility issues.")
        print("Suggestions:")
        print("  1. Check PyReason version: pip show pyreason")
        print("  2. Try reinstalling: pip uninstall pyreason && pip install pyreason")
        print("  3. Check Python version compatibility (3.7-3.10)")
        sys.exit(1)
    elif success_count < total_demos:
        print("\n⚠️  Some demos failed. Check logs above for details.")
    else:
        print("\n✓ All demos completed successfully!")
    
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n⚠️  Fatal error: {e}")
        sys.exit(1)
