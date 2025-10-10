from argumentation import build_argument_graph, LegalArgument
from pyreason_integration import PyReasonConnector
import logging

logging.basicConfig(level=logging.INFO)


def main():
    # Create arguments
    arg1 = LegalArgument(
        id="A1",
        premise="Contract exists",
        conclusion="Agreement is binding",
        supporting_args=set(),
        attacking_args=set(),
    )
    arg2 = LegalArgument(
        id="A2",
        premise="Clause is unconscionable",
        conclusion="Agreement is not binding",
        supporting_args=set(),
        attacking_args={"A1"},
    )

    # Build argument graph
    graph = build_argument_graph([arg1, arg2])

    # Connect to PyReason
    connector = PyReasonConnector(
        arguments={arg.id: arg for arg in [arg1, arg2]},
        attacks={(arg.id, target) for arg in [arg1, arg2] for target in arg.attacking_args},
    )

    results = connector.run_reasoning()
    print("\nPyReason results:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    # Print graph structure
    print("\nArgument Graph:")
    print("Nodes:", graph.nodes())
    print("Edges:", graph.edges())


if __name__ == "__main__":
    main()
