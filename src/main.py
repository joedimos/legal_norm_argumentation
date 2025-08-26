from argumentation import build_argument_graph, LegalArgument
from src.argumentation import build_argument_graph
from src.pyreason_integration import PyReasonConnector

import logging
logging.basicConfig(level=logging.INFO)

def main():
    
    arg1 = LegalArgument(
        id="A1",
        premises={"Contract exists"},
        conclusion="Agreement is binding",
        supports=set(),
        attacks=set()
    )
    arg2 = LegalArgument(
        id="A2",
        premises={"Clause is unconscionable"},
        conclusion="Agreement is not binding",
        supports=set(),
        attacks={"A1"}
    )

    graph = build_argument_graph([arg1, arg2])

    G = build_argument_graph()  # however you construct it


    connector = PyReasonConnector(G)
    results = connector.run_reasoning()
    print(results)

    print("Nodes:", graph.nodes())
    print("Edges:", graph.edges())

if __name__ == "__main__":
    main()
