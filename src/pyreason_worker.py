"""Private subprocess entry point. Never imported by the main reasoner."""
from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path

from .pyreason_connector import BACKEND_VERSION, PREDICATES, RULES, digest, interval


def execute(payload, pr):
    import networkx as nx

    if payload.get("rule_set_sha256") != digest(RULES):
        raise ValueError("incompatible rules")
    pr.reset()
    pr.reset_rules()
    pr.reset_settings()
    pr.settings.store_interpretation_changes = True
    pr.settings.persistent = True
    graph = nx.DiGraph()
    graph.add_nodes_from(payload["nodes"])
    graph.add_edges_from(payload["attacks"] + payload["supports"])
    pr.load_graph(graph)
    end = payload["timesteps"]
    for node, bound in payload["nodes"].items():
        pr.add_fact(pr.Fact(f"accepted({node}):{bound}", name=f"input_{node}", start_time=0, end_time=end))
    for label in ("attacks", "supports"):
        for i, (source, target) in enumerate(payload[label]):
            pr.add_fact(pr.Fact(f"{label}({source},{target}):[1,1]", name=f"{label}_{i}",
                                start_time=0, end_time=end))
    for name, text in RULES:
        pr.add_rule(pr.Rule(text, name=name))
    interpretation = pr.reason(timesteps=end)
    data = interpretation.get_dict()
    if not data or any(type(t) is not int or not 0 <= t <= end for t in data):
        raise ValueError("invalid temporal output")
    final = data[max(data)]
    # get_dict contains trace changes; absent predicates retain open-world [0,1].
    return {predicate: {node: interval(dict(final.get(node, {}).items()).get(predicate, [0, 1]))
                        for node in payload["nodes"]} for predicate in PREDICATES}


def main():
    target = Path(sys.argv[2])
    result = {"status": "unknown"}
    try:
        backend_version = version("pyreason")
        if backend_version != BACKEND_VERSION:
            result["error"] = "unsupported_version"
        else:
            import pyreason as pr
            payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
            if payload.get("mode") == "normative_activation_v1":
                from .pyreason_program import execute_program
                annotations = execute_program(payload, pr)
            else:
                annotations = execute(payload, pr)
            result = {"status": "observed", "backend_version": backend_version,
                      "input_sha256": digest(payload), "annotations": annotations}
    except Exception:
        result = {"status": "unknown"}  # No dependency diagnostics/source text.
    target.write_text(json.dumps(result, sort_keys=True, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
