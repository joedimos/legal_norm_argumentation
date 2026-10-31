"""Private subprocess entry point. Never imported by the main reasoner."""

from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path

from .pyreason_connector import (
    BACKEND_VERSION,
    MAX_INPUT_BYTES,
    MAX_OUTPUT_BYTES,
    PREDICATES,
    RULES,
    ResourceBound,
    digest,
    interval,
)
from .pyreason_io import read_json


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
            pr.add_fact(pr.Fact(f"{label}({source},{target}):[1,1]", name=f"{label}_{i}", start_time=0, end_time=end))
    for name, text in RULES:
        pr.add_rule(pr.Rule(text, name=name))
    interpretation = pr.reason(timesteps=end)
    data = interpretation.get_dict()
    if not data or any(type(t) is not int or not 0 <= t <= end for t in data):
        raise ValueError("invalid temporal output")
    final = data[max(data)]
    # get_dict contains trace changes; absent predicates retain open-world [0,1].
    return {
        predicate: {
            node: interval(dict(final.get(node, {}).items()).get(predicate, [0, 1])) for node in payload["nodes"]
        }
        for predicate in PREDICATES
    }


def process_request(source, target):
    result = {"status": "unknown"}
    try:
        backend_version = version("pyreason")
        if backend_version != BACKEND_VERSION:
            result["error"] = "unsupported_version"
        else:
            import pyreason as pr

            payload = read_json(source, MAX_INPUT_BYTES, 'input_bound')
            if payload.get("mode") == "runtime_probe":
                from .pyreason_runtime import native_identity
                annotations = native_identity(pr)
            elif payload.get("mode") == "normative_activation_v1":
                from .pyreason_program import execute_program

                annotations = execute_program(payload, pr)
            elif payload.get("mode") == "graph_temporal_v1":
                from .pyreason_graph import execute_native

                annotations = execute_native(payload, pr)
            else:
                annotations = execute(payload, pr)
            result = {
                "status": "observed",
                "backend_version": backend_version,
                "input_sha256": digest(payload),
                "annotations": annotations,
            }
    except ResourceBound as exc:
        result = {'status':'unknown', 'error':str(exc)}
    except Exception:
        result = {"status": "unknown"}  # No dependency diagnostics/source text.
    encoded = json.dumps(result, sort_keys=True, allow_nan=False)
    if len(encoded.encode('utf-8')) > MAX_OUTPUT_BYTES:
        encoded = json.dumps({'status':'unknown','error':'output_bound'})
    temporary = target.with_suffix(".pending")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(target)


def main():
    if sys.argv[1] != "--serve":
        process_request(Path(sys.argv[1]), Path(sys.argv[2]))
        return
    import select

    from .pyreason_pool import IDLE_SECONDS, MAX_REQUESTS

    root = Path(sys.argv[2]).resolve()
    for _ in range(MAX_REQUESTS):
        if not select.select([sys.stdin], [], [], IDLE_SECONDS)[0]:
            return
        request_id = sys.stdin.readline(257).strip()
        if not request_id.isascii() or not request_id.isdigit() or len(request_id) > 3:
            return
        process_request(root / (request_id + ".input.json"), root / (request_id + ".output.json"))


if __name__ == "__main__":
    main()
