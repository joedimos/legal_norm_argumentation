"""Fixed finite program grammar for the active PyReason norm-activation backend."""

from __future__ import annotations

import re

from .pyreason_connector import ResourceBound, interval

MODE = "normative_activation_v1"
OUTCOMES = {
    "inactive_temporal": ("inactive", "outside_temporal_scope"),
    "unknown_date": ("undetermined", "evaluation_date_unknown"),
    "inactive_subject": ("inactive", "different_subject"),
    "inactive_condition": ("inactive", "condition_false"),
    "unknown_condition": ("undetermined", "condition_unknown"),
    "exception_applies": ("defeated", "exception_applies"),
    "unknown_exception": ("undetermined", "exception_unknown"),
    "active": ("active", "conditions_satisfied"),
}
LIMITS = {"max_norms": 64, "max_predicates_per_list": 32, "max_rules": 4096, "max_facts": 16384, "max_clauses": 65536}
LABEL = re.compile(r"(?:scope|actor|[ce][0-9]{1,2})_[tfu]\Z")
NODE = re.compile(r"n[0-9]{1,2}\Z")


def validate_program(program):
    if (
        not isinstance(program, dict)
        or set(program) != {"mode", "nodes", "rules", "timesteps"}
        or program["mode"] != MODE
        or type(program["timesteps"]) is not int
        or program["timesteps"] != 1
    ):
        raise ValueError("invalid activation program")
    nodes, rules = program["nodes"], program["rules"]
    if not isinstance(nodes, dict) or not nodes or not isinstance(rules, list):
        raise ValueError("invalid program nodes/rules")
    if len(nodes) > LIMITS["max_norms"] or len(rules) > LIMITS["max_rules"]:
        raise ResourceBound("program_size_bound")
    facts, clauses = 0, 0
    for node, values in nodes.items():
        if not isinstance(node, str) or not NODE.fullmatch(node) or not isinstance(values, dict):
            raise ValueError("invalid program node")
        facts += len(values)
        if any(
            not isinstance(label, str) or not LABEL.fullmatch(label) or type(value) is not bool
            for label, value in values.items()
        ):
            raise ValueError("invalid program fact")
    for rule in rules:
        if (
            not isinstance(rule, dict)
            or set(rule) != {"node", "head", "body"}
            or not isinstance(rule["node"], str)
            or rule["node"] not in nodes
            or not isinstance(rule["head"], str)
            or rule["head"] not in OUTCOMES
            or not isinstance(rule["body"], list)
        ):
            raise ValueError("invalid activation rule")
        clauses += len(rule["body"]) + 1  # Include the node-marker guard.
        for literal in rule["body"]:
            if (
                not isinstance(literal, list)
                or len(literal) != 2
                or not isinstance(literal[0], str)
                or literal[0] not in nodes[rule["node"]]
                or type(literal[1]) is not bool
            ):
                raise ValueError("invalid rule literal")
    if facts > LIMITS["max_facts"] or clauses > LIMITS["max_clauses"]:
        raise ResourceBound("program_size_bound")
    return {"facts": facts, "rules": len(rules), "clauses": clauses, "norms": len(nodes)}


def execute_program(program, pr):
    """No source strings or arbitrary rule text enter the engine grammar."""
    import networkx as nx

    validate_program(program)
    pr.reset()
    pr.reset_rules()
    pr.reset_settings()
    pr.settings.store_interpretation_changes = True
    pr.settings.persistent = True
    graph = nx.DiGraph()
    graph.add_nodes_from(program["nodes"])
    pr.load_graph(graph)
    for node, values in program["nodes"].items():
        for label, value in {**values, f"tag_{node}": True}.items():
            bit = int(value)
            pr.add_fact(pr.Fact(f"{label}({node}):[{bit},{bit}]", name=f"{node}_{label}", start_time=0, end_time=1))
    for index, rule in enumerate(program["rules"]):
        body = [f"tag_{rule['node']}(X):[1,1]"]
        body.extend(f"{label}(X):[{int(value)},{int(value)}]" for label, value in rule["body"])
        pr.add_rule(pr.Rule(f"{rule['head']}(X):[1,1] <- " + ", ".join(body), name=f"activation_{index}"))
    data = pr.reason(timesteps=1).get_dict()
    if not isinstance(data, dict) or not data or any(type(t) is not int or not 0 <= t <= 1 for t in data):
        raise ValueError("invalid activation trace")
    final = data[max(data)]
    return {
        outcome: {node: interval(dict(final.get(node, {}).items()).get(outcome, [0, 1])) for node in program["nodes"]}
        for outcome in OUTCOMES
    }
