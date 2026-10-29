"""Synthetic, unreviewed PyReason-shaped provider for disposable wheel tests only.

This is a deterministic test double, not the third-party engine. Never installed
in an application environment or represented as real-engine qualification.
"""
import json
import re
from types import SimpleNamespace

settings = SimpleNamespace()
facts = []
rules = []


def reset():
    facts.clear()


def reset_rules():
    rules.clear()


def reset_settings():
    return None


def load_graph(graph):
    return None


class Fact:
    def __init__(self, text, **kwargs):
        head, bound = text.split(":")
        predicate, components = head.rstrip(")").split("(")
        self.predicate = predicate
        self.components = components.split(",")
        self.bound = json.loads(bound)


class Rule:
    def __init__(self, text, **kwargs):
        self.text = text


def add_fact(fact):
    facts.append(fact)


def add_rule(rule):
    rules.append(rule)


def reason(timesteps):
    if not any(f.predicate == "accepted" for f in facts):
        # The activation fragment has only known Boolean flags in rule bodies.
        nodes = {}
        for fact in facts:
            nodes.setdefault(fact.components[0], {})[fact.predicate] = fact.bound
        pattern = re.compile(r"([a-z0-9_]+)\(X\):\[([01]),[01]\]")
        for rule in rules:
            head, body = rule.text.split(" <- ")
            head = head.split("(X)")[0]
            literals = pattern.findall(body)
            for values in nodes.values():
                if all(values.get(label) == [int(bit), int(bit)] for label, bit in literals):
                    values[head] = [1, 1]
        return SimpleNamespace(get_dict=lambda: {timesteps: nodes})
    nodes = {f.components[0]: {"accepted": f.bound} for f in facts if f.predicate == "accepted"}
    attacks = [f.components for f in facts if f.predicate == "attacks"]
    for source, target in attacks:
        if nodes[source]["accepted"][0] >= 0.5:
            nodes[target]["defeated"] = [1, 1]
    for source, target in attacks:
        if nodes[source].get("defeated") == [1, 1]:
            nodes[target]["reinstated"] = [1, 1]
    for fact in facts:
        if fact.predicate == "supports" and nodes[fact.components[0]]["accepted"][0] >= 0.5:
            nodes[fact.components[1]]["strengthened"] = [1, 1]
    return SimpleNamespace(get_dict=lambda: {timesteps: nodes})
