"""Preference-based defeat: b>a blocks attack a->b; incomparability preserves it."""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from .argumentation import OptimizedAAF


@dataclass(frozen=True)
class Preference:
    preferred: str
    disfavored: str
    rationale: str


def preference_framework(arguments, attacks, preferences=()):
    base = OptimizedAAF(arguments, attacks=set(attacks))
    order = nx.DiGraph()
    order.add_nodes_from(base.arguments)
    for preference in preferences:
        if preference.preferred not in base.arguments or preference.disfavored not in base.arguments:
            raise ValueError("preference references an unknown argument")
        if not preference.rationale.strip():
            raise ValueError("preferences require an explicit rationale")
        order.add_edge(preference.preferred, preference.disfavored)
    if not nx.is_directed_acyclic_graph(order):
        raise ValueError("strict preferences must be acyclic and irreflexive")
    closure = set(nx.transitive_closure_dag(order).edges())
    suppressed = {(a, b) for a, b in base.attacks if (b, a) in closure}
    return OptimizedAAF(base.arguments, attacks=base.attacks - suppressed), suppressed


def analyze_preferences(arguments, attacks, preferences=(), *, semantics="grounded"):
    if semantics not in {"grounded", "preferred", "stable"}:
        raise ValueError("unsupported argumentation semantics")
    framework, suppressed = preference_framework(arguments, attacks, preferences)
    if semantics != "grounded" and len(framework.arguments) > 16:
        raise ValueError("extension enumeration is limited to 16 arguments")
    if semantics == "grounded":
        extensions = [framework.get_grounded_extension()]
    elif semantics == "preferred":
        extensions = framework.get_preferred_extensions()
    else:
        extensions = framework.get_stable_extensions()
    # An empty family of stable extensions never licenses vacuous acceptance.
    skeptical = set.intersection(*extensions) if extensions else set()
    credulous = set.union(*extensions) if extensions else set()
    return {"semantics": semantics, "extensions": sorted(sorted(e) for e in extensions),
            "skeptical": sorted(skeptical), "credulous": sorted(credulous),
            "defeats": sorted(map(list, framework.attacks)),
            "suppressed_attacks": sorted(map(list, suppressed)),
            "grounded_labels": framework.grounded_labeling(),
            "has_extensions": bool(extensions)}
