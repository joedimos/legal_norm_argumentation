"""Bounded propositional ASPIC+ instantiation with explicit attack sites.

No contraposition or premises are invented. Cyclic derivation systems return
unknown rather than silently identifying an infinite family with a finite one.
"""

from __future__ import annotations

from itertools import product

import networkx as nx

from .argumentation import OptimizedAAF
from .phase4 import BoundHit, Budget, envelope, fields, integer


def contrary(literal):
    return literal[1:] if literal.startswith("~") else "~" + literal


def analyze_aspic(request, *, lifting=None):
    fields(
        request,
        "vle-aspic/v1",
        {
            "premises",
            "rules",
            "contraries",
            "preferences",
            "lifting",
            "semantics",
            "max_arguments",
            "max_depth",
            "max_states",
            "timeout_ms",
        },
        {"premises", "rules"},
    )
    lifting = lifting or request.get("lifting", "last")
    semantics = request.get("semantics", "grounded")
    if lifting not in {"last", "weakest"} or semantics not in {"grounded", "preferred", "stable"}:
        raise ValueError("invalid lifting or semantics")
    maximum = integer(request.get("max_arguments", 128), 1, 256, "max_arguments")
    depth_limit = integer(request.get("max_depth", 16), 1, 32, "max_depth")
    budget = Budget(max_states=request.get("max_states", 65536), timeout_ms=request.get("timeout_ms", 5000))
    premises, rules = request["premises"], request["rules"]
    if not isinstance(premises, list) or not isinstance(rules, list) or len(premises) + len(rules) > 128:
        raise ValueError("at most 128 premises and rules")
    ids = set()
    for item, keys in [(p, {"id", "literal", "kind"}) for p in premises] + [
        (r, {"id", "antecedents", "conclusion", "kind"}) for r in rules
    ]:
        if not isinstance(item, dict) or set(item) != keys:
            raise ValueError("invalid premise/rule fields")
        if any(
            not isinstance(item[k], str) or not item[k].strip() or len(item[k]) > 512 for k in keys - {"antecedents"}
        ):
            raise ValueError("invalid premise/rule identifiers")
        if item["id"].startswith("premise:"):
            raise ValueError("premise: is reserved for internal premise attack sites")
        if item["id"] in ids:
            raise ValueError("premise/rule ids must be globally unique")
        ids.add(item["id"])
    if any(p["kind"] not in {"axiom", "ordinary"} for p in premises):
        raise ValueError("invalid premise kind")
    for rule in rules:
        if (
            rule["kind"] not in {"strict", "defeasible"}
            or not isinstance(rule["antecedents"], list)
            or len(rule["antecedents"]) > 16
            or len(set(rule["antecedents"])) != len(rule["antecedents"])
            or any(not isinstance(a, str) or not a.strip() for a in rule["antecedents"])
        ):
            raise ValueError("invalid rule")
    raw_contraries = request.get("contraries", {})
    if (
        not isinstance(raw_contraries, dict)
        or len(raw_contraries) > 256
        or any(
            not isinstance(k, str)
            or not isinstance(v, list)
            or len(v) > 128
            or any(not isinstance(x, str) or not x for x in v)
            for k, v in raw_contraries.items()
        )
    ):
        raise ValueError("invalid contraries")

    def is_contrary(a, b):
        return a == contrary(b) or a in raw_contraries.get(b, [])

    order = nx.DiGraph()
    order.add_nodes_from(ids)
    preferences = request.get("preferences", [])
    if not isinstance(preferences, list) or len(preferences) > 256:
        raise ValueError("invalid preferences")
    for p in preferences:
        if (
            not isinstance(p, dict)
            or set(p) != {"preferred", "disfavored", "rationale"}
            or p["preferred"] not in ids
            or p["disfavored"] not in ids
            or not isinstance(p["rationale"], str)
            or not p["rationale"].strip()
        ):
            raise ValueError("invalid preference")
        order.add_edge(p["preferred"], p["disfavored"])
    if not nx.is_directed_acyclic_graph(order):
        raise ValueError("preference order must be acyclic")
    preferred = set(nx.transitive_closure_dag(order).edges())
    by_rule = {r["id"]: r for r in rules}
    args = {}
    lookup = {}

    def add(key, conclusion, top, children=(), ordinary=(), defeasible=(), last=(), depth=0):
        if key in lookup:
            return
        budget.tick()
        if len(args) >= maximum:
            raise BoundHit("argument_limit")
        if depth > depth_limit:
            raise BoundHit("depth_limit")
        id = f"a{len(args):04d}"
        subargs = {id} | {s for child in children for s in args[child]["subarguments"]}
        args[id] = {
            "id": id,
            "conclusion": conclusion,
            "top": top,
            "children": list(children),
            "subarguments": sorted(subargs),
            "ordinary": sorted(set(ordinary)),
            "defeasible": sorted(set(defeasible)),
            "last": sorted(set(last)),
            "depth": depth,
        }
        lookup[key] = id

    def weaker(a, b):
        # Elitist strict set lifting: some fallible element of a is worse than
        # every fallible element of b. Empty means strict/firm and cannot lose.
        aa = set(a["ordinary"]) | set(a["last" if lifting == "last" else "defeasible"])
        bb = set(b["ordinary"]) | set(b["last" if lifting == "last" else "defeasible"])
        return bool(aa) and (not bb or any(all((y, x) in preferred for y in bb) for x in aa))

    def result(status, reason, **extra):
        return envelope(
            "vle-aspic-result/v1",
            request,
            status=status,
            reason=reason,
            lifting=lifting,
            set_lifting="elitist",
            semantics=semantics,
            bounds={**budget.to_dict(), "max_arguments": maximum, "max_depth": depth_limit},
            **extra,
        )

    try:
        dependency = nx.DiGraph()
        for r in rules:
            for a in r["antecedents"]:
                dependency.add_edge(a, r["conclusion"])
        if not nx.is_directed_acyclic_graph(dependency):
            raise BoundHit("cyclic_derivations")
        for p in sorted(premises, key=lambda x: x["id"]):
            add(
                ("premise", p["id"]),
                p["literal"],
                "premise:" + p["id"],
                ordinary=[p["id"]] if p["kind"] == "ordinary" else [],
            )
        while True:
            before = len(args)
            for r in sorted(rules, key=lambda x: x["id"]):
                budget.tick()
                choices = [
                    [a["id"] for a in args.values() if a["conclusion"] == literal] for literal in r["antecedents"]
                ]
                for children in product(*choices):
                    budget.tick()
                    ds = {d for c in children for d in args[c]["defeasible"]}
                    last = {d for c in children for d in args[c]["last"]}
                    if r["kind"] == "defeasible":
                        ds.add(r["id"])
                        last = {r["id"]}
                    add(
                        (r["id"], children),
                        r["conclusion"],
                        r["id"],
                        children,
                        ordinary={p for c in children for p in args[c]["ordinary"]},
                        defeasible=ds,
                        last=last,
                        depth=1 + max((args[c]["depth"] for c in children), default=0),
                    )
            if len(args) == before:
                break
        attacks, defeats = [], set()
        for a, b in product(args.values(), repeat=2):
            for site in b["subarguments"]:
                budget.tick()
                target = args[site]
                kind = None
                if target["top"].startswith("premise:"):
                    if target["ordinary"] and is_contrary(a["conclusion"], target["conclusion"]):
                        kind = "undermining"
                elif by_rule[target["top"]]["kind"] == "defeasible":
                    if is_contrary(a["conclusion"], "rule:" + target["top"]):
                        kind = "undercutting"
                    elif is_contrary(a["conclusion"], target["conclusion"]):
                        kind = "rebutting"
                if kind:
                    successful = (
                        kind == "undercutting"
                        or not is_contrary(target["conclusion"], a["conclusion"])
                        or not weaker(a, target)
                    )
                    attacks.append(
                        {"from": a["id"], "to": b["id"], "site": site, "kind": kind, "successful": successful}
                    )
                    if successful:
                        defeats.add((a["id"], b["id"]))
        framework = OptimizedAAF(sorted(args), attacks=defeats)
        if semantics != "grounded" and (len(args) > 16 or 2 ** len(args) > budget.max_states - budget.used):
            raise BoundHit("extension_limit")
        budget.tick()
        extensions = (
            [framework.get_grounded_extension()]
            if semantics == "grounded"
            else framework.get_preferred_extensions()
            if semantics == "preferred"
            else framework.get_stable_extensions()
        )
        budget.tick()  # discard all positive results if the entire analysis timed out
        conclusions = [sorted({args[a]["conclusion"] for a in e}) for e in extensions]
        strict = [r for r in rules if r["kind"] == "strict"]
        transposed = all(
            any(
                set(s["antecedents"]) == (set(r["antecedents"]) - {a}) | {contrary(r["conclusion"])}
                and s["conclusion"] == contrary(a)
                for s in strict
            )
            for r in strict
            for a in r["antecedents"]
        )
        axioms = [p["literal"] for p in premises if p["kind"] == "axiom"]

        def strict_closure(values):
            closure = set(values)
            while True:
                old = set(closure)
                for rule in strict:
                    budget.tick()
                    if set(rule["antecedents"]) <= closure:
                        closure.add(rule["conclusion"])
                if old == closure:
                    return closure

        axiom_closure = strict_closure(axioms)
        preconditions = {
            "axioms_strictly_consistent": not any(is_contrary(a, b) for a in axiom_closure for b in axiom_closure),
            "acyclic_order": True,
            "strict_transposition": transposed,
            "axioms_directly_consistent": not any(is_contrary(a, b) for a in axioms for b in axioms),
        }
        postulates = {
            "subargument_closure": all(set(args[a]["subarguments"]) <= e for e in extensions for a in e),
            "direct_consistency": all(not any(is_contrary(a, b) for a in c for b in c) for c in conclusions),
            "strict_closure": all(
                all(not set(r["antecedents"]) <= set(c) or r["conclusion"] in c for r in strict) for c in conclusions
            ),
        }
        closed_conclusions = [strict_closure(c) for c in conclusions]
        postulates["indirect_consistency"] = all(
            not any(is_contrary(a, b) for a in c for b in c) for c in closed_conclusions
        )
        budget.tick()
        return result(
            "analyzed",
            "bounded_structured_argumentation",
            arguments=list(args.values()),
            attacks=attacks,
            defeats=sorted(map(list, defeats)),
            extensions=sorted(sorted(e) for e in extensions),
            grounded_labels=framework.grounded_labeling(),
            conclusions=conclusions,
            has_extensions=bool(extensions),
            rationality={
                "preconditions": preconditions,
                "observed_postulates": postulates,
                "universal_guarantee": False,
            },
        )
    except BoundHit as exc:
        return result(
            "unknown",
            str(exc),
            arguments=[],
            attacks=[],
            defeats=[],
            extensions=[],
            grounded_labels={},
            conclusions=[],
            has_extensions=False,
            rationality={"universal_guarantee": False},
        )


def analyze_norms(norms, context, *, lifting="last"):
    """Compatibility bridge for *active*, single-context legacy typed norms.

    Unknown activations yield unknown. Exceptions are evaluated by the existing
    activation function; this bridge does not synthesize exception arguments.
    """
    from .conflicts import detect_conflicts
    from .defeasible import activation
    from .priority import resolve_pair

    if len(norms) > 64 or len({n.id for n in norms}) != len(norms):
        raise ValueError("requires up to 64 uniquely identified norms")
    states = {n.id: activation(n, context).status for n in norms}
    active = {n.id: n for n in norms if states[n.id] == "active"}
    contraries = {id: [] for id in active}
    prefs = []
    for c in detect_conflicts(list(active.values())):
        contraries[c.left_id].append(c.right_id)
        contraries[c.right_id].append(c.left_id)
        p = resolve_pair(active[c.left_id], active[c.right_id])
        if p.status == "resolved":
            prefs.append({"preferred": p.winner_id, "disfavored": p.loser_id, "rationale": p.principle})
    result = analyze_aspic(
        {
            "schema": "vle-aspic/v1",
            "premises": [],
            "rules": [{"id": id, "antecedents": [], "conclusion": id, "kind": "defeasible"} for id in active],
            "contraries": contraries,
            "preferences": prefs,
            "lifting": lifting,
        }
    )
    if "undetermined" in states.values():
        result.update(
            status="unknown",
            reason="unknown_activation",
            extensions=[],
            conclusions=[],
            has_extensions=False,
            grounded_labels={},
            arguments=[],
            attacks=[],
            defeats=[],
            rationality={"universal_guarantee": False},
        )
    result["activations"] = states
    result["norm_labels"] = {
        a["top"]: result["grounded_labels"][a["id"]]
        for a in result["arguments"]
        if a["id"] in result["grounded_labels"]
    }
    return result
