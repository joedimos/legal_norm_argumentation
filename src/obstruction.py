"""Bounded relational local/global consistency, with explicit modeling boundaries."""

from __future__ import annotations

from itertools import combinations, product

import networkx as nx

from .certificates import enumeration_evidence, make_certificate
from .checker import check_certificate
from .conflicts import detect_conflicts, scopes_overlap
from .model_checking import verify
from .norm_codec import decode_norm
from .phase4 import BoundHit, Budget, envelope, fields, integer
from .references import ReferenceResolver
from .temporal import Formula, evaluate_formula, validate_formula
from .temporal_deontic import TemporalNorm


def gyo_acyclic(cover):
    edges = [set(s) for s in cover]
    while edges:
        old = [set(s) for s in edges]
        for i in range(len(edges) - 1, -1, -1):
            if any(i != j and edges[i] <= edges[j] for j in range(len(edges))):
                edges.pop(i)
        counts = {a: sum(a in e for e in edges) for e in edges for a in e}
        edges = [{a for a in e if counts[a] > 1} for e in edges]
        edges = [e for e in edges if e]
        if edges == old:
            return False
    return True


def parse_cells(raw, horizon):
    if not isinstance(raw, list) or not 1 <= len(raw) <= 12:
        raise ValueError("requires1-12 norm cells")
    cells = []
    for c in raw:
        if (
            not isinstance(c, dict)
            or set(c) - {"id", "property", "norm", "window", "references"}
            or not isinstance(c.get("id"), str)
            or not c["id"]
        ):
            raise ValueError("invalid norm cell")
        if ("property" in c) == ("norm" in c):
            raise ValueError("one explicit property or typed norm required")
        norm = None
        if "norm" in c:
            norm = decode_norm(c["norm"])
            window = c.get("window", [0, horizon - 1])
            if not isinstance(window, list) or len(window) != 2 or window[1] >= horizon:
                raise ValueError("invalid norm window")
            temporal = TemporalNorm(norm, *window)
            f = temporal.constraint()
            names = validate_formula(f) | validate_formula(temporal.guard()) | {temporal.action_atom}
        else:
            f = Formula.from_dict(c["property"])
            names = validate_formula(f)
        refs = c.get("references", [])
        if not isinstance(refs, list) or len(refs) > 32 or any(not isinstance(r, str) for r in refs):
            raise ValueError("invalid references")
        cells.append({"id": c["id"], "formula": f, "atoms": names, "norm": norm, "references": refs})
    if len({c["id"] for c in cells}) != len(cells):
        raise ValueError("duplicate cell ids")
    return sorted(cells, key=lambda c: c["id"])


def join_tree_reduce(cells, horizon, budget):
    """Full relational semijoins, not pair-SAT. Return replayable global section."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(cells)))
    for i, j in combinations(range(len(cells)), 2):
        graph.add_edge(i, j, weight=len(cells[i]["atoms"] & cells[j]["atoms"]))
    tree = nx.maximum_spanning_tree(graph)
    for name in set().union(*(c["atoms"] for c in cells)):
        holders = [i for i, c in enumerate(cells) if name in c["atoms"]]
        if len(holders) > 1 and not nx.is_connected(tree.subgraph(holders)):
            raise BoundHit("join_tree_not_verified")
    relations = []
    for c in cells:
        names = sorted(c["atoms"])
        rows = []
        if len(names) * horizon > 20 or 2 ** (len(names) * horizon) > budget.max_states:
            raise BoundHit("local_relation_limit")
        for bits in product((False, True), repeat=len(names) * horizon):
            budget.tick()
            row = {(t, n): bits[t * len(names) + i] for t in range(horizon) for i, n in enumerate(names)}
            if evaluate_formula(c["formula"], [{n: row[t, n] for n in names} for t in range(horizon)]) is True:
                rows.append(row)
        relations.append(rows)
    parent = {0: None}
    order = [0]
    for i in order:
        for j in sorted(tree.neighbors(i)):
            if j not in parent:
                parent[j] = i
                order.append(j)
    for child in reversed(order[1:]):
        p = parent[child]
        shared = sorted({(t, n) for t in range(horizon) for n in cells[p]["atoms"] & cells[child]["atoms"]})
        signatures = {tuple(r[k] for k in shared) for r in relations[child]}
        relations[p] = [r for r in relations[p] if tuple(r[k] for k in shared) in signatures]
        budget.tick()
    if not relations[0]:
        return None, {
            "join_tree": sorted([cells[a]["id"], cells[b]["id"]] for a, b in tree.edges()),
            "empty_after_semijoins": True,
        }
    assigned = {}
    for i in order:
        budget.tick()
        row = next((r for r in relations[i] if all(k not in assigned or assigned[k] == v for k, v in r.items())), None)
        if row is None:
            raise BoundHit("join_witness_failed")
        assigned.update(row)
    names = sorted(set().union(*(c["atoms"] for c in cells)))
    return [{n: assigned[t, n] for n in names} for t in range(horizon)], {
        "join_tree": sorted([cells[a]["id"], cells[b]["id"]] for a, b in tree.edges()),
        "empty_after_semijoins": False,
    }


def obstruction(request, *, backend=None):
    if isinstance(request, dict) and request.get("schema") == "vle-obstruction-timeline/v1":
        return obstruction_persistence(request)
    fields(
        request,
        "vle-obstruction/v1",
        {
            "cells",
            "horizon",
            "assume_reviewed_demo",
            "backend",
            "max_states",
            "timeout_ms",
            "reference_index",
            "crosscheck_shortcut",
        },
        {"cells", "horizon"},
    )
    h = integer(request["horizon"], 1, 32, "horizon")
    cells = parse_cells(request["cells"], h)
    budget = Budget(max_states=request.get("max_states", 65536), timeout_ms=request.get("timeout_ms", 5000))
    backend = backend or request.get("backend", "exhaustive")
    if backend not in {"exhaustive", "z3"}:
        raise ValueError("backend must be exhaustive or z3")
    if type(request.get("assume_reviewed_demo", False)) is not bool:
        raise ValueError("demo flag must be Boolean")
    base = {
        "backend": backend,
        "horizon": h,
        "input_basis": "explicit_assumed_reviewed_demo_not_expert_validation",
        "acyclic_cover": gyo_acyclic([c["atoms"] for c in cells]),
        "minimal_family": [],
        "cycle": [],
        "witness": None,
        "nerve": {"vertices": [c["id"] for c in cells], "simplices": []},
        "pair_checks": [],
        "certificate": None,
    }

    def done(status, reason, **extra):
        return envelope(
            "vle-obstruction-report/v1",
            request,
            **{**base, **extra, "status": status, "reason": reason, "bounds": budget.to_dict()},
        )

    if not request.get("assume_reviewed_demo", False):
        return done("unknown", "review_evidence_required_or_explicit_demo_basis")
    cache = {}

    def sat(indices):
        indices = tuple(indices)
        if indices not in cache:
            budget.tick()
            result = verify(
                Formula("false"),
                horizon=h,
                assumptions=[cells[i]["formula"] for i in indices],
                backend=backend,
                max_traces=budget.max_states,
                timeout_ms=budget.timeout_ms,
            )
            budget.tick()
            if result.status not in {"counterexample", "inconsistent_assumptions"}:
                raise BoundHit("solver_" + result.status)
            cache[indices] = result.counterexample
        return cache[indices]

    try:
        graph = nx.Graph()
        graph.add_nodes_from(c["id"] for c in cells)
        resolver = ReferenceResolver(request.get("reference_index", {}))
        unresolved = []
        for size in range(1, len(cells) + 1):
            for indices in combinations(range(len(cells)), size):
                budget.tick()
                group = [cells[i] for i in indices]
                common = set.intersection(*(c["atoms"] for c in group))
                target = all(c["norm"] is not None for c in group) and all(
                    scopes_overlap(a["norm"], b["norm"]) for a, b in combinations(group, 2)
                )
                if common or target or size == 1:
                    base["nerve"]["simplices"].append([c["id"] for c in group])
                if size == 2 and (common or target):
                    graph.add_edge(group[0]["id"], group[1]["id"])
        for c in cells:
            for ref in c["references"]:
                resolved = resolver.resolve(ref)
                if resolved.status == "resolved" and resolved.source_id in graph:
                    graph.add_edge(c["id"], resolved.source_id)
                else:
                    unresolved.append({"cell": c["id"], "reference": ref})
        base["reference_findings"] = unresolved
        typed = [c["norm"] for c in cells if c["norm"] is not None]
        base["existing_pairwise_conflicts"] = [c.to_dict() for c in detect_conflicts(typed)]
        base["existing_pairwise_applicable"] = len(typed) == len(cells)
        for i, j in combinations(range(len(cells)), 2):
            witness = sat((i, j))
            base["pair_checks"].append({"ids": [cells[i]["id"], cells[j]["id"]], "consistent": witness is not None})
        shortcut_witness = None
        shortcut = {}
        if base["acyclic_cover"]:
            shortcut_witness, shortcut = join_tree_reduce(cells, h, budget)
        base["acyclic_shortcut"] = shortcut
        full = (
            sat(tuple(range(len(cells))))
            if request.get("crosscheck_shortcut", True) or not base["acyclic_cover"]
            else shortcut_witness
        )
        if base["acyclic_cover"] and (shortcut_witness is None) != (full is None):
            raise BoundHit("shortcut_disagrees_with_global_check")
        if full is not None:
            names = set().union(*(c["atoms"] for c in cells))
            full = [{n: s.get(n, False) for n in sorted(names)} for s in full]
            if not all(evaluate_formula(c["formula"], full) is True for c in cells):
                raise BoundHit("global_replay_failed")
            q = {
                "property": {"op": "false"},
                "horizon": h,
                "assumptions": [c["formula"].to_dict() for c in cells],
                "observations": [],
                "binding": {"cell_ids": [c["id"] for c in cells]},
            }
            cert = make_certificate(q, "violation", {"trace": full})
            used = set().union(*(validate_formula(c["formula"]) for c in cells))
            cert["evidence"]["trace"] = [{n: s[n] for n in sorted(used)} for s in full]
            if not check_certificate(cert)["valid"]:
                raise BoundHit("certificate_rejected")
            return done(
                "pairwise_conflict" if base["existing_pairwise_conflicts"] else "globally_consistent",
                "explicit_model_satisfiability_not_legal_compliance",
                witness=full,
                certificate=cert,
            )
        minimal = None
        for size in range(1, len(cells) + 1):
            for indices in combinations(range(len(cells)), size):
                if sat(indices) is None:
                    minimal = indices
                    break
            if minimal is not None:
                break
        ids = [cells[i]["id"] for i in minimal]
        cycles = nx.cycle_basis(graph.subgraph(ids))
        cycle = min(cycles, key=lambda c: (len(c), sorted(c))) if cycles else []
        if cycle:
            cycle = cycle + [cycle[0]]
        q = {
            "property": {"op": "true"},
            "horizon": h,
            "assumptions": [cells[i]["formula"].to_dict() for i in minimal],
            "observations": [],
            "binding": {"cell_ids": ids, "overlap_edges": sorted(map(list, graph.subgraph(ids).edges()))},
        }
        evidence, _ = enumeration_evidence(q, budget)
        names = set().union(*(cells[i]["atoms"] for i in minimal))
        evidence.update(minimal_ids=ids, cycle=cycle, deletion_witnesses=[])
        for removed in minimal:
            witness = sat(tuple(i for i in minimal if i != removed))
            # Empty sub-family is satisfiable; verifier returns [{}] for atom-free query.
            evidence["deletion_witnesses"].append(
                {
                    "removed": cells[removed]["id"],
                    "trace": [{n: s.get(n, False) for n in sorted(names)} for s in witness],
                }
            )
        cert = make_certificate(q, "conflict", evidence)
        # Certificate variable universe follows actual formulas, not unused metadata.
        if not check_certificate(cert, max_states=budget.max_states)["valid"]:
            raise BoundHit("certificate_rejected")
        status = (
            "pairwise_conflict" if len(minimal) <= 2 else "cyclic_obstruction" if cycle else "higher_order_conflict"
        )
        return done(status, "minimal_unsatisfiable_explicit_family", minimal_family=ids, cycle=cycle, certificate=cert)
    except BoundHit as exc:
        return done("unknown", str(exc), minimal_family=[], cycle=[], witness=None, certificate=None)


def obstruction_persistence(request):
    """Track explicitly requested families; unknown frames censor continuity."""
    from datetime import date

    fields(
        request,
        "vle-obstruction-timeline/v1",
        {"snapshots", "tracked_families", "max_states", "timeout_ms"},
        {"snapshots", "tracked_families"},
    )
    snapshots = request["snapshots"]
    families = request["tracked_families"]
    if (
        not isinstance(snapshots, list)
        or not 1 <= len(snapshots) <= 16
        or not isinstance(families, list)
        or len(families) > 16
    ):
        raise ValueError("timeline supports1-16 snapshots and0-16 explicitly tracked families")
    dates = []
    for s in snapshots:
        if (
            not isinstance(s, dict)
            or set(s) != {"version", "effective_date", "request"}
            or not isinstance(s["version"], str)
        ):
            raise ValueError("snapshot fields")
        dates.append(date.fromisoformat(s["effective_date"]))
    if dates != sorted(set(dates)):
        raise ValueError("snapshot dates must be strictly increasing")
    budget = Budget(max_states=request.get("max_states", 65536), timeout_ms=request.get("timeout_ms", 5000))
    rows = []
    intervals = []
    try:
        for family in families:
            if not isinstance(family, list) or not family or len(family) != len(set(family)):
                raise ValueError("tracked family ids")
            birth = None
            uncertain = False
            for index, s in enumerate(snapshots):
                budget.tick()
                raw = s["request"]
                cells = [c for c in raw["cells"] if c["id"] in family]
                if len(cells) != len(family):
                    state = "absent"
                else:
                    r = obstruction({**raw, "cells": cells})
                    state = (
                        "unknown"
                        if r["status"] == "unknown"
                        else "present"
                        if r["minimal_family"] == sorted(family)
                        else "absent"
                    )
                budget.tick()
                rows.append({"family": family, "version": s["version"], "state": state})
                if state == "unknown":
                    uncertain = True
                elif state == "present" and birth is None:
                    birth = index
                elif state == "absent" and birth is not None:
                    intervals.append(
                        {
                            "family": family,
                            "birth": snapshots[birth]["version"],
                            "death": s["version"],
                            "left_censored": birth == 0,
                            "right_censored": False,
                            "continuity_unknown": uncertain,
                        }
                    )
                    birth = None
                    uncertain = False
            if birth is not None:
                intervals.append(
                    {
                        "family": family,
                        "birth": snapshots[birth]["version"],
                        "death": None,
                        "left_censored": birth == 0,
                        "right_censored": True,
                        "continuity_unknown": uncertain,
                    }
                )
        return envelope(
            "vle-obstruction-timeline-report/v1",
            request,
            status="reported",
            frames=rows,
            intervals=intervals,
            bounds=budget.to_dict(),
            scope="tracked_family_lifetimes_not_persistent_homology_or_complete_discovery",
        )
    except BoundHit as exc:
        return envelope(
            "vle-obstruction-timeline-report/v1",
            request,
            status="unknown",
            reason=str(exc),
            frames=[],
            intervals=[],
            bounds=budget.to_dict(),
        )
