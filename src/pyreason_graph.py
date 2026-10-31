"""Bounded relational/temporal interval inference with an independent oracle.

This finite constant-head Horn fragment is deliberately smaller than PyReason's
entire language. Missing information is [0,1]; only exact Boolean answers can
become ordinary norm facts. No source text is interpolated into rule syntax.
"""

from __future__ import annotations

import itertools
import subprocess  # nosec B404
import time

from .pyreason_connector import (
    BACKEND_VERSION,
    MAX_INPUT_BYTES,
    MAX_OUTPUT_BYTES,
    ResourceBound,
    WorkerFailure,
    current_worker_mode,
    digest,
    interval,
    run_worker,
)

MODE = "graph_temporal_v1"
LIMITS = {
    "max_entities": 16,
    "max_predicates": 32,
    "max_facts": 256,
    "max_rules": 32,
    "max_queries": 128,
    "max_horizon": 8,
    "max_groundings": 8192,
    "max_updates": 10000,
}


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("invalid finite bound")
    return value


def normalize(raw):
    if (
        not isinstance(raw, dict)
        or set(raw) - {"schema", "entities", "predicates", "facts", "rules", "queries", "bindings", "horizon", "kind", "review_status"}
        or raw.get("schema") != "vle-pyreason-graph/v1"
    ):
        raise ValueError("invalid graph request")
    if raw.get("review_status", "unreviewed") != "unreviewed" or raw.get("kind", "synthetic") not in {"synthetic", "source_derived"}:
        raise ValueError("graph proposals cannot self-assert review")

    def items(key, limit):
        value = raw.get(key, [])
        if not isinstance(value, list):
            raise ValueError("graph lists must be arrays")
        if len(value) > limit:
            raise ResourceBound("graph_" + key + "_bound")
        return value

    entities = items("entities", LIMITS["max_entities"])
    if (
        not entities
        or any(not isinstance(x, str) or not x or len(x) > 128 or x.startswith("$") for x in entities)
        or len(set(entities)) != len(entities)
    ):
        raise ValueError("invalid graph entities")
    emap = {x: f"n{i}" for i, x in enumerate(sorted(entities))}
    predicates = raw.get("predicates")
    if isinstance(predicates, dict) and len(predicates) > LIMITS["max_predicates"]:
        raise ResourceBound("graph_predicate_bound")
    if not isinstance(predicates, dict) or not 1 <= len(predicates) <= LIMITS["max_predicates"]:
        raise ValueError("invalid graph predicates")
    if any(
        not isinstance(k, str) or not k or len(k) > 128 or type(v) is not int or v not in (1, 2)
        for k, v in predicates.items()
    ):
        raise ValueError("only unary/binary predicates are supported")
    pmap = {x: f"p{i}" for i, x in enumerate(sorted(predicates))}
    if type(raw.get("horizon")) is int and raw["horizon"] > LIMITS["max_horizon"]:
        raise ResourceBound("graph_horizon_bound")
    horizon = integer(raw.get("horizon"), 0, LIMITS["max_horizon"])

    def atom(value, variables=False):
        if (
            not isinstance(value, dict)
            or set(value) != {"predicate", "arguments"}
            or value["predicate"] not in predicates
        ):
            raise ValueError("invalid graph atom")
        args = value["arguments"]
        if not isinstance(args, list) or len(args) != predicates[value["predicate"]]:
            raise ValueError("invalid graph arity")
        if variables:
            if any(
                not isinstance(x, str)
                or not x.startswith("$")
                or not x[1:].isascii()
                or not x[1:].isalpha()
                or len(x) > 16
                for x in args
            ):
                raise ValueError("rule arguments must be explicit variables")
            return [pmap[value["predicate"]], args]
        if any(not isinstance(x, str) or x not in emap for x in args):
            raise ValueError("undeclared graph entity")
        return [pmap[value["predicate"]], [emap[x] for x in args]]

    facts = []
    for f in items("facts", LIMITS["max_facts"]):
        if not isinstance(f, dict) or set(f) != {"atom", "interval", "start", "end"}:
            raise ValueError("invalid graph fact")
        start = integer(f["start"], 0, horizon)
        end = integer(f["end"], start, horizon)
        facts.append({"atom": atom(f["atom"]), "interval": interval(f["interval"]), "start": start, "end": end})
    rules = []
    for r in items("rules", LIMITS["max_rules"]):
        if (
            not isinstance(r, dict)
            or set(r) != {"id", "head", "body", "interval", "delay"}
            or not isinstance(r["id"], str)
            or not r["id"]
            or len(r["id"]) > 128
        ):
            raise ValueError("invalid graph rule")
        if not isinstance(r["body"], list) or not 1 <= len(r["body"]) <= 8:
            raise ValueError("rule body requires 1-8 clauses")
        body = []
        for b in r["body"]:
            if not isinstance(b, dict) or set(b) != {"atom", "interval"}:
                raise ValueError("invalid graph clause")
            bound = interval(b["interval"])
            if bound == [0, 1]:
                raise ValueError("unobserved information cannot establish a premise")
            body.append({"atom": atom(b["atom"], True), "interval": bound})
        head = atom(r["head"], True)
        variables = sorted({v for b in body for v in b["atom"][1]})
        if len(variables) > 3 or not set(head[1]) <= set(variables):
            raise ValueError("unsafe/unbounded head variables")
        bound = interval(r["interval"])
        if bound in ([0, 0], [1, 1]) and any(b["interval"] not in ([0, 0], [1, 1]) for b in body):
            raise ValueError("soft premises cannot create Boolean certainty")
        rules.append(
            {"id": r["id"], "head": head, "body": body, "interval": bound, "delay": integer(r["delay"], 0, horizon)}
        )
    if len({r["id"] for r in rules}) != len(rules):
        raise ValueError("duplicate graph rule id")
    queries = []
    for q in items("queries", LIMITS["max_queries"]):
        if (
            not isinstance(q, dict)
            or set(q) != {"id", "atom", "at"}
            or not isinstance(q["id"], str)
            or not q["id"]
            or len(q["id"]) > 128
        ):
            raise ValueError("invalid graph query")
        queries.append({"id": q["id"], "atom": atom(q["atom"]), "at": integer(q["at"], 0, horizon)})
    if not queries or len({q["id"] for q in queries}) != len(queries):
        raise ValueError("queries must be present and unique")
    bindings = raw.get("bindings", {})
    if (
        not isinstance(bindings, dict)
        or len(bindings) > 128
        or any(
            not isinstance(k, str) or not k or len(k) > 512 or v not in {q["id"] for q in queries}
            for k, v in bindings.items()
        )
    ):
        raise ValueError("invalid fact bindings")
    program = {
        "mode": MODE,
        "entities": sorted(emap.values()),
        "predicates": {pmap[k]: v for k, v in predicates.items()},
        "facts": facts,
        "rules": rules,
        "queries": queries,
        "horizon": horizon,
    }
    # Eager bounded grounding rejects excessive joins before native execution.
    ground(program)
    return program, {"entities": emap, "predicates": pmap}, bindings


def key(atom):
    return (atom[0], tuple(atom[1]))


def ground(program):
    result = []
    for rule in program["rules"]:
        variables = sorted({v for b in rule["body"] for v in b["atom"][1]})
        if len(result) + len(program["entities"]) ** len(variables) > LIMITS["max_groundings"]:
            raise ResourceBound("graph_grounding_bound")
        for values in itertools.product(program["entities"], repeat=len(variables)):
            assignment = dict(zip(variables, values, strict=True))

            def instantiate(a, assignment=assignment):
                return (a[0], tuple(assignment[v] for v in a[1]))

            result.append(
                (
                    instantiate(rule["head"]),
                    rule["interval"],
                    rule["delay"],
                    [(instantiate(b["atom"]), b["interval"]) for b in rule["body"]],
                )
            )
    return result


class IntervalConflict(ValueError):
    pass


def reference(program, *, deadline=None):
    """Independent finite least-information closure, not the native engine."""
    grounded = ground(program)
    scheduled = {}
    frames = []
    updates = 0

    def add(state, a, b):
        nonlocal updates
        old = state.get(a, [0, 1])
        new = [max(old[0], b[0]), min(old[1], b[1])]
        if new[0] > new[1]:
            raise IntervalConflict("inconsistent_interval_constraints")
        if new != old:
            state[a] = new
            updates += 1
            if updates > LIMITS["max_updates"]:
                raise ResourceBound("graph_update_bound")
            return True
        return False

    for tick in range(program["horizon"] + 1):
        state = scheduled.pop(tick, {})
        for fact in program["facts"]:
            if fact["start"] <= tick <= fact["end"]:
                add(state, key(fact["atom"]), fact["interval"])
        fired = set()
        changed = True
        while changed:
            changed = False
            for index, (head, bound, delay, body) in enumerate(grounded):
                if deadline is not None and time.monotonic() >= deadline:
                    raise ResourceBound("graph_reference_time_bound")
                if index in fired or tick + delay > program["horizon"]:
                    continue
                if all(
                    (value := state.get(a, [0, 1]))[0] >= required[0] and value[1] <= required[1]
                    for a, required in body
                ):
                    fired.add(index)
                    if delay:
                        add(scheduled.setdefault(tick + delay, {}), head, bound)
                    else:
                        changed = add(state, head, bound) or changed
        frames.append(state)
    return {q["id"]: frames[q["at"]].get(key(q["atom"]), [0, 1]) for q in program["queries"]}


def execute_native(program, pr):
    import networkx as nx

    pr.reset()
    pr.reset_rules()
    pr.reset_settings()
    pr.settings.store_interpretation_changes = True
    pr.settings.persistent = False
    graph = nx.DiGraph()
    graph.add_nodes_from(program["entities"])
    graph.add_edges_from(f["atom"][1] for f in program["facts"] if len(f["atom"][1]) == 2)
    pr.load_graph(graph)

    def text(a, variables=None):
        return a[0] + "(" + ",".join(variables[x] if variables is not None else x for x in a[1]) + ")"

    for i, f in enumerate(program["facts"]):
        pr.add_fact(
            pr.Fact(
                text(f["atom"]) + ":" + str(f["interval"]),
                name=f"graph_fact_{i}",
                start_time=f["start"],
                end_time=f["end"],
            )
        )
    for i, r in enumerate(program["rules"]):
        variables = {v: f"x{j}" for j, v in enumerate(sorted({v for b in r["body"] for v in b["atom"][1]}))}
        body = ", ".join(text(b["atom"], variables) + ":" + str(b["interval"]) for b in r["body"])
        rule = text(r["head"], variables) + ":" + str(r["interval"]) + f" <-{r['delay']} " + body
        pr.add_rule(pr.Rule(rule, name=f"graph_rule_{i}", infer_edges=len(r["head"][1]) == 2))
    if not program['rules']:
        # Upstream requires a rule for observation-only queries. Identity cannot
        # add information to false, soft or unobserved facts.
        predicate = min(program['predicates'])
        arity = program['predicates'][predicate]
        atom = predicate + '(' + ','.join(f'x{i}' for i in range(arity)) + ')'
        pr.add_rule(pr.Rule(atom + ':[1,1] <-0 ' + atom + ':[1,1]', name='observation_identity', infer_edges=arity == 2))
    data = pr.reason(timesteps=program["horizon"]).get_dict()
    if not isinstance(data, dict) or any(type(t) is not int or not 0 <= t <= program["horizon"] for t in data):
        raise ValueError("invalid graph trace")
    answers = {}
    for q in program["queries"]:
        component = q["atom"][1][0] if len(q["atom"][1]) == 1 else tuple(q["atom"][1])
        values = dict(data.get(q["at"], {}).get(component, {}).items())
        answers[q["id"]] = interval(values.get(q["atom"][0], [0, 1]))
    return answers


def infer_graph(raw, *, timeout_seconds=60):
    report = {
        "schema": "vle-pyreason-inference/v1",
        "status": "unknown",
        "review_status": "unreviewed",
        "decision_authorized": False,
        "legal_certification": False,
        "bounds": {**LIMITS, "timeout_seconds": timeout_seconds if type(timeout_seconds) is int and 1 <= timeout_seconds <= 300 else None,
                   "max_input_bytes": MAX_INPUT_BYTES, "max_output_bytes": MAX_OUTPUT_BYTES},
        "answers": {},
        "derived_facts": {},
    }
    started = time.monotonic()
    try:
        integer(timeout_seconds, 1, 300)
        program, maps, bindings = normalize(raw)
        expected = reference(program, deadline=started + timeout_seconds)
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise ResourceBound("graph_total_time_bound")
        report.update(
            program=program, program_sha256=digest(program), source_program_sha256=digest(raw), symbol_maps=maps
        )
        output = run_worker(program, remaining)
        answers = output["annotations"]
        if not isinstance(answers, dict) or set(answers) != set(expected):
            raise ValueError("incomplete graph answers")
        answers = {k: interval(v) for k, v in answers.items()}
        if answers != expected:
            report.update(
                reason="graph_backend_disagreement",
                disagreements=[
                    {"query": k, "native": answers[k], "reference": v} for k, v in expected.items() if answers[k] != v
                ],
            )
            return report
        facts = {
            name: (True if answers[q] == [1, 1] else False)
            for name, q in bindings.items()
            if answers[q] in ([0, 0], [1, 1])
        }
        if time.monotonic() - started >= timeout_seconds:
            raise ResourceBound("graph_total_time_bound")
        report.update(
            status="computed",
            backend_version=BACKEND_VERSION,
            worker_mode=current_worker_mode(),
            answers=answers,
            derived_facts=facts,
            uncertain_bindings=sorted(name for name in bindings if name not in facts),
            independent_check="matched",
        )
    except (ResourceBound, IntervalConflict, WorkerFailure) as exc:
        report.update(reason=str(exc), bound_hit=isinstance(exc, ResourceBound))
    except subprocess.TimeoutExpired:
        report.update(reason="graph_native_timeout", bound_hit=True)
    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        OSError,
        OverflowError,
        RecursionError,
        subprocess.SubprocessError,
    ):
        report["reason"] = "invalid_or_incomplete_graph_computation"
    return report
