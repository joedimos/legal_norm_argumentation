"""Independent finite-model certificate checker. Standard library only.

This implementation is tested, not mechanically verified. The caller's trusted
query digest is the boundary between checking a model and checking the intended
model. A certificate never attests legal faithfulness or review authority.
"""

from __future__ import annotations

import hashlib
import json
from itertools import product
from time import monotonic

ARITY = {
    "true": 0,
    "false": 0,
    "atom": 0,
    "not": 1,
    "and": 2,
    "or": 2,
    "next": 1,
    "weak_next": 1,
    "eventually": 1,
    "always": 1,
    "until": 2,
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def formula_atoms(node):
    count = 0
    names = set()

    def walk(n, depth):
        nonlocal count
        count += 1
        if count > 512 or depth > 48 or not isinstance(n, dict) or set(n) - {"op", "args", "name"}:
            raise ValueError("formula bounds/fields")
        op = n.get("op")
        args = n.get("args", [])
        if op not in ARITY or not isinstance(args, list) or len(args) != ARITY[op]:
            raise ValueError("formula operator/arity")
        if op == "atom":
            name = n.get("name")
            if not isinstance(name, str) or not name.strip() or len(name) > 512:
                raise ValueError("invalid atom")
            names.add(name)
        elif n.get("name") is not None:
            raise ValueError("non-atom name")
        for a in args:
            walk(a, depth + 1)

    walk(node, 0)
    return names


def evaluate(node, trace, position=0, *, tick=lambda: None):
    formula_atoms(node)
    if (
        not isinstance(trace, (list, tuple))
        or not 1 <= len(trace) <= 128
        or any(
            not isinstance(s, dict)
            or any(not isinstance(k, str) or v is not None and type(v) is not bool for k, v in s.items())
            for s in trace
        )
    ):
        raise ValueError("invalid trace")
    if type(position) is not int or not 0 <= position < len(trace):
        raise ValueError("invalid position")

    def conjunction(a, b):
        return False if a is False or b is False else True if a is True and b is True else None

    def disjunction(a, b):
        return True if a is True or b is True else False if a is False and b is False else None

    memo = {}

    def visit(n, t):
        key = (id(n), t)
        if key in memo:
            return memo[key]
        tick()
        op = n["op"]
        args = n.get("args", [])
        if op == "atom":
            value = trace[t].get(n["name"])
        elif op in {"true", "false"}:
            value = op == "true"
        elif op == "not":
            inner = visit(args[0], t)
            value = None if inner is None else not inner
        elif op in {"and", "or"}:
            value = (conjunction if op == "and" else disjunction)(visit(args[0], t), visit(args[1], t))
        elif op in {"next", "weak_next"}:
            value = op == "weak_next" if t + 1 == len(trace) else visit(args[0], t + 1)
        elif op in {"always", "eventually"}:
            value = op == "always"
            for k in range(t, len(trace)):
                value = (conjunction if op == "always" else disjunction)(value, visit(args[0], k))
        else:
            # Witness-based until, independent of the producer's recurrence.
            value = False
            for k in range(t, len(trace)):
                prefix = True
                for j in range(t, k):
                    prefix = conjunction(prefix, visit(args[0], j))
                value = disjunction(value, conjunction(prefix, visit(args[1], k)))
        memo[key] = value
        return value

    return visit(node, position)


def merkle(leaves):
    if not leaves:
        return hashlib.sha256(b"E").hexdigest()
    nodes = [hashlib.sha256(b"L" + canonical(x)).digest() for x in leaves]
    while len(nodes) > 1:
        if len(nodes) % 2:
            nodes.append(nodes[-1])
        nodes = [hashlib.sha256(b"N" + nodes[i] + nodes[i + 1]).digest() for i in range(0, len(nodes), 2)]
    return nodes[0].hex()


def validate_query(q):
    if not isinstance(q, dict) or set(q) != {"property", "horizon", "assumptions", "observations", "binding"}:
        raise ValueError("query fields")
    if type(q["horizon"]) is not int or not 1 <= q["horizon"] <= 128:
        raise ValueError("horizon")
    if not isinstance(q["binding"], dict) or not isinstance(q["assumptions"], list) or len(q["assumptions"]) > 64:
        raise ValueError("query assumptions/binding")
    names = formula_atoms(q["property"])
    for a in q["assumptions"]:
        names |= formula_atoms(a)
    observations = q["observations"]
    if not isinstance(observations, list) or len(observations) > q["horizon"]:
        raise ValueError("observations")
    for s in observations:
        if not isinstance(s, dict) or any(
            not isinstance(k, str) or not k.strip() or len(k) > 512 or v is not None and type(v) is not bool
            for k, v in s.items()
        ):
            raise ValueError("observation values")
        names.update(s)
    if len(names) > 64:
        raise ValueError("atom limit")
    return sorted(names)


def check_certificate(certificate, *, expected_query_sha256=None, max_states=65536, timeout_ms=5000):
    result = {
        "valid": False,
        "decision_authorized": False,
        "legal_certification": False,
        "trusted_query_matched": False,
    }
    try:
        if (
            type(max_states) is not int
            or not 1 <= max_states <= 1000000
            or type(timeout_ms) is not int
            or not 1 <= timeout_ms <= 60000
        ):
            raise ValueError("checker limits")
        c = certificate
        if (
            not isinstance(c, dict)
            or set(c)
            != {
                "schema",
                "type",
                "query",
                "query_sha256",
                "evidence",
                "decision_authorized",
                "legal_certification",
                "review_status",
            }
            or c["schema"] != "vle-certificate/v1"
            or c["decision_authorized"] is not False
            or c["legal_certification"] is not False
            or c["review_status"] != "unreviewed"
        ):
            raise ValueError("certificate fields/authorization")
        q = c["query"]
        names = validate_query(q)
        if c["query_sha256"] != sha(q):
            raise ValueError("query digest mismatch")
        if expected_query_sha256 is not None and c["query_sha256"] != expected_query_sha256:
            raise ValueError("trusted query mismatch")
        result["trusted_query_matched"] = expected_query_sha256 is not None
        started = monotonic()
        work = 0

        def tick():
            nonlocal work
            work += 1
            if work > 2000000 or (monotonic() - started) * 1000 >= timeout_ms:
                raise ValueError("checker resource limit")

        fixed = {(t, k): v for t, s in enumerate(q["observations"]) for k, v in s.items() if v is not None}

        def valid_trace(trace):
            if (
                not isinstance(trace, list)
                or len(trace) != q["horizon"]
                or any(
                    not isinstance(s, dict) or set(s) != set(names) or any(type(v) is not bool for v in s.values())
                    for s in trace
                )
            ):
                raise ValueError("witness must be complete at exact horizon")
            if any(trace[t][k] != v for (t, k), v in fixed.items()):
                raise ValueError("observation mismatch")

        def admissible(trace, assumptions):
            return all(evaluate(a, trace, tick=tick) is True for a in assumptions)

        evidence = c["evidence"]
        kind = c["type"]
        if kind == "unknown":
            if (
                not isinstance(evidence, dict)
                or set(evidence) != {"bound", "limit"}
                or evidence["bound"]
                not in {"state_limit", "timeout", "work_limit", "encoding_limit", "unsupported_fragment"}
                or type(evidence["limit"]) is not int
                or evidence["limit"] < 0
            ):
                raise ValueError("unknown carries a bound diagnostic only")
        elif kind == "violation":
            if set(evidence) != {"trace"}:
                raise ValueError("violation fields")
            trace = evidence["trace"]
            valid_trace(trace)
            if not admissible(trace, q["assumptions"]) or evaluate(q["property"], trace, tick=tick) is not False:
                raise ValueError("not a violation of the bound query")
        elif kind in {"compliant_within_bound", "conflict"}:
            required = {"state_count", "admissible_count", "merkle_root"}
            if kind == "conflict":
                required |= {"minimal_ids", "deletion_witnesses", "cycle"}
            if not isinstance(evidence, dict) or set(evidence) != required:
                raise ValueError("enumeration evidence fields")
            free = [(t, k) for t in range(q["horizon"]) for k in names if (t, k) not in fixed]
            if len(free) > max_states.bit_length() or 2 ** len(free) > max_states:
                raise ValueError("checker state limit")
            if (2 ** len(free)) * (len(names) * q["horizon"] + 1) > 262144:
                raise ValueError("checker certificate cell limit 262144")
            leaves = []
            feasible = 0
            all_true = True
            for index, values in enumerate(product((False, True), repeat=len(free))):
                tick()
                assignment = dict(fixed)
                assignment.update(zip(free, values, strict=True))
                trace = [{k: assignment[t, k] for k in names} for t in range(q["horizon"])]
                allowed = admissible(trace, q["assumptions"])
                value = evaluate(q["property"], trace, tick=tick)
                leaves.append({"index": index, "trace": trace, "admissible": allowed, "value": value})
                feasible += allowed
                all_true &= not allowed or value is True
            if (
                type(evidence["state_count"]) is not int
                or type(evidence["admissible_count"]) is not int
                or evidence["state_count"] != len(leaves)
                or evidence["admissible_count"] != feasible
                or evidence["merkle_root"] != merkle(leaves)
            ):
                raise ValueError("enumeration count/root mismatch")
            if kind == "compliant_within_bound" and (not feasible or not all_true):
                raise ValueError("not nonvacuous compliance")
            if kind == "conflict":
                ids = q["binding"].get("cell_ids")
                if (
                    feasible
                    or not isinstance(ids, list)
                    or len(ids) != len(q["assumptions"])
                    or len(ids) != len(set(ids))
                    or evidence["minimal_ids"] != ids
                    or len(evidence["deletion_witnesses"]) != len(ids)
                ):
                    raise ValueError("conflict family/refutation")
                for i, w in enumerate(evidence["deletion_witnesses"]):
                    if set(w) != {"removed", "trace"} or w["removed"] != ids[i]:
                        raise ValueError("deletion id")
                    valid_trace(w["trace"])
                    if not admissible(w["trace"], q["assumptions"][:i] + q["assumptions"][i + 1 :]):
                        raise ValueError("family not demonstrated minimal")
                cycle = evidence["cycle"]
                edges = {tuple(sorted(e)) for e in q["binding"].get("overlap_edges", [])}
                if cycle and (
                    len(cycle) < 4
                    or cycle[0] != cycle[-1]
                    or len(set(cycle[:-1])) != len(cycle) - 1
                    or not set(cycle) <= set(ids)
                    or any(tuple(sorted((a, b))) not in edges for a, b in zip(cycle, cycle[1:], strict=False))
                ):
                    raise ValueError("invalid declared overlap cycle")
        else:
            raise ValueError("unsupported certificate type")
        tick()
        return {**result, "valid": True, "certificate_type": kind, "diagnostic_only": kind == "unknown"}
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError) as exc:
        return {**result, "reason": str(exc)}
