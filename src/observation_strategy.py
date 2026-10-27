"""Shared finite minimax kernel for labelled worlds and noiseless Boolean queries."""
from __future__ import annotations

from math import inf


class PlanningLimit(Exception):
    states = 0


def optimal_strategy(*, world_count, true_mask, masks, queries, max_states, check_limit):
    """Return an exact decision DAG or infinite cost; callers own world semantics."""
    states = {}

    def solve(state):
        check_limit()
        if state in states:
            return states[state][0]
        if len(states) >= max_states:
            raise PlanningLimit("planning_state_limit")
        states[state] = inf, None
        positive = state & true_mask
        if not positive or positive == state:
            states[state] = 0, "true" if positive else "false"
            return 0
        best, choice = inf, None
        for index, ((_, _, price), mask) in enumerate(zip(queries, masks, strict=True)):
            low, high = state & ~mask, state & mask
            if not low or not high:
                continue
            left = solve(low)
            if left == inf:
                continue
            value = price + max(left, solve(high))
            if value < best:
                best, choice = value, index
        states[state] = best, choice
        return best

    nodes, ids = [], {}

    def emit(state):
        check_limit()
        if state in ids:
            return ids[state]
        node_id = f"n{len(ids)}"
        ids[state] = node_id
        cost, choice = states[state]
        node = {"id": node_id, "completion_count": state.bit_count(), "worst_case_cost": cost}
        nodes.append(node)
        if choice in {"true", "false"}:
            node["verdict"] = "definitely_true" if choice == "true" else "definitely_false"
        else:
            step, name, price = queries[choice]
            node.update({"step": step, "atom": name, "cost": price,
                         "on_false": emit(state & ~masks[choice]), "on_true": emit(state & masks[choice])})
        return node_id

    try:
        root = (1 << world_count) - 1
        cost = solve(root)
        plan = None if cost == inf else {"root": emit(root), "nodes": nodes}
        return {"cost": cost, "states": len(states), "plan": plan}
    except PlanningLimit as exc:
        exc.states = len(states)
        raise
