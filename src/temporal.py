"""Finite-trace propositional temporal logic; no natural-language parser or eval.

Complete valuations have classical LTLf semantics. Missing observations use a
sound, conservative strong-Kleene extension (not supervaluation).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache

ARITY = {"true": 0, "false": 0, "atom": 0, "not": 1, "and": 2, "or": 2,
         "next": 1, "weak_next": 1, "eventually": 1, "always": 1, "until": 2}
MAX_STEPS = 128
MAX_NODES = 512
MAX_DEPTH = 48


@dataclass(frozen=True)
class Formula:
    op: str
    args: tuple[Formula, ...] = ()
    name: str | None = None

    def __post_init__(self):
        if not isinstance(self.op, str) or self.op not in ARITY or len(self.args) != ARITY[self.op]:
            raise ValueError("unsupported operator or invalid arity")
        if not isinstance(self.args, tuple) or any(not isinstance(x, Formula) for x in self.args):
            raise ValueError("formula arguments must be a tuple of formulas")
        if self.op == "atom":
            if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 512:
                raise ValueError("atom name must contain 1-512 characters")
        elif self.name is not None:
            raise ValueError("only atoms have names")

    def to_dict(self) -> dict:
        return {"op": self.op, **({"name": self.name} if self.op == "atom" else {}),
                **({"args": [a.to_dict() for a in self.args]} if self.args else {})}

    @classmethod
    def from_dict(cls, data: Mapping) -> Formula:
        count = 0

        def parse(node, depth):
            nonlocal count
            count += 1
            if depth > MAX_DEPTH or count > MAX_NODES:
                raise ValueError("formula exceeds depth/node limit")
            if not isinstance(node, Mapping) or set(node) - {"op", "args", "name"}:
                raise ValueError("invalid formula fields")
            args = node.get("args", [])
            if not isinstance(args, list) or len(args) > 2:
                raise ValueError("formula args must be a list of at most two operands")
            return cls(node.get("op"), tuple(parse(a, depth + 1) for a in args), node.get("name"))

        return parse(data, 0)


def atom(name: str) -> Formula:
    return Formula("atom", name=name)


def combine(op: str, *args: Formula) -> Formula:
    return Formula(op, args)


def validate_formula(formula: Formula) -> set[str]:
    pending = [(formula, 0)]
    names = set()
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if not isinstance(node, Formula) or depth > MAX_DEPTH or count > MAX_NODES:
            raise ValueError("formula exceeds depth/node limit or contains an invalid node")
        if node.op == "atom":
            names.add(node.name)
        pending.extend((arg, depth + 1) for arg in node.args)
    return names


def validate_trace(trace: Sequence[Mapping[str, bool | None]]) -> None:
    if not 1 <= len(trace) <= MAX_STEPS:
        raise ValueError(f"trace must have 1-{MAX_STEPS} states")
    for state in trace:
        if not isinstance(state, Mapping) or any(
            not isinstance(key, str) or (value is not None and type(value) is not bool)
            for key, value in state.items()
        ):
            raise ValueError("trace states must map atom names to bool or null")


def _and(left, right):
    return False if left is False or right is False else None if left is None or right is None else True


def _or(left, right):
    return True if left is True or right is True else None if left is None or right is None else False


def evaluate_formula(formula: Formula, trace: Sequence[Mapping[str, bool | None]], position: int = 0) -> bool | None:
    validate_formula(formula)
    validate_trace(trace)
    if type(position) is not int or not 0 <= position < len(trace):
        raise ValueError("position must address an observed state")

    @cache
    def visit(node, index):
        op, args = node.op, node.args
        if op in {"true", "false"}:
            return op == "true"
        if op == "atom":
            return trace[index].get(node.name)
        if op == "not":
            value = visit(args[0], index)
            return None if value is None else not value
        if op in {"and", "or"}:
            return (_and if op == "and" else _or)(visit(args[0], index), visit(args[1], index))
        if op in {"next", "weak_next"}:
            return op == "weak_next" if index + 1 == len(trace) else visit(args[0], index + 1)
        if op in {"eventually", "always"}:
            result = op == "always"
            for step in range(index, len(trace)):
                result = (_and if op == "always" else _or)(result, visit(args[0], step))
            return result
        # Strong until: a right-hand witness is required before the finite end.
        result = False
        for step in reversed(range(index, len(trace))):
            result = _or(visit(args[1], step), _and(visit(args[0], step), result))
        return result

    return visit(formula, position)
