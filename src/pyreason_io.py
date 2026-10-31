"""Strict, size-bounded private worker JSON. No source text in diagnostics."""

from __future__ import annotations

from .audit_bundle import canonical_json as validate_json
from .json_boundary import strict_json


def read_json(path, limit, bound):
    # Bound the read itself: a prior stat check does not bound subsequent I/O.
    with path.open("rb") as source:
        raw = source.read(limit + 1)
    if len(raw) > limit:
        from .pyreason_connector import ResourceBound

        raise ResourceBound(bound)
    value = strict_json(raw)
    validate_json(value)  # Also rejects exponent overflow and nesting beyond 64 levels.
    if not isinstance(value, dict):
        raise ValueError("worker envelope must be an object")
    return value
