"""Shared fail-closed envelopes and deterministic resource limits for Phase 4."""
from __future__ import annotations

from time import monotonic

from .audit_bundle import canonical_json, digest


def integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return value


class BoundHit(Exception):
    pass


class Budget:
    def __init__(self, *, max_states=65536, timeout_ms=5000):
        self.max_states = integer(max_states, 1, 1000000, 'max_states')
        self.timeout_ms = integer(timeout_ms, 1, 60000, 'timeout_ms')
        self.started = monotonic()
        self.used = 0

    def tick(self):
        if self.used >= self.max_states:
            raise BoundHit('state_limit')
        if (monotonic() - self.started) * 1000 >= self.timeout_ms:
            raise BoundHit('timeout')
        self.used += 1

    def to_dict(self):
        return {'max_states': self.max_states, 'timeout_ms': self.timeout_ms, 'checked_states': self.used}


def envelope(schema, request, **fields):
    return {'schema': schema, **fields, 'request_sha256': digest(canonical_json(request)),
            'review_status': 'unreviewed', 'decision_authorized': False, 'legal_certification': False}


def fields(request, schema, allowed, required=()):
    if (not isinstance(request, dict) or request.get('schema') != schema
            or set(request) - ({'schema'} | set(allowed)) or not set(required) <= set(request)):
        raise ValueError(f'invalid {schema} request fields')
