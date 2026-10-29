"""Versioned raw research requests for the selectable normative reasoning backend."""

from __future__ import annotations

from .compliance import evaluate
from .norm_codec import decode_norm
from .normative_model import FactContext
from .pyreason_connector import digest


def evaluate_request(request, *, backend=None, timeout_seconds=None):
    if (
        not isinstance(request, dict)
        or set(request) - {"schema", "norms", "context", "backend", "timeout_seconds"}
        or request.get("schema") != "vle-normative-evaluation/v1"
    ):
        raise ValueError("invalid normative evaluation request")
    selected = backend if backend is not None else request.get("backend", "core")
    timeout = timeout_seconds if timeout_seconds is not None else request.get("timeout_seconds", 60)
    if selected not in {"core", "pyreason"} or type(timeout) is not int or not 1 <= timeout <= 300:
        raise ValueError("invalid backend/timeout")
    raw = request.get("norms")
    if not isinstance(raw, list) or not 1 <= len(raw) <= 64:
        raise ValueError("normative evaluation requires 1-64 norms")
    norms = [decode_norm(n) for n in raw]
    data = request.get("context", {})
    if not isinstance(data, dict) or set(data) - {"facts", "actor", "observed_actions", "at_date"}:
        raise ValueError("invalid factual context")
    facts, actions = data.get("facts", {}), data.get("observed_actions", [])
    if (
        not isinstance(facts, dict)
        or len(facts) > 5000
        or any(not isinstance(k, str) or not k or len(k) > 512 or type(v) is not bool for k, v in facts.items())
    ):
        raise ValueError("facts must be bounded named Boolean observations")
    if (
        not isinstance(actions, list)
        or len(actions) > 1000
        or any(not isinstance(a, str) or not a or len(a) > 512 for a in actions)
    ):
        raise ValueError("invalid action observations")
    if any(
        data.get(key) is not None and (not isinstance(data[key], str) or not data[key] or len(data[key]) > 256)
        for key in ("actor", "at_date")
    ):
        raise ValueError("invalid actor/date")
    context = FactContext(
        facts=facts, actor=data.get("actor"), observed_actions=tuple(actions), at_date=data.get("at_date")
    )
    result = evaluate(norms, context, backend=selected, pyreason_timeout_seconds=timeout)
    return {
        "schema": "vle-normative-evaluation-result/v1",
        "status": result["status"],
        "backend": selected,
        "request_sha256": digest({"request": request, "backend": selected, "timeout_seconds": timeout}),
        "compliance": result,
        "review_status": "unreviewed",
        "decision_authorized": False,
        "legal_certification": False,
    }
