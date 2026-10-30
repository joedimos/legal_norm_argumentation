"""Bounded advisory bridge; PyReason annotations are not legal verdicts."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess  # nosec B404
import sys
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import BoundedSemaphore

PREDICATES = ("accepted", "defeated", "reinstated", "strengthened")
BACKEND_VERSION = "3.8.0"
MAX_OUTPUT_BYTES = 2_000_000
MAX_PARALLEL_WORKERS = 1
_WORKER_SLOT = BoundedSemaphore(MAX_PARALLEL_WORKERS)
_TRANSPORT = ContextVar("pyreason_transport", default="isolated")


@contextmanager
def worker_transport(mode):
    if mode not in {"isolated", "pooled"}:
        raise ValueError("unsupported PyReason transport")
    token = _TRANSPORT.set(mode)
    try:
        yield
    finally:
        _TRANSPORT.reset(token)


RULES = (
    ("defeat", "defeated(Y):[1,1] <- attacks(X,Y):[1,1], accepted(X):[0.5,1]"),
    ("reinstate", "reinstated(Y):[1,1] <- attacks(X,Y):[1,1], defeated(X):[1,1]"),
    ("support", "strengthened(Y):[1,1] <- supports(X,Y):[1,1], accepted(X):[0.5,1]"),
)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class ResourceBound(ValueError):
    """No partial annotations may escape when a configured limit is reached."""


class WorkerFailure(ValueError):
    """Sanitized worker failure; never contains engine exception text."""


def current_worker_mode():
    return _TRANSPORT.get()


def run_worker(payload, timeout_seconds):
    """Bound native-engine concurrency per process; never queue without a bound."""
    if (type(timeout_seconds) not in {int, float} or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 300):
        raise ValueError("timeout must be positive and bounded by 300 seconds")
    if not _WORKER_SLOT.acquire(blocking=False):
        raise ResourceBound("worker_concurrency_bound")
    try:
        if _TRANSPORT.get() == "pooled":
            from .pyreason_pool import close_worker, invoke

            try:
                return validate_receipt(invoke(payload, timeout_seconds), payload)
            except BaseException:
                close_worker()
                raise
        return _invoke_worker(payload, timeout_seconds)
    finally:
        _WORKER_SLOT.release()


def _invoke_worker(payload, timeout_seconds):
    """Shared pinned, isolated transport for both supported program fragments."""
    encoded = canonical_json(payload)
    with tempfile.TemporaryDirectory(prefix="vle-pyreason-") as directory:
        source, target = Path(directory) / "input.json", Path(directory) / "output.json"
        source.write_text(encoded, encoding="utf-8")
        subprocess.run(
            [sys.executable, "-m", "src.pyreason_worker", str(source), str(target)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
            check=True,
        )  # nosec B603
        if target.stat().st_size > MAX_OUTPUT_BYTES:
            raise ResourceBound("output_bound")
        output = json.loads(target.read_text(encoding="utf-8"))
    return validate_receipt(output, payload)


def validate_receipt(output, payload):
    if output.get("status") != "observed":
        raise WorkerFailure(
            "pyreason_unsupported_version"
            if output.get("error") == "unsupported_version"
            else "pyreason_unavailable_or_incompatible"
        )
    if output.get("backend_version") != BACKEND_VERSION or output.get("input_sha256") != digest(payload):
        raise WorkerFailure("pyreason_invalid_or_incompatible")
    return output


def interval(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = (value, value)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("invalid interval")
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in value):
        raise ValueError("invalid interval")
    low, high = map(float, value)
    if not 0 <= low <= high <= 1:
        raise ValueError("invalid interval")
    return [low, high]


def prepare(arguments, attacks, supports, timesteps):
    if type(timesteps) is not int:
        raise ValueError("horizon must be 1..100")
    if not 1 <= timesteps <= 100:
        raise ResourceBound("horizon_bound")
    if not isinstance(arguments, dict):
        raise ValueError("invalid arguments")
    if len(arguments) > 256:
        raise ResourceBound("argument_bound")
    if any(not isinstance(key, str) or not key or len(key) > 512 for key in arguments):
        raise ValueError("invalid argument identifier")
    ids = sorted(arguments)
    # Never interpolate source identifiers into executable rule syntax.
    names = {key: f"n{i}" for i, key in enumerate(ids)}
    edges = []
    for relation in (attacks, supports if supports is not None else set()):
        if not isinstance(relation, (set, frozenset, tuple, list)):
            raise ValueError("invalid relation")
        if len(relation) > 4096:
            raise ResourceBound("edge_bound")
        normalized = set()
        for edge in relation:
            if (
                not isinstance(edge, (tuple, list))
                or len(edge) != 2
                or any(not isinstance(key, str) or key not in names for key in edge)
            ):
                raise ValueError("dangling edge")
            normalized.add((names[edge[0]], names[edge[1]]))
        edges.append(sorted(normalized))
    if any(not isinstance(metadata, dict) for metadata in arguments.values()):
        raise ValueError("invalid argument metadata")
    return {
        "nodes": {names[key]: interval(arguments[key].get("confidence", [0, 1])) for key in ids},
        "attacks": edges[0],
        "supports": edges[1],
        "timesteps": timesteps,
        "rule_set_sha256": digest(RULES),
    }, names


class PyReasonConnector:
    """Each run uses a fresh process to isolate PyReason's global state.

    Legacy predicate keys are preserved, but are advisory annotations only.
    No fixed-point, Dung acceptance, or legal-compliance claim is inferred.
    """

    def run_reasoning(self, arguments, attacks, supports=None, timesteps=10, *, timeout_seconds=60):
        result = {key: {} for key in PREDICATES}
        result.update(
            schema="vle-pyreason-report/v1",
            status="unknown",
            converged=False,
            decision_authorized=False,
            legal_certification=False,
            review_status="unreviewed",
            semantics="advisory_graph_annotations",
            bounds={
                "timesteps": timesteps if type(timesteps) is int and 1 <= timesteps <= 100 else None,
                "timeout_seconds": timeout_seconds
                if type(timeout_seconds) is int and 1 <= timeout_seconds <= 300
                else None,
                "max_arguments": 256,
                "max_edges_per_relation": 4096,
                "max_timesteps": 100,
                "max_timeout_seconds": 300,
                "max_output_bytes": MAX_OUTPUT_BYTES,
                "max_parallel_workers": MAX_PARALLEL_WORKERS,
            },
        )
        try:
            if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 300:
                raise ValueError("timeout must be 1..300 seconds")
            payload, names = prepare(arguments, attacks, supports, timesteps)
            result["input_sha256"] = digest(payload)
            result["argument_map_sha256"] = digest(names)
            result["rule_set_sha256"] = payload["rule_set_sha256"]
            output = run_worker(payload, timeout_seconds)
            annotations = self._extract_results(output["annotations"], set(names.values()))
            if annotations["accepted"] != payload["nodes"]:
                raise ValueError("lost or changed input annotations")
            reverse = {value: key for key, value in names.items()}
            for predicate in PREDICATES:
                result[predicate] = {reverse[key]: value for key, value in annotations[predicate].items()}
            result.update(status="observed_within_bound", backend_version=output["backend_version"])
        except subprocess.TimeoutExpired:
            result.update(error="pyreason_timeout", bound_hit=True)
        except ResourceBound as exc:
            result.update(error="pyreason_" + str(exc), bound_hit=True)
        except WorkerFailure as exc:
            result["error"] = str(exc)
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
            # No partial annotations or provider exception text escape.
            for key in PREDICATES:
                result[key] = {}
            result["error"] = "pyreason_invalid_or_incompatible"
        return result

    @staticmethod
    def _extract_results(data, arg_ids):
        if not isinstance(data, dict) or set(data) != set(PREDICATES):
            raise ValueError("invalid backend report")
        result = {}
        for predicate in PREDICATES:
            values = data[predicate]
            if not isinstance(values, dict) or set(values) != arg_ids:
                raise ValueError("incomplete backend report")
            result[predicate] = {key: interval(values[key]) for key in sorted(arg_ids)}
        return result
