"""Bounded advisory bridge; PyReason annotations are not legal verdicts."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path

PREDICATES = ("accepted", "defeated", "reinstated", "strengthened")


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
    if type(timesteps) is not int or not 1 <= timesteps <= 100:
        raise ValueError("horizon must be 1..100")
    if not isinstance(arguments, dict) or len(arguments) > 256:
        raise ValueError("argument bound")
    if any(not isinstance(key, str) or not key or len(key) > 512 for key in arguments):
        raise ValueError("invalid argument identifier")
    ids = sorted(arguments)
    # Never interpolate source identifiers into executable rule syntax.
    names = {key: f"n{i}" for i, key in enumerate(ids)}
    edges = []
    for relation in (attacks, supports or set()):
        if len(relation) > 4096:
            raise ValueError("edge bound")
        normalized = []
        for edge in relation:
            if len(edge) != 2 or any(key not in names for key in edge):
                raise ValueError("dangling edge")
            normalized.append([names[edge[0]], names[edge[1]]])
        edges.append(sorted(normalized))
    return {"nodes": {names[key]: interval(arguments[key].get("confidence", [0, 1])) for key in ids},
            "attacks": edges[0], "supports": edges[1], "timesteps": timesteps}, names


class PyReasonConnector:
    """Each run uses a fresh process to isolate PyReason's global state.

    Legacy predicate keys are preserved, but are advisory annotations only.
    No fixed-point, Dung acceptance, or legal-compliance claim is inferred.
    """

    def run_reasoning(self, arguments, attacks, supports=None, timesteps=10, *, timeout_seconds=60):
        result = {key: {} for key in PREDICATES}
        result.update(schema="vle-pyreason-report/v1", status="unknown", converged=False,
                      decision_authorized=False, legal_certification=False, review_status="unreviewed",
                      semantics="advisory_graph_annotations", bounds={"timesteps": timesteps,
                      "timeout_seconds": timeout_seconds, "max_arguments": 256, "max_edges_per_relation": 4096})
        try:
            if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 300:
                raise ValueError("timeout must be 1..300 seconds")
            payload, names = prepare(arguments, attacks, supports, timesteps)
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            result["input_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
            with tempfile.TemporaryDirectory(prefix="vle-pyreason-") as directory:
                source, target = Path(directory) / "input.json", Path(directory) / "output.json"
                source.write_text(encoded, encoding="utf-8")
                subprocess.run([sys.executable, "-m", "src.pyreason_worker", str(source), str(target)],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=timeout_seconds, check=True)  # nosec B603
                if target.stat().st_size > 2_000_000:
                    raise ValueError("output bound")
                output = json.loads(target.read_text(encoding="utf-8"))
            if output.get("status") != "observed":
                result["error"] = "pyreason_unavailable_or_incompatible"
                return result
            if not isinstance(output.get("backend_version"), str) or not output["backend_version"]:
                raise ValueError("missing backend version")
            annotations = self._extract_results(output["annotations"], set(names.values()))
            reverse = {value: key for key, value in names.items()}
            for predicate in PREDICATES:
                result[predicate] = {reverse[key]: value for key, value in annotations[predicate].items()}
            result.update(status="observed_within_bound", backend_version=output["backend_version"])
        except subprocess.TimeoutExpired:
            result.update(error="pyreason_timeout", bound_hit=True)
        except (ValueError, TypeError, KeyError, AttributeError, OSError, subprocess.SubprocessError):
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
