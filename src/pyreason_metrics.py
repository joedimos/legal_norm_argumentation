"""Finite latency accounting shared by the native producer and release gate."""

from __future__ import annotations

import math


def milliseconds(value):
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        raise ValueError("latency must be a finite nonnegative number")
    return value


def latency_summary(samples):
    if not isinstance(samples, list) or not 1 <= len(samples) <= 128:
        raise ValueError("latency samples must be a bounded array")
    ordered = sorted(milliseconds(x) for x in samples)
    milliseconds(sum(samples))
    count = len(ordered)
    return {
        "count": count,
        "p50_ms": ordered[(count - 1) // 2],
        "p95_ms": ordered[(95 * count + 99) // 100 - 1],
        "max_ms": ordered[-1],
        "samples_ms": list(samples),
    }


def validate_performance(performance, iterations):
    if type(iterations) is not int or not 1 <= iterations <= 128 or not isinstance(performance, dict):
        raise ValueError("invalid performance population")
    schedule = performance.get("norm_counts")
    if not isinstance(schedule, list) or any(type(n) is not int for n in schedule):
        raise ValueError("workload sizes must be integers")
    if (
        performance.get("population") != "after_explicit_startup_probe"
        or performance.get("startup_included_in_samples") is not False
        or type(performance.get("native_is_faster")) is not bool
        or performance.get("norm_counts") != [(1, 8, 32, 64)[i % 4] for i in range(iterations)]
    ):
        raise ValueError("performance population or workload differs")
    totals = {}
    for backend in ("native", "core"):
        summary = performance[backend]
        if not isinstance(summary, dict) or type(summary.get("count")) is not int or summary["count"] != iterations:
            raise ValueError("incomplete latency population")
        expected = latency_summary(summary["samples_ms"])
        if len(summary["samples_ms"]) != iterations:
            raise ValueError("latency sample count differs")
        for name in ("p50_ms", "p95_ms", "max_ms"):
            if milliseconds(summary[name]) != expected[name]:
                raise ValueError("latency summary contradicts raw observations")
        totals[backend] = sum(summary["samples_ms"])
    if performance["native_is_faster"] != (totals["native"] < totals["core"]):
        raise ValueError("speed comparison contradicts raw observations")
    return totals
