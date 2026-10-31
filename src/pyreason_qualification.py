"""Native-only, bounded correctness and latency qualification. No legal approvals."""

from __future__ import annotations

import json
import re
import subprocess  # nosec B404
import time
from dataclasses import replace

from .compliance import evaluate
from .defeasible import activation
from .norm_codec import decode_norm
from .normative_model import FactContext
from .pyreason_connector import BACKEND_VERSION, ResourceBound, digest, run_worker, worker_transport
from .pyreason_graph import infer_graph
from .pyreason_metrics import latency_summary
from .pyreason_normative import run_activation_backend
from .pyreason_runtime import contract_sha256
from .resources import resource_path


def qualification_corpus():
    return json.loads(resource_path("benchmarks", "pyreason-native-v1.json").read_text())


def qualify_native(*, iterations=32, timeout_seconds=600, worker_mode="pooled", revision=None):
    if (
        type(iterations) is not int
        or not 1 <= iterations <= 128
        or type(timeout_seconds) is not int
        or not 1 <= timeout_seconds <= 900
    ):
        raise ValueError("qualification bounds: 1-128 iterations, 1-900 seconds")
    if revision is not None and (not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)):
        raise ValueError("revision must be a full lowercase Git SHA")
    if not isinstance(worker_mode,str) or worker_mode not in {"isolated","pooled"}:
        raise ValueError("unsupported native worker mode")
    corpus = qualification_corpus()
    report = {
        "schema": "vle-pyreason-qualification/v2",
        "status": "unknown",
        "provider": "native",
        "corpus_sha256": digest(corpus),
        "adapter_contract_sha256": contract_sha256(),
        "build_revision": revision,
        "scope": "synthetic_unreviewed_implementation_and_performance_only",
        "worker_mode": worker_mode,
        "bounds": {"timeout_seconds": timeout_seconds, "iterations": iterations, "max_cases": 256, "max_worker_seconds": 300},
        "checks": [],
        "decision_authorized": False,
        "legal_certification": False,
    }
    started = time.monotonic()
    deadline = started + timeout_seconds

    def budget():
        remaining = min(300, int(deadline - time.monotonic()))
        if remaining < 1:
            raise ResourceBound("qualification_total_time_bound")
        return remaining

    def check(name, passed, **details):
        report["checks"].append({"name": name, "passed": bool(passed), **details})
        if not passed:
            raise ValueError("native qualification mismatch")

    try:
        with worker_transport(worker_mode):
            stamp = time.monotonic()
            identity = run_worker({"mode": "runtime_probe"}, budget())["annotations"]
            if identity.get("provider") != "native" or identity["packages"]["pyreason"] != BACKEND_VERSION or identity.get("adapter_contract_sha256") != report["adapter_contract_sha256"]:
                raise ValueError("not the native engine")
            report["runtime"] = identity
            report["startup_ms"] = (time.monotonic() - stamp) * 1000
            # Many independent point contexts become separate nodes in bounded batches.
            points = corpus["activation_cases"]
            if len(points) > 256:
                raise ResourceBound("qualification_case_bound")
            passed = 0
            for offset in range(0, len(points), 32):
                batch = points[offset : offset + 32]
                norms = []
                facts = {}
                for i, c in enumerate(batch):
                    n = decode_norm(c["norm"])
                    rename = {p.name: f"case{i}:{p.name}" for p in (*n.conditions, *n.exceptions)}
                    norms.append(
                        replace(
                            n,
                            id=f"case{i}",
                            conditions=tuple(replace(p, name=rename[p.name]) for p in n.conditions),
                            exceptions=tuple(replace(p, name=rename[p.name]) for p in n.exceptions),
                        )
                    )
                    facts.update({rename[k]: v for k, v in c["facts"].items()})
                context = FactContext(facts)
                result = run_activation_backend(norms, context, timeout_seconds=budget())
                check(f"activation_batch_{offset // 32}", result["status"] == "computed")
                check(
                    f"activation_values_{offset // 32}",
                    all(result["activations"][n.id]["status"] == activation(n, context).status for n in norms),
                )
                passed += len(norms)
            report["native_activation_cases"] = passed
            for case in corpus["graph_cases"]:
                result = infer_graph(case["program"], timeout_seconds=budget())
                check(case["id"], result["status"] == "computed" and result["answers"] == case["expected"],
                      inference_status=result["status"], failure_reason=result.get("reason"),
                      disagreements=result.get("disagreements", []))
            # New predicates must affect a normative verdict, not merely appear in diagnostics.
            example = json.loads(resource_path("examples", "pyreason-normative.json").read_text())
            graph = json.loads(resource_path("examples", "pyreason-graph.json").read_text())
            rule = decode_norm(example["norms"][0])
            context = FactContext({"protected": True}, observed_actions=("reproduce",))
            base = evaluate([rule], context)
            enriched = evaluate(
                [rule],
                context,
                backend="pyreason",
                pyreason_program=graph,
                pyreason_worker_mode=worker_mode,
                pyreason_timeout_seconds=budget(),
            )
            check(
                "derived_fact_controls_compliance",
                base["status"] == "indeterminate" and enriched["status"] == "compliant",
            )
            latencies = []
            baseline = []
            for i in range(iterations):
                size = (1, 8, 32, 64)[i % 4]
                workload = [replace(rule, id=f"load_{j}", action=f"act_{j}") for j in range(size)]
                stamp = time.monotonic()
                expected = evaluate(workload, context)
                baseline.append((time.monotonic() - stamp) * 1000)
                stamp = time.monotonic()
                actual = evaluate(
                    workload,
                    context,
                    backend="pyreason",
                    pyreason_worker_mode=worker_mode,
                    pyreason_timeout_seconds=budget(),
                )
                latencies.append((time.monotonic() - stamp) * 1000)
                check(
                    f"performance_result_{i}",
                    actual["status"] == expected["status"] and actual["reasoning"]["backend"]["status"] == "computed",
                )

            report["performance"] = {
                "population": "after_explicit_startup_probe",
                "norm_counts": [(1, 8, 32, 64)[i % 4] for i in range(iterations)],
                "native": latency_summary(latencies),
                "core": latency_summary(baseline),
                "startup_included_in_samples": False,
                "native_is_faster": sum(latencies) < sum(baseline),
            }
            if time.monotonic() >= deadline:
                raise ResourceBound("qualification_total_time_bound")
            if contract_sha256() != report["adapter_contract_sha256"]:
                raise ValueError("adapter changed during qualification")
            report["status"] = "passed"
            report["native_graph_cases"] = len(corpus["graph_cases"])
    except ResourceBound as exc:
        report.update(reason=str(exc), truncated=True, bound_hit=True)
    except subprocess.TimeoutExpired as exc:
        report.update(reason="qualification_worker_timeout", truncated=True, bound_hit=True,
                      worker_timeout_seconds=exc.timeout)
    except Exception:
        # Full per-check failures remain reviewable; never claim partial qualification.
        report["reason"] = "native_qualification_failed_or_incomplete"
    report["elapsed_ms"] = (time.monotonic() - started) * 1000
    return report


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iterations", type=int, default=32)
    p.add_argument("--timeout-seconds", type=int, default=600)
    p.add_argument("--worker-mode", choices=["isolated", "pooled"], default="pooled")
    p.add_argument("--revision")
    args = p.parse_args()
    result = qualify_native(
        iterations=args.iterations,
        timeout_seconds=args.timeout_seconds,
        worker_mode=args.worker_mode,
        revision=args.revision,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
