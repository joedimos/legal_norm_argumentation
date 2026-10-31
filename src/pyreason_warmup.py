"""Bounded explicit native startup; no live model or external service is used."""
from __future__ import annotations

import json
import subprocess  # nosec B404
import time

from .pyreason_connector import ResourceBound, run_worker, worker_transport
from .pyreason_graph import infer_graph
from .pyreason_runtime import installed_identity
from .resources import resource_path


def warm_native(*, timeout_seconds=300, worker_mode="pooled"):
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 300:
        raise ValueError("warmup timeout must be 1-300 seconds")
    if not isinstance(worker_mode,str) or worker_mode not in {"isolated","pooled"}:
        raise ValueError("unsupported native worker mode")
    started = time.monotonic()
    report = {"schema": "vle-pyreason-warmup/v1", "status": "unknown", "worker_mode": worker_mode,
              "timeout_seconds": timeout_seconds, "decision_authorized": False, "legal_certification": False}
    try:
        with worker_transport(worker_mode):
            report["runtime"] = run_worker({"mode": "runtime_probe"}, timeout_seconds)["annotations"]
            if report["runtime"] != installed_identity():
                raise ValueError("native runtime fingerprint differs")
            remaining = int(timeout_seconds - (time.monotonic() - started))
            if remaining < 1:
                raise ResourceBound("warmup_total_time_bound")
            program = json.loads(resource_path("examples", "pyreason-graph.json").read_text())
            inferred = infer_graph(program, timeout_seconds=remaining)
            if inferred["status"] != "computed" or inferred["derived_facts"] != {"licensed": True}:
                raise ValueError("native graph warmup did not complete")
            if time.monotonic() >= started + timeout_seconds:
                raise ResourceBound("warmup_total_time_bound")
            report["status"] = "ready"
    except ResourceBound as exc:
        report.update(reason=str(exc), bound_hit=True)
    except subprocess.TimeoutExpired as exc:
        report.update(reason="warmup_worker_timeout", bound_hit=True, worker_timeout_seconds=exc.timeout)
    except Exception:
        report["reason"] = "native_warmup_incomplete"
    report["elapsed_ms"] = (time.monotonic() - started) * 1000
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--worker-mode", choices=["isolated", "pooled"], default="pooled")
    args = parser.parse_args()
    result = warm_native(timeout_seconds=args.timeout_seconds, worker_mode=args.worker_mode)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
