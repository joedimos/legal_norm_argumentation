"""Synthetic/unreviewed boundary and fault regressions. No live model or network."""

import json
import subprocess
from pathlib import Path

import pytest

from src import pyreason_connector as connector
from src import pyreason_pool as pool
from src.pyreason_graph import infer_graph
from src.pyreason_metrics import latency_summary, validate_performance
from tests.test_pyreason_connector import receipt
from tests.test_pyreason_pool import processes as processes


@pytest.mark.parametrize("fault", ["duplicate_status", "duplicate_atom", "nan", "overflow", "deep", "utf8", "array"])
def test_corrupt_json_receipts_cannot_become_observations(monkeypatch, fault):
    def corrupted(cmd, **kwargs):
        raw = json.dumps(receipt(cmd))
        if fault == "duplicate_status":
            raw = '{"status":"unknown",' + raw[1:]
        elif fault == "duplicate_atom":
            raw = raw.replace('"n0": [1.0, 1.0]', '"n0": [0,0], "n0": [1,1]', 1)
        elif fault == "nan":
            raw = raw[:-1] + ',"ignored":NaN}'
        elif fault == "overflow":
            raw = raw[:-1] + ',"ignored":1e10000}'
        elif fault == "deep":
            raw = raw[:-1] + ',"ignored":' + ("[" * 66) + "0" + ("]" * 66) + "}"
        elif fault == "utf8":
            Path(cmd[-1]).write_bytes(b"\xff")
            return
        else:
            raw = "[]"
        Path(cmd[-1]).write_text(raw)

    monkeypatch.setattr(subprocess, "run", corrupted)
    report = connector.PyReasonConnector().run_reasoning({"a": {"confidence": [1, 1]}}, set())
    assert report["status"] == "unknown" and report["accepted"] == {}
    assert not report["decision_authorized"] and not report["legal_certification"]


def test_input_size_and_json_types_are_checked_before_any_process(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("invalid payload reached worker"))
    monkeypatch.setattr(connector, "MAX_INPUT_BYTES", 10)
    with pytest.raises(connector.ResourceBound, match="input_bound"):
        connector.run_worker({"large": "x" * 100}, 1)
    with pytest.raises(ValueError):
        connector.run_worker({1: "coerced-key"}, 1)


@pytest.mark.parametrize("mode", ["isolated", "pooled"])
def test_full_transport_deadline_rejects_a_late_valid_receipt(monkeypatch, mode):
    clock = [0.0]
    monkeypatch.setattr(connector.time, "monotonic", lambda: clock[0])
    closed = []

    def late(payload, timeout):
        clock[0] = 2.0
        return {
            "status": "observed",
            "backend_version": connector.BACKEND_VERSION,
            "input_sha256": connector.digest(payload),
            "annotations": {},
        }

    monkeypatch.setattr(connector, "_invoke_worker", late)
    monkeypatch.setattr(pool, "invoke", late)
    monkeypatch.setattr(pool, "close_worker", lambda: closed.append(True))
    with connector.worker_transport(mode), pytest.raises(subprocess.TimeoutExpired):
        connector.run_worker({}, 1)
    assert connector._WORKER_SLOT.acquire(blocking=False)
    connector._WORKER_SLOT.release()
    assert bool(closed) == (mode == "pooled")


def test_idle_worker_is_retired_before_the_child_idle_deadline(processes, monkeypatch):
    clock = [1.0]
    monkeypatch.setattr(pool.time, "monotonic", lambda: clock[0])
    with connector.worker_transport("pooled"):
        connector.run_worker({"first": True}, 1)
        clock[0] += pool.IDLE_SECONDS - 1
        assert connector.run_worker({"second": True}, 1)["annotations"] == {"second": True}
    assert len(processes) == 2 and processes[0].dead


def test_child_exit_race_during_cleanup_does_not_leak_state(processes, monkeypatch):
    with connector.worker_transport("pooled"):
        connector.run_worker({}, 1)
    process = processes[0]
    root = process.root

    def exited():
        process.dead = True
        raise ProcessLookupError("already exited")

    monkeypatch.setattr(process, "kill", exited)
    pool.close_worker()
    assert pool._PROCESS is None and pool._DIRECTORY is None and not root.exists()
    pool.close_worker()


def test_failed_reap_still_cleans_private_files_and_references(processes, monkeypatch):
    with connector.worker_transport("pooled"):
        connector.run_worker({}, 1)
    process = processes[0]

    def hung(**kwargs):
        raise subprocess.TimeoutExpired("synthetic-worker", 5)

    monkeypatch.setattr(process, "wait", hung)
    with pytest.raises(subprocess.TimeoutExpired):
        pool.close_worker()
    assert pool._PROCESS is None and pool._DIRECTORY is None and not process.root.exists()


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), True, 0, 301, [], object()])
def test_invalid_timeout_always_returns_serializable_unknown(timeout):
    report = infer_graph({}, timeout_seconds=timeout)
    assert report["status"] == "unknown" and report["bounds"]["timeout_seconds"] is None
    json.dumps(report, allow_nan=False)


def test_latency_convention_matches_historical_reports():
    summary = latency_summary([8, 2, 4, 1])
    assert summary["p50_ms"] == 2 and summary["p95_ms"] == 8 and summary["max_ms"] == 8
    path = Path(__file__).resolve().parents[1] / "docs/PYREASON_NATIVE_QUALIFICATION.json"
    historical = json.loads(path.read_text())
    validate_performance(historical["performance"], historical["bounds"]["iterations"])


@pytest.mark.parametrize("values", ["x" * 32, [], [True], [float("nan")], [float("inf")], [-1], [1e308] * 2])
def test_latency_arrays_require_actual_finite_nonnegative_observations(values):
    with pytest.raises(ValueError):
        latency_summary(values)


@pytest.mark.parametrize(
    "fault",
    ["samples", "p95", "core_count", "speed", "schedule", "boolean_schedule", "elapsed", "case_count", "timeout"],
)
def test_correct_hashes_and_signatures_do_not_excuse_bad_accounting(tmp_path, monkeypatch, fault):
    import tests.test_pyreason_release as fixtures
    from src.service.pyreason_release import pyreason_release_gate

    original = fixtures.write_json

    def invalid_native(path, value):
        if path.name == "native.json":
            if fault == "samples":
                value["performance"]["native"]["samples_ms"] = "x" * 32
            elif fault == "p95":
                value["performance"]["native"]["p95_ms"] = 0
            elif fault == "core_count":
                value["performance"]["core"]["count"] = 31
            elif fault == "speed":
                value["performance"]["native_is_faster"] = True
            elif fault == "schedule":
                value["performance"]["norm_counts"] = [64] * 32
            elif fault == "boolean_schedule":
                value["performance"]["norm_counts"][0] = True
            elif fault == "elapsed":
                value["elapsed_ms"] = 1
            elif fault == "case_count":
                value["native_activation_cases"] = 162.0
            else:
                value["bounds"]["timeout_seconds"] = True
        return original(path, value)

    monkeypatch.setattr(fixtures, "write_json", invalid_native)
    cfg, _, _ = fixtures.synthetic_backend_graph(tmp_path, monkeypatch)
    assert not pyreason_release_gate(cfg)["passed"]


def test_qualification_timeout_records_which_worker_bound_was_hit(monkeypatch):
    from src import pyreason_qualification as qualification

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("SECRET_COMMAND", 1)

    monkeypatch.setattr(qualification, "run_worker", timeout)
    report = qualification.qualify_native(iterations=1, timeout_seconds=2)
    assert report["status"] == "unknown" and report["truncated"] and report["bound_hit"]
    assert report["bounds"]["max_worker_seconds"] == 300 and report["worker_timeout_seconds"] == 1
    assert "SECRET_COMMAND" not in json.dumps(report)


def test_warmup_never_marks_a_late_or_mismatching_runtime_ready(monkeypatch):
    from src import pyreason_warmup as warmup

    clock = [0.0]
    monkeypatch.setattr(warmup.time, "monotonic", lambda: clock[0])
    identity = {"provider": "synthetic_fixture_only"}
    monkeypatch.setattr(warmup, "run_worker", lambda *a: {"annotations": identity})
    monkeypatch.setattr(warmup, "installed_identity", lambda: identity)

    def late(*a, **k):
        clock[0] = 2.0
        return {"status": "computed", "derived_facts": {"licensed": True}}

    monkeypatch.setattr(warmup, "infer_graph", late)
    report = warmup.warm_native(timeout_seconds=1)
    assert report["status"] == "unknown" and report["bound_hit"]
    clock[0] = 0.0
    monkeypatch.setattr(warmup, "installed_identity", lambda: {"provider": "different"})
    assert warmup.warm_native(timeout_seconds=1)["status"] == "unknown"


def test_private_worker_bounds_reads_and_never_writes_oversized_receipts(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from src import pyreason_worker as worker

    source, target = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text('{"large":"' + ("x" * 100) + '"}')
    monkeypatch.setattr(worker, "version", lambda _: connector.BACKEND_VERSION)
    monkeypatch.setitem(sys.modules, "pyreason", SimpleNamespace())
    monkeypatch.setattr(worker, "MAX_INPUT_BYTES", 10)
    worker.process_request(source, target)
    assert json.loads(target.read_text()) == {"status": "unknown", "error": "input_bound"}
    monkeypatch.setattr(worker, "MAX_INPUT_BYTES", 1000)
    monkeypatch.setattr(worker, "MAX_OUTPUT_BYTES", 256)
    monkeypatch.setattr(worker, "execute", lambda *a: {"secret": "PRIVATE" * 1000})
    worker.process_request(source, target)
    assert json.loads(target.read_text()) == {"status": "unknown", "error": "output_bound"}
    assert target.stat().st_size <= 256 and "PRIVATE" not in target.read_text()
