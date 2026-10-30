"""Synthetic unreviewed transport, not evidence about native engine semantics."""

import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import pyreason_pool as pool
from src.pyreason_connector import BACKEND_VERSION, ResourceBound, WorkerFailure, digest, run_worker, worker_transport


@pytest.fixture
def processes(monkeypatch):
    pool.close_worker()
    created = []

    class Process:
        def __init__(self, cmd, **kwargs):
            self.root = Path(cmd[-1])
            self.dead = False
            self.respond = True
            self.bad = False
            self.stdin = SimpleNamespace(write=self.write, flush=lambda: None, close=lambda: None)
            created.append(self)

        def write(self, data):
            if self.respond:
                id = data.decode().strip()
                payload = json.loads((self.root / (id + ".input.json")).read_text())
                result = {
                    "status": "observed",
                    "backend_version": BACKEND_VERSION,
                    "input_sha256": "0" * 64 if self.bad else digest(payload),
                    "annotations": payload,
                }
                (self.root / (id + ".output.json")).write_text(json.dumps(result))

        def poll(self):
            return 0 if self.dead else None

        def kill(self):
            self.dead = True

        def wait(self, **kwargs):
            return 0

    monkeypatch.setattr(subprocess, "Popen", Process)
    yield created
    pool.close_worker()


def test_pool_reuses_worker_without_replaying_or_reusing_results(processes, monkeypatch):
    with worker_transport("pooled"):
        assert run_worker({"value": 1}, 1)["annotations"] == {"value": 1}
        assert run_worker({"value": 2}, 1)["annotations"] == {"value": 2}
        assert len(processes) == 1
        assert list(processes[0].root.iterdir()) == []
        monkeypatch.setattr(pool, "MAX_REQUESTS", 2)
        assert run_worker({"value": 3}, 1)["annotations"] == {"value": 3}
        assert len(processes) == 2 and processes[0].dead
    assert run_worker.__module__ == "src.pyreason_connector"


def test_expired_idle_worker_restarts_and_invalid_receipt_recycles(processes):
    with worker_transport("pooled"):
        run_worker({}, 1)
        processes[0].dead = True
        run_worker({}, 1)
        assert len(processes) == 2
        processes[1].bad = True
        with pytest.raises(WorkerFailure):
            run_worker({}, 1)
        assert pool._PROCESS is None
        run_worker({}, 1)
        assert len(processes) == 3


def test_deadline_kills_worker_and_never_retries(processes, monkeypatch):
    with worker_transport("pooled"):
        run_worker({}, 1)
        processes[0].respond = False
        clock = iter([0.0, 2.0])
        monkeypatch.setattr(pool.time, "monotonic", lambda: next(clock))
        with pytest.raises(subprocess.TimeoutExpired):
            run_worker({}, 1)
        assert processes[0].dead and len(processes) == 1 and pool._PROCESS is None


def test_late_or_oversized_reply_is_rejected(processes, monkeypatch):
    import src.pyreason_connector as connector

    with worker_transport("pooled"):
        monkeypatch.setattr(connector, "MAX_OUTPUT_BYTES", 1)
        with pytest.raises(ResourceBound):
            run_worker({}, 1)
        assert pool._PROCESS is None
        monkeypatch.setattr(connector, "MAX_OUTPUT_BYTES", 2_000_000)
        clock = iter([0.0, 2.0])
        monkeypatch.setattr(pool.time, "monotonic", lambda: next(clock))
        with pytest.raises(subprocess.TimeoutExpired):
            run_worker({}, 1)
        assert pool._PROCESS is None


def test_server_eof_invalid_id_and_request_limit(monkeypatch, tmp_path):
    import select

    from src import pyreason_worker as worker

    monkeypatch.setattr(sys, "argv", ["worker", "--serve", str(tmp_path)])
    monkeypatch.setattr(select, "select", lambda *args: ([sys.stdin], [], []))
    for text in ("../escape\n", "\n", "10000\n"):
        monkeypatch.setattr(sys, "stdin", io.StringIO(text))
        worker.main()
        assert list(tmp_path.iterdir()) == []
    calls = []
    monkeypatch.setattr(worker, "process_request", lambda a, b: calls.append((a, b)))
    monkeypatch.setattr(sys, "stdin", io.StringIO("0\n" * 129))
    worker.main()
    assert len(calls) == 128 and calls[0][0].parent == tmp_path
    monkeypatch.setattr(select, "select", lambda *args: ([], [], []))
    calls.clear()
    worker.main()
    assert calls == []


@pytest.mark.parametrize("timeout", [0, True, float("nan"), float("inf"), 301])
def test_transport_bounds_are_checked_before_start(timeout, processes):
    with pytest.raises(ValueError):
        run_worker({}, timeout)
    assert processes == []
    with pytest.raises(ValueError), worker_transport("invalid"):
        pass
