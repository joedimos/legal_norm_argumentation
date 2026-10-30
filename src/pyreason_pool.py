"""One reusable native worker per process; bounded requests, idle time and I/O."""

from __future__ import annotations

import atexit
import json
import subprocess  # nosec B404
import sys
import tempfile
import time
from pathlib import Path

_PROCESS = None
_DIRECTORY = None
_COUNT = 0
MAX_REQUESTS = 128
IDLE_SECONDS = 30


def close_worker():
    global _PROCESS, _DIRECTORY, _COUNT
    if _PROCESS is not None:
        if _PROCESS.poll() is None:
            _PROCESS.kill()
        _PROCESS.wait(timeout=5)
        if _PROCESS.stdin:
            _PROCESS.stdin.close()
    if _DIRECTORY is not None:
        _DIRECTORY.cleanup()
    _PROCESS, _DIRECTORY, _COUNT = None, None, 0


def invoke(payload, timeout_seconds):
    from .pyreason_connector import MAX_OUTPUT_BYTES, ResourceBound, WorkerFailure, canonical_json

    global _PROCESS, _DIRECTORY, _COUNT
    started = time.monotonic()
    if _PROCESS is None or _PROCESS.poll() is not None or _COUNT >= MAX_REQUESTS:
        close_worker()
        _DIRECTORY = tempfile.TemporaryDirectory(prefix="vle-pyreason-pool-")
        _PROCESS = subprocess.Popen(  # nosec B603
            [sys.executable, "-m", "src.pyreason_worker", "--serve", _DIRECTORY.name],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    request_id = str(_COUNT)
    _COUNT += 1
    source = Path(_DIRECTORY.name) / (request_id + ".input.json")
    target = Path(_DIRECTORY.name) / (request_id + ".output.json")
    try:
        source.write_text(canonical_json(payload), encoding="utf-8")
        _PROCESS.stdin.write((request_id + "\n").encode())
        _PROCESS.stdin.flush()
        while not target.exists():
            if time.monotonic() - started >= timeout_seconds:
                raise subprocess.TimeoutExpired("pyreason_pool", timeout_seconds)
            if _PROCESS.poll() is not None:
                raise WorkerFailure("pyreason_unavailable_or_incompatible")
            time.sleep(0.01)
        if time.monotonic() - started >= timeout_seconds:
            raise subprocess.TimeoutExpired("pyreason_pool", timeout_seconds)
        if target.stat().st_size > MAX_OUTPUT_BYTES:
            raise ResourceBound("output_bound")
        return json.loads(target.read_text(encoding="utf-8"))
    except BaseException:
        close_worker()
        raise
    finally:
        source.unlink(missing_ok=True)
        target.unlink(missing_ok=True)


atexit.register(close_worker)
