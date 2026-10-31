"""One reusable native worker per process; bounded requests, idle time and I/O."""

from __future__ import annotations

import atexit
import subprocess  # nosec B404
import sys
import tempfile
import time
from pathlib import Path

_PROCESS = None
_DIRECTORY = None
_COUNT = 0
_LAST_USED = None
MAX_REQUESTS = 128
IDLE_SECONDS = 30


def close_worker():
    global _PROCESS, _DIRECTORY, _COUNT, _LAST_USED
    process, directory = _PROCESS, _DIRECTORY
    # Drop references even if a concurrent child exit or failed reap raises.
    _PROCESS, _DIRECTORY, _COUNT, _LAST_USED = None, None, 0, None
    try:
        if process is not None:
            try:
                if process.poll() is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                process.wait(timeout=5)
            finally:
                if process.stdin:
                    process.stdin.close()
    finally:
        if directory is not None:
            directory.cleanup()


def invoke(payload, timeout_seconds):
    from .pyreason_connector import MAX_OUTPUT_BYTES, WorkerFailure, canonical_json
    from .pyreason_io import read_json

    global _PROCESS, _DIRECTORY, _COUNT, _LAST_USED
    started = time.monotonic()
    if (
        _PROCESS is None
        or _PROCESS.poll() is not None
        or _COUNT >= MAX_REQUESTS
        or (_LAST_USED is not None and started - _LAST_USED >= max(0, IDLE_SECONDS - 1))
    ):
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
        output = read_json(target, MAX_OUTPUT_BYTES, "output_bound")
        _LAST_USED = time.monotonic()
        if _LAST_USED - started >= timeout_seconds:
            raise subprocess.TimeoutExpired("pyreason_pool", timeout_seconds)
        return output
    except BaseException:
        close_worker()
        raise
    finally:
        source.unlink(missing_ok=True)
        target.unlink(missing_ok=True)


atexit.register(close_worker)
