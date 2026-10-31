"""Synthetic, unreviewed HTTP fixtures: loopback only, never real providers."""

from __future__ import annotations

import gzip
import hashlib
import io
import subprocess
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread
from urllib.parse import urlsplit

import pytest
import urllib3

from src import http_transport as transport
from src import http_worker, legal_transport, model_runtime
from src.audit_bundle import canonical_json, digest
from src.http_transport import BoundedHTTPClient, HTTPTransportError
from src.json_boundary import strict_json
from src.legal_transport import ImportTransportError, LegalJSONTransport
from src.model_runtime import ModelError, request_json

SYNTHETIC = b'{"synthetic":true,"review_status":"unreviewed"}'


@pytest.fixture(autouse=True)
def loopback_only(monkeypatch):
    original = BoundedHTTPClient.request

    def guarded(self, method, url, **kwargs):
        assert urlsplit(url).hostname == "127.0.0.1", "Tests may only contact the loopback fixture"
        return original(self, method, url, **kwargs)

    monkeypatch.setattr(BoundedHTTPClient, "request", guarded)


@contextmanager
def server(*, mode="normal", body=SYNTHETIC, status=200, headers=None):
    calls, stop = [], Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.reply()

        def do_POST(self):
            self.reply()

        def reply(self):
            content = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            calls.append((self.command, self.path, dict(self.headers), content))
            if mode == "headers" and stop.wait(4):
                return
            try:
                self.send_response(status)
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.end_headers()
                if mode == "drip":
                    for value in body:
                        self.wfile.write(bytes([value]))
                        self.wfile.flush()
                        if stop.wait(0.12):
                            return
                else:
                    self.wfile.write(body)
            except (OSError, ConnectionError):
                pass  # A deadline deliberately kills the client mid-response.

        def log_message(self, *_):
            pass

    instance = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    instance.daemon_threads = True
    thread = Thread(target=lambda: instance.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{instance.server_port}/fixture", calls
    finally:
        stop.set()
        instance.shutdown()
        instance.server_close()
        thread.join(1)


@pytest.mark.parametrize("mode", ["drip", "headers"])
@pytest.mark.parametrize("client", ["model", "import"])
def test_wall_deadline_stops_slow_body_and_headers_without_retry(mode, client):
    with server(mode=mode) as (url, calls):
        started = time.monotonic()
        if client == "model":
            with pytest.raises(ModelError, match="transport_timeout"):
                request_json("GET", url, timeout=1)
        else:
            with pytest.raises(ImportTransportError, match="request_time_bound"):
                LegalJSONTransport(timeout=1).request("GET", url)
        elapsed = time.monotonic() - started
        # Whole fixture takes >4 seconds; allow scheduler/kill margin, not a
        # timing assertion at the exact deadline. Child startup counts too.
        assert elapsed < 2.5
        assert len(calls) == 1


def test_completed_request_case_insensitive_headers_and_one_shot_post():
    with server(headers={"Content-Type": "application/json"}) as (url, calls):
        result = BoundedHTTPClient().request("POST", url, body="é", timeout=3)
        assert result.status == 200 and result.headers["content-type"] == "application/json"
        assert result.read(decode_content=False) == SYNTHETIC
        result.release_conn()
        result.close()
        assert len(calls) == 1 and calls[0][3] == "é".encode()
    with server() as (url, calls):
        assert LegalJSONTransport().post_json(url, {"synthetic": True}, {}) == strict_json(SYNTHETIC)
        assert strict_json(calls[0][3]) == {"synthetic": True}


def test_redirect_and_rate_limit_do_not_drain_error_body_or_follow():
    with server(mode="drip", status=302, headers={"Location": "/never-follow"}) as (url, calls):
        with pytest.raises(ModelError, match="provider_http_error"):
            request_json("POST", url, timeout=3, payload={"synthetic": True}, api_key="synthetic-secret")
        assert len(calls) == 1 and calls[0][2]["Authorization"] == "Bearer synthetic-secret"
    with server(mode="drip", status=429, headers={"Retry-After": "7200"}) as (url, calls):
        with pytest.raises(ImportTransportError) as error:
            LegalJSONTransport().request("GET", url)
        assert error.value.retry_after == 7200 and len(calls) == 1


@pytest.mark.parametrize("length", [None, "100000", "not-a-number"])
def test_byte_bound_with_missing_or_invalid_length(length):
    headers = {"Content-Length": length} if length else {}
    with server(body=b"x" * 20, headers=headers) as (url, _):
        with pytest.raises(ImportTransportError, match="response_size_bound"):
            LegalJSONTransport(max_bytes=10).request("GET", url)


def test_gzip_is_bounded_after_decoding_and_models_require_identity():
    raw = gzip.compress(b"x" * 3000)
    with server(body=raw, headers={"Content-Encoding": "gzip"}) as (url, _):
        with pytest.raises(ImportTransportError, match="response_size_bound"):
            LegalJSONTransport(max_bytes=50).request("GET", url)
        with pytest.raises(ModelError, match="unexpected_content_encoding"):
            request_json("GET", url, timeout=3)
    with server(body=gzip.compress(SYNTHETIC), headers={"Content-Encoding": "gzip"}) as (url, _):
        assert LegalJSONTransport().request("GET", url) == strict_json(SYNTHETIC)


@pytest.mark.parametrize("raw", [b'{"x":1e10000}', b'{"x":"\\ud800"}', b'{"x":1,"x":2}', b'{"x":NaN}', b"[]"])
def test_provider_json_failures_are_sanitized_and_not_retried(raw):
    with server(body=raw) as (url, calls):
        with pytest.raises(ModelError, match="invalid_provider_(json|envelope)"):
            request_json("GET", url, timeout=3)
        assert len(calls) == 1


def private_request(**updates):
    return {
        "schema": "vle-http-request/v1",
        "method": "GET",
        "url": "http://127.0.0.1/fixture",
        "body": None,
        "headers": {},
        "seconds": 3,
        "max_bytes": 100,
        "identity_only": True,
        **updates,
    }


@pytest.mark.parametrize(
    "updates",
    [
        {"method": "DELETE"},
        {"url": "file:///tmp/x"},
        {"url": "http://@x/"},
        {"url": "http://x:bad/"},
        {"url": "http://x/a\n"},
        {"seconds": True},
        {"seconds": float("inf")},
        {"max_bytes": True},
        {"identity_only": 1},
        {"body": "@@"},
        {"body": 1},
        {"headers": {"x": "a\nb"}},
        {"headers": {"x": 1}},
        {"schema": "wrong"},
    ],
)
def test_private_request_validation(updates):
    with pytest.raises((ValueError, TypeError)):
        transport.validate_request(private_request(**updates))


@pytest.mark.parametrize("options", [{"max_bytes": True}, {"max_bytes": 0}, {"identity_only": 1}])
def test_client_constructor_rejects_invalid_bounds(options):
    with pytest.raises(ValueError):
        BoundedHTTPClient(**options)


@pytest.mark.parametrize(
    "options",
    [{"timeout": float("nan")}, {"deadline_seconds": 0}, {"delay": True}, {"delay": float("inf")}, {"max_bytes": -1}],
)
def test_import_constructor_rejects_invalid_bounds(options):
    with pytest.raises(ValueError):
        LegalJSONTransport(**options)


def fake_receipt(monkeypatch, *, update=None, error=None, content=SYNTHETIC):
    paths = []

    def run(command, **kwargs):
        assert kwargs["timeout"] > 0 and kwargs["check"] is True
        assert kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL
        source, target, binary = map(Path, command[-3:])
        assert source.stat().st_mode & 0o777 == 0o600
        paths.append(source.parent)
        raw = source.read_bytes()
        result = {
            "request_sha256": digest(raw),
            "status": 200,
            "headers": {},
            "body_bytes": len(content),
            "body_sha256": hashlib.sha256(content).hexdigest(),
        }
        if error:
            result["error"] = error
        result.update(update or {})
        target.write_bytes(canonical_json(result))
        binary.write_bytes(content)

    monkeypatch.setattr(transport.subprocess, "run", run)
    return paths


@pytest.mark.parametrize(
    "change",
    [
        {"request_sha256": "wrong"},
        {"body_sha256": "wrong"},
        {"body_bytes": 0},
        {"status": True},
        {"headers": {"Content-Type": []}},
        {"headers": {"x": "y"}},
    ],
)
def test_corrupt_receipt_never_returns_data_and_cleans_up(monkeypatch, change):
    paths = fake_receipt(monkeypatch, update=change)
    with pytest.raises(HTTPTransportError, match="invalid_receipt"):
        BoundedHTTPClient().request("GET", "http://127.0.0.1/fixture", timeout=3)
    assert len(paths) == 1 and not paths[0].exists()


@pytest.mark.parametrize(
    "code,model,legal",
    [
        ("timeout", "transport_timeout", "request_time_bound"),
        ("response_size_bound", "response_exceeds_1_MiB", "response_size_bound"),
        ("capacity_bound", "runtime_capacity_bound", "transport_capacity_bound"),
        ("encoding", "unexpected_content_encoding", "upstream_transport_or_json_error"),
        ("request_failed", "runtime_unreachable", "upstream_transport_or_json_error"),
    ],
)
def test_wrapper_failure_codes_and_capacity_release(monkeypatch, code, model, legal):
    paths = fake_receipt(monkeypatch, error=code)
    if code == "capacity_bound":
        # Worker receipts cannot invent a parent capacity result.
        monkeypatch.setattr(transport, "_SLOT", type("Busy", (), {"acquire": lambda *_a, **_k: False})())
    with pytest.raises(ModelError, match=model):
        request_json("GET", "http://127.0.0.1/fixture", timeout=3)
    with pytest.raises(ImportTransportError, match=legal):
        LegalJSONTransport().request("GET", "http://127.0.0.1/fixture")
    assert all(not p.exists() for p in paths)


def test_subprocess_is_killed_reaped_and_slots_reusable(monkeypatch):
    original, popen, children = subprocess.run, subprocess.Popen, []

    def record(*args, **kwargs):
        process = popen(*args, **kwargs)
        children.append(process)
        return process

    def delayed(command, **kwargs):
        # Substitute a trusted blocked process; no production test-only hook.
        return original([command[0], "-c", "import time; time.sleep(5)"], **kwargs)

    monkeypatch.setattr(transport.subprocess, "Popen", record)
    monkeypatch.setattr(transport.subprocess, "run", delayed)
    with pytest.raises(HTTPTransportError, match="timeout"):
        BoundedHTTPClient().request("GET", "http://127.0.0.1/fixture", timeout=0.1)
    assert len(children) == 1 and children[0].poll() is not None
    fake_receipt(monkeypatch)
    assert BoundedHTTPClient().request("GET", "http://127.0.0.1/fixture", timeout=3).read() == SYNTHETIC


def test_parent_preflight_limits_before_any_process(monkeypatch):
    def forbidden(*_a, **_kw):
        raise AssertionError("must not spawn")

    monkeypatch.setattr(transport.subprocess, "run", forbidden)
    monkeypatch.setattr(transport, "MAX_REQUEST_BYTES", 2)
    client = BoundedHTTPClient()
    with pytest.raises(HTTPTransportError, match="request_size_bound"):
        client.request("POST", "http://127.0.0.1/fixture", body=b"xxx", timeout=3)
    for body in ([], 1):
        with pytest.raises(ValueError, match="body"):
            client.request("POST", "http://127.0.0.1/fixture", body=body, timeout=3)
    for key in ("retries", "redirect"):
        with pytest.raises(ValueError, match="forbids"):
            client.request("GET", "http://127.0.0.1/fixture", timeout=3, **{key: True})
    monkeypatch.setattr(transport, "MAX_ENVELOPE_BYTES", 2)
    with pytest.raises(HTTPTransportError, match="request_size_bound"):
        client.request("GET", "http://127.0.0.1/fixture", timeout=3)


class Response(io.BytesIO):
    status, headers = 200, {}

    def read(self, *a, **kw):
        return super().read(*a)

    def release_conn(self):
        pass


def test_late_decode_and_late_rate_limit_wakeup_fail_closed(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(legal_transport.time, "monotonic", lambda: clock[0])
    client = LegalJSONTransport(timeout=1)
    monkeypatch.setattr(client.http, "request", lambda *_a, **_kw: Response(SYNTHETIC))

    def parse(raw):
        clock[0] = 2
        return strict_json(raw)

    monkeypatch.setattr(legal_transport, "strict_json", parse)
    with pytest.raises(ImportTransportError, match="request_time_bound"):
        client.request("GET", "http://127.0.0.1/fixture")
    client.delay, client.last_request, client.deadline = 2, 0.5, 3
    monkeypatch.setattr(legal_transport.time, "sleep", lambda _: clock.__setitem__(0, 4))
    with pytest.raises(ImportTransportError, match="request_time_bound"):
        client.request("GET", "http://127.0.0.1/fixture")


def test_model_late_decode_never_yields_success(monkeypatch):
    clock = [0.0]
    fake_receipt(monkeypatch)
    monkeypatch.setattr(model_runtime.time, "monotonic", lambda: clock[0])

    def parse(raw):
        clock[0] = 2
        return strict_json(raw)

    monkeypatch.setattr(model_runtime, "strict_json", parse)
    with pytest.raises(ModelError, match="transport_timeout"):
        request_json("GET", "http://127.0.0.1/fixture", timeout=1)


def test_worker_envelope_is_bound_and_errors_do_not_echo_secrets(tmp_path, monkeypatch):
    source, target, binary = (tmp_path / name for name in ("request.json", "result.json", "body.bin"))
    source.write_bytes(canonical_json(private_request(headers={"Authorization": "synthetic-secret"})))

    def fail(_):
        raise RuntimeError("synthetic-secret")

    monkeypatch.setattr(http_worker, "execute", fail)
    http_worker.process_request(source, target, binary)
    receipt = strict_json(target.read_bytes())
    assert receipt["error"] == "request_failed" and "synthetic-secret" not in target.read_text()
    assert receipt["body_bytes"] == 0 and binary.read_bytes() == b""
    assert receipt["request_sha256"] == digest(source.read_bytes())
    monkeypatch.setattr(http_worker, "MAX_ENVELOPE_BYTES", 2)
    http_worker.process_request(source, target, binary)
    assert strict_json(target.read_bytes())["error"] == "request_failed"


def test_worker_timeout_and_cleanup_in_process(tmp_path, monkeypatch):
    response = Response(SYNTHETIC)
    calls = []

    class Pool:
        def request(self, *a, **kw):
            assert kw["redirect"] is False and kw["retries"] is False
            calls.append("request")
            return response

        def clear(self):
            calls.append("clear")

    monkeypatch.setattr(http_worker.urllib3, "PoolManager", lambda **kw: Pool())
    result, body = http_worker.execute(private_request())
    assert result["status"] == 200 and body == SYNTHETIC
    assert response.closed and calls == ["request", "clear"]
    source, target, binary = (tmp_path / name for name in ("request.json", "result.json", "body.bin"))
    source.write_bytes(canonical_json(private_request()))

    def timeout(_):
        raise urllib3.exceptions.ReadTimeoutError(None, "private", "private")

    monkeypatch.setattr(http_worker, "execute", timeout)
    http_worker.process_request(source, target, binary)
    assert strict_json(target.read_bytes())["error"] == "timeout"


def test_existing_long_batch_budget_remains_supported():
    client = LegalJSONTransport(deadline_seconds=600)
    assert 599 < client.deadline - time.monotonic() <= 600


def test_error_marker_even_if_falsy_cannot_be_a_success(monkeypatch):
    fake_receipt(monkeypatch, update={"error": False})
    with pytest.raises(HTTPTransportError, match="request_failed"):
        BoundedHTTPClient().request("GET", "http://127.0.0.1/fixture", timeout=3)


def test_late_valid_receipt_never_returns_data(monkeypatch):
    fake_receipt(monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(transport.time, "monotonic", lambda: clock[0])

    def late(raw):
        clock[0] = 4
        return strict_json(raw)

    monkeypatch.setattr(transport, "strict_json", late)
    with pytest.raises(HTTPTransportError, match="timeout"):
        BoundedHTTPClient().request("GET", "http://127.0.0.1/fixture", timeout=3)
