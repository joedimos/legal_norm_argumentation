"""One-shot HTTP processes: hard wall-time/byte/capacity bounds, no replay."""

from __future__ import annotations

import base64
import hashlib
import io
import math
import subprocess  # nosec B404
import sys
import tempfile
import time
from pathlib import Path
from threading import BoundedSemaphore
from urllib.parse import urlsplit

from urllib3 import HTTPHeaderDict
from urllib3.util import Timeout

from .audit_bundle import canonical_json, digest
from .json_boundary import strict_json

MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_ENVELOPE_BYTES = 24 * 1024 * 1024
MAX_RESPONSE_BYTES = 40_000_000
MAX_RECEIPT_BYTES = 65536
MAX_CONCURRENT_REQUESTS = 4
_SLOT = BoundedSemaphore(MAX_CONCURRENT_REQUESTS)


class HTTPTransportError(RuntimeError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def seconds(value, *, limit=120):
    if type(value) not in {int, float} or not 0 < value <= limit or not math.isfinite(value):
        raise ValueError(f"HTTP budget must be finite, positive and at most {limit} seconds")
    return value


def validate_request(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "method", "url", "headers", "body", "seconds", "max_bytes", "identity_only"}
        or value["schema"] != "vle-http-request/v1"
        or value["method"] not in ("GET", "POST")
    ):
        raise ValueError("invalid private HTTP request")
    url = value["url"]
    if not isinstance(url, str) or len(url) > 8192 or any(ord(c) < 33 or ord(c) == 127 for c in url):
        raise ValueError("invalid HTTP URL")
    parsed = urlsplit(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError("invalid HTTP URL")
    _ = parsed.port
    headers = value["headers"]
    if (
        not isinstance(headers, dict)
        or len(headers) > 32
        or any(
            not isinstance(k, str)
            or not k
            or len(k) > 256
            or not isinstance(v, str)
            or len(v) > 8192
            or any(ord(c) < 32 or ord(c) == 127 for c in k + v)
            for k, v in headers.items()
        )
    ):
        raise ValueError("invalid HTTP headers")
    seconds(value["seconds"])
    if (
        type(value["max_bytes"]) is not int
        or not 1 <= value["max_bytes"] <= MAX_RESPONSE_BYTES
        or type(value["identity_only"]) is not bool
    ):
        raise ValueError("invalid HTTP response bound")
    body = base64.b64decode(value["body"], validate=True) if isinstance(value["body"], str) else None
    if value["body"] is not None and body is None or body is not None and len(body) > MAX_REQUEST_BYTES:
        raise ValueError("invalid HTTP request body")
    return body


class BufferedResponse(io.BytesIO):
    def __init__(self, status, headers, body):
        super().__init__(body)
        self.status = status
        self.headers = HTTPHeaderDict(headers)

    def read(self, amt=-1, decode_content=None, cache_content=False):
        return super().read(amt)

    def release_conn(self):
        pass  # The network process and socket have already ended.


class BoundedHTTPClient:
    def __init__(self, *, max_bytes=MAX_RESPONSE_BYTES, identity_only=False):
        if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_RESPONSE_BYTES or type(identity_only) is not bool:
            raise ValueError("invalid HTTP client bounds")
        self.max_bytes, self.identity_only = max_bytes, identity_only

    def clear(self):
        pass  # No parent-side live sockets or pools.

    def request(
        self,
        method,
        url,
        *,
        body=None,
        headers=None,
        timeout=None,
        retries=False,
        redirect=False,
        preload_content=False,
    ):
        started = time.monotonic()
        budget = seconds(timeout.total if isinstance(timeout, Timeout) else timeout)
        if retries is not False or redirect is not False:
            raise ValueError("bounded HTTP transport forbids automatic retry and redirect")
        if isinstance(body, str):
            body = body.encode("utf-8")
        if body is not None and not isinstance(body, bytes):
            raise ValueError("HTTP body must be bytes or text")
        if body is not None and len(body) > MAX_REQUEST_BYTES:
            raise HTTPTransportError("request_size_bound")
        request = {
            "schema": "vle-http-request/v1",
            "method": method,
            "url": url,
            "headers": dict(headers or {}),
            "body": base64.b64encode(body).decode("ascii") if body is not None else None,
            "seconds": budget,
            "max_bytes": self.max_bytes,
            "identity_only": self.identity_only,
        }
        validate_request(request)
        encoded = canonical_json(request)
        if len(encoded) > MAX_ENVELOPE_BYTES:
            raise HTTPTransportError("request_size_bound")
        if not _SLOT.acquire(blocking=False):
            raise HTTPTransportError("capacity_bound")
        try:
            with tempfile.TemporaryDirectory(prefix="vle-http-") as directory:
                root = Path(directory)
                source, target, binary = root / "request.json", root / "result.json", root / "body.bin"
                source.write_bytes(encoded)
                source.chmod(0o600)
                remaining = budget - (time.monotonic() - started)
                if remaining <= 0:
                    raise HTTPTransportError("timeout")
                subprocess.run(
                    [sys.executable, "-m", "src.http_worker", str(source), str(target), str(binary)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=remaining,
                    check=True,
                )  # nosec B603
                with target.open("rb") as stream:
                    raw = stream.read(MAX_RECEIPT_BYTES + 1)
                if len(raw) > MAX_RECEIPT_BYTES:
                    raise HTTPTransportError("invalid_receipt")
                result = strict_json(raw)
                if not isinstance(result, dict) or result.get("request_sha256") != digest(encoded):
                    raise HTTPTransportError("invalid_receipt")
                if "error" in result:
                    error = result["error"]
                    raise HTTPTransportError(
                        error if error in ("timeout", "response_size_bound", "encoding") else "request_failed"
                    )
                if (
                    type(result.get("status")) is not int
                    or not 100 <= result["status"] <= 599
                    or not isinstance(result.get("headers"), dict)
                    or set(result["headers"]) - {"Content-Type", "Content-Length", "Content-Encoding", "Retry-After"}
                    or any(not isinstance(v, str) or len(v) > 4096 for v in result["headers"].values())
                    or type(result.get("body_bytes")) is not int
                    or not 0 <= result["body_bytes"] <= self.max_bytes
                ):
                    raise HTTPTransportError("invalid_receipt")
                with binary.open("rb") as stream:
                    content = stream.read(self.max_bytes + 1)
                if (
                    len(content) != result["body_bytes"]
                    or len(content) > self.max_bytes
                    or hashlib.sha256(content).hexdigest() != result.get("body_sha256")
                ):
                    raise HTTPTransportError("invalid_receipt")
                if time.monotonic() - started >= budget:
                    raise HTTPTransportError("timeout")
                return BufferedResponse(result["status"], result["headers"], content)
        except subprocess.TimeoutExpired as exc:
            raise HTTPTransportError("timeout") from exc
        except (OSError, ValueError, TypeError, KeyError, RecursionError, subprocess.SubprocessError) as exc:
            raise HTTPTransportError("request_failed") from exc
        finally:
            _SLOT.release()
