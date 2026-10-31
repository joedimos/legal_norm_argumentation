"""Private one-shot HTTP worker. Output contains no exception text or redirects."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import urllib3

from .audit_bundle import canonical_json, digest
from .http_transport import MAX_ENVELOPE_BYTES, validate_request
from .json_boundary import strict_json

HEADERS = ("Content-Type", "Content-Length", "Content-Encoding", "Retry-After")


def execute(request):
    body = validate_request(request)
    pool = urllib3.PoolManager(num_pools=1, cert_reqs="CERT_REQUIRED")
    response = None
    try:
        response = pool.request(
            request["method"],
            request["url"],
            body=body,
            headers={"Accept-Encoding": "identity", **request["headers"]},
            timeout=urllib3.Timeout(total=request["seconds"]),
            retries=False,
            redirect=False,
            preload_content=False,
        )
        headers = {k: str(response.headers[k]) for k in HEADERS if k in response.headers}
        if any(len(v) > 4096 for v in headers.values()):
            return {"error": "request_failed"}, b""
        status = response.status
        if status != 200:
            return {"status": status, "headers": headers}, b""
        if request["identity_only"] and response.headers.get("Content-Encoding", "identity").lower() not in (
            "",
            "identity",
        ):
            return {"error": "encoding"}, b""
        length = response.headers.get("Content-Length")
        if length is not None and (
            not str(length).isascii() or not str(length).isdigit() or int(length) > request["max_bytes"]
        ):
            return {"error": "response_size_bound"}, b""
        # Native library limits each inactivity wait; the parent also kills the
        # entire process at its aggregate deadline, including DNS and headers.
        raw = response.read(request["max_bytes"] + 1, decode_content=not request["identity_only"])
        if len(raw) > request["max_bytes"]:
            return {"error": "response_size_bound"}, b""
        return {"status": status, "headers": headers}, raw
    finally:
        if response is not None:
            response.close()
            response.release_conn()
        pool.clear()


def process_request(source, target, binary):
    result = {"error": "request_failed"}
    request = None
    content = b""
    try:
        with source.open("rb") as stream:
            raw = stream.read(MAX_ENVELOPE_BYTES + 1)
        if len(raw) > MAX_ENVELOPE_BYTES:
            raise ValueError("private request size bound")
        request = strict_json(raw)
        validate_request(request)
        result, content = execute(request)
    except urllib3.exceptions.TimeoutError:
        result = {"error": "timeout"}
    except Exception:
        result = {"error": "request_failed"}  # Never serialize URLs, bodies, keys or exception text.
    if request is not None:
        result["request_sha256"] = digest(canonical_json(request))
    result.update(body_bytes=len(content), body_sha256=hashlib.sha256(content).hexdigest())
    pending = binary.with_suffix(".pending")
    pending.write_bytes(content)
    pending.replace(binary)
    pending = target.with_suffix(".pending")
    pending.write_text(json.dumps(result, sort_keys=True, allow_nan=False), encoding="utf-8")
    pending.replace(target)


if __name__ == "__main__":
    process_request(*(Path(p) for p in sys.argv[1:4]))
