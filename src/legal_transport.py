"""Bounded read-only legal-data transport; retry waits belong to the durable worker."""

from __future__ import annotations

import math
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import urllib3
from urllib3.util import Timeout

from .audit_bundle import canonical_json
from .http_transport import BoundedHTTPClient, HTTPTransportError, seconds
from .json_boundary import strict_json


class ImportTransportError(RuntimeError):
    def __init__(self, code, *, retry_after=60):
        super().__init__(code)
        self.retry_after = retry_after


def retry_delay(value, *, now=None):
    now = now or datetime.now(UTC)
    try:
        seconds = int(value) if str(value).isdigit() else int((parsedate_to_datetime(value) - now).total_seconds())
        return max(1, min(seconds, 31536000))
    except (ValueError, TypeError, OverflowError):
        return 60


class LegalJSONTransport:
    def __init__(self, *, timeout=12, max_bytes=8_000_000, delay=0, deadline_seconds=120):
        seconds(timeout)
        seconds(deadline_seconds, limit=600)
        if type(delay) not in (int, float) or not 0 <= delay <= 60 or not math.isfinite(delay):
            raise ValueError("import delay must be finite and between 0 and 60 seconds")
        self.http = BoundedHTTPClient(max_bytes=max_bytes)
        self.timeout, self.max_bytes, self.delay = timeout, max_bytes, delay
        self.deadline = time.monotonic() + deadline_seconds
        self.last_request = None

    def request(self, method, url, *, headers=None, body=None):
        # URLs are constructed/allowlisted by each connector. Never follow redirects
        # with credentials, fetch binary download_url fields or request paid PACER data.
        remaining = self.deadline - time.monotonic()
        wait = max(0, self.delay - (time.monotonic() - self.last_request)) if self.last_request is not None else 0
        if remaining <= wait:
            raise ImportTransportError("request_time_bound")
        if wait:
            time.sleep(wait)
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ImportTransportError("request_time_bound")
        self.last_request = time.monotonic()
        request_deadline = min(self.deadline, self.last_request + self.timeout)
        response = None
        try:
            response = self.http.request(
                method,
                url,
                body=body,
                headers={"Accept": "application/json", "User-Agent": "VLE-LEGATO-IP-import/0.8", **(headers or {})},
                timeout=Timeout(total=min(self.timeout, remaining)),
                retries=False,
                redirect=False,
                preload_content=False,
            )
            if response.status != 200:
                raise ImportTransportError(
                    f"upstream_http_{response.status}",
                    retry_after=retry_delay(response.headers.get("Retry-After")),
                )
            length = response.headers.get("Content-Length")
            if length and int(length) > self.max_bytes:
                raise ImportTransportError("response_size_bound")
            raw = response.read(self.max_bytes + 1)
            if len(raw) > self.max_bytes:
                raise ImportTransportError("response_size_bound")
            if time.monotonic() >= request_deadline:
                raise ImportTransportError("request_time_bound")
            result = strict_json(raw)
            if not isinstance(result, dict):
                raise ImportTransportError("invalid_response_shape")
            if time.monotonic() >= request_deadline:
                raise ImportTransportError("request_time_bound")
            return result
        except ImportTransportError:
            raise
        except HTTPTransportError as exc:
            raise ImportTransportError(
                {
                    "timeout": "request_time_bound",
                    "response_size_bound": "response_size_bound",
                    "capacity_bound": "transport_capacity_bound",
                }.get(exc.code, "upstream_transport_or_json_error")
            ) from exc
        except (urllib3.exceptions.HTTPError, OSError, ValueError, TypeError, RecursionError) as exc:
            raise ImportTransportError("upstream_transport_or_json_error") from exc
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    def post_json(self, url, body, headers):
        return self.request(
            "POST", url, body=canonical_json(body), headers={"Content-Type": "application/json", **headers}
        )
