"""Phase 4 analyses and tests must not contact networks or live providers."""

import socket

import pytest
import urllib3

from src.http_transport import BoundedHTTPClient


@pytest.fixture(autouse=True)
def phase4_no_network(request, monkeypatch):
    if not any(prefix in request.node.path.name for prefix in ("test_phase4_", "test_phase5_", "test_automatic_imports", "test_release_", "test_pyreason_")):
        return

    def blocked(*args, **kwargs):
        raise AssertionError("Research/import tests must remain offline; use recorded/fake providers")

    monkeypatch.setattr(urllib3.PoolManager, "request", blocked)
    monkeypatch.setattr(BoundedHTTPClient, "request", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
