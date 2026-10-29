"""Phase 4 analyses and tests must not contact networks or live providers."""

import socket

import pytest
import urllib3


@pytest.fixture(autouse=True)
def phase4_no_network(request, monkeypatch):
    if "test_phase4_" not in request.node.path.name:
        return

    def blocked(*args, **kwargs):
        raise AssertionError("Phase 4 tests must remain offline; use recorded/fake providers")

    monkeypatch.setattr(urllib3.PoolManager, "request", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
