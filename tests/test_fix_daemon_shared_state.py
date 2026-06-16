"""Regression: the shared daemon must build BrainState ONCE per process, not per
MCP request.

stateless_http (daemon/server.py) makes FastMCP re-enter brain_lifespan on every
request. The old lifespan built db + model + vectors + writer-election and tore
them down each time -> the model reloaded (~15 s/call), writer election churned,
and the startup reconcile aborted mid-scan with "Cannot operate on a closed
database", so the index never converged (24 % coverage bug, 2026-06-16).

Fix: a process-global singleton built once; the daemon never tears it down per
request (graceful shutdown runs at process exit via atexit), while the stdio /
in-process path keeps its per-lifespan _shutdown (cmd_serve os._exit's right
after, which would skip atexit).
"""
from __future__ import annotations

import asyncio

import brain_mcp.server as server


def test_acquire_shared_state_builds_once_in_daemon(monkeypatch):
    monkeypatch.setattr(server, "_SHARED_STATE", None, raising=False)
    monkeypatch.setenv("GYSTC_NO_PARENT_WATCHDOG", "1")  # daemon mode
    registered = []
    monkeypatch.setattr(server.atexit, "register", lambda *a, **k: registered.append(a))

    build_calls = []
    sentinel = object()

    def fake_build():
        build_calls.append(1)
        return sentinel

    monkeypatch.setattr(server, "_build_brain_state", fake_build, raising=False)

    s1 = server._acquire_shared_state()
    s2 = server._acquire_shared_state()

    assert s1 is sentinel and s2 is sentinel
    assert len(build_calls) == 1, "BrainState must be built once per process, not per request"
    assert len(registered) == 1, "daemon teardown must be wired exactly once via atexit"


def test_daemon_lifespan_reuses_singleton_without_per_request_teardown(monkeypatch):
    monkeypatch.setattr(server, "_SHARED_STATE", None, raising=False)
    monkeypatch.setenv("GYSTC_NO_PARENT_WATCHDOG", "1")  # daemon mode
    monkeypatch.setattr(server.atexit, "register", lambda *a, **k: None)
    sentinel = object()
    monkeypatch.setattr(server, "_build_brain_state", lambda: sentinel, raising=False)
    shutdowns = []
    monkeypatch.setattr(server, "_shutdown", lambda st: shutdowns.append(st))

    async def two_requests():
        async with server.brain_lifespan(server.mcp) as a:
            pass
        async with server.brain_lifespan(server.mcp) as b:
            pass
        return a, b

    a, b = asyncio.run(two_requests())
    assert a is sentinel and b is sentinel, "both requests share the one singleton state"
    assert shutdowns == [], "daemon must NOT _shutdown the shared state on per-request lifespan exit"


def test_stdio_lifespan_tears_down_per_lifespan(monkeypatch):
    monkeypatch.setattr(server, "_SHARED_STATE", None, raising=False)
    monkeypatch.delenv("GYSTC_NO_PARENT_WATCHDOG", raising=False)  # stdio / in-process
    sentinel = object()
    monkeypatch.setattr(server, "_build_brain_state", lambda: sentinel, raising=False)
    shutdowns = []
    monkeypatch.setattr(server, "_shutdown", lambda st: shutdowns.append(st))

    async def one_session():
        async with server.brain_lifespan(server.mcp) as st:
            assert st is sentinel

    asyncio.run(one_session())
    assert shutdowns == [sentinel], "stdio lifespan must tear down on exit (cmd_serve os._exit skips atexit)"
    assert server._SHARED_STATE is None, "stdio teardown resets the singleton"
