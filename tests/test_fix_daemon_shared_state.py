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
    monkeypatch.setattr(server.mcp.settings, "stateless_http", True, raising=False)
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
    monkeypatch.setattr(server.mcp.settings, "stateless_http", True, raising=False)
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


def test_is_shared_daemon_requires_stateless_http(monkeypatch):
    """Greptile #2: GYSTC_NO_PARENT_WATCHDOG alone must NOT mark a process as the
    shared daemon. `gystc serve --direct` (stdio) can inherit that env var; if it
    then took the daemon branch it would skip _shutdown AND os._exit past atexit
    -> the final FAISS save / DB close / reindex drain are lost. The defining trait
    of the shared daemon is stateless_http (per-request lifespan re-entry)."""
    monkeypatch.setenv("GYSTC_NO_PARENT_WATCHDOG", "1")
    monkeypatch.setattr(server.mcp.settings, "stateless_http", False, raising=False)
    assert server._is_shared_daemon() is False, "env var without stateless_http is NOT the daemon"
    monkeypatch.setattr(server.mcp.settings, "stateless_http", True, raising=False)
    assert server._is_shared_daemon() is True, "env var + stateless_http IS the daemon"


def test_stdio_with_leaked_watchdog_env_still_tears_down(monkeypatch):
    """Greptile #2: a stdio server that inherited GYSTC_NO_PARENT_WATCHDOG must
    still run per-lifespan _shutdown (it is not the stateless daemon)."""
    monkeypatch.setattr(server, "_SHARED_STATE", None, raising=False)
    monkeypatch.setenv("GYSTC_NO_PARENT_WATCHDOG", "1")  # leaked into serve --direct
    monkeypatch.setattr(server.mcp.settings, "stateless_http", False, raising=False)
    monkeypatch.setattr(server.atexit, "register", lambda *a, **k: None)
    sentinel = object()
    monkeypatch.setattr(server, "_build_brain_state", lambda: sentinel, raising=False)
    shutdowns = []
    monkeypatch.setattr(server, "_shutdown", lambda st: shutdowns.append(st))

    async def one_session():
        async with server.brain_lifespan(server.mcp) as st:
            assert st is sentinel

    asyncio.run(one_session())
    assert shutdowns == [sentinel], "stdio must tear down even with the watchdog env var leaked in"


def test_daemon_shutdown_releases_writer_lock_before_daemon_lock():
    """Greptile #1: on daemon shutdown the BrainState teardown (which releases
    writer.lock + saves the index) must run BEFORE daemon.lock is released, so the
    successor daemon that wins daemon.lock also wins writer.lock and comes up as
    writer -- not read-only, which would accept brain_store with persist=False and
    drop the note from the index until a later reconcile."""
    from brain_mcp.daemon.server import _ReleaseLockOnShutdown

    events = []

    class FakeLock:
        def release(self):
            events.append("daemon_lock_released")

    async def app(scope, receive, send):
        message = await receive()
        events.append(("forwarded", message["type"]))

    wrapper = _ReleaseLockOnShutdown(app, FakeLock(),
                                     on_shutdown=lambda: events.append("brain_teardown"))

    async def drive():
        async def receive():
            return {"type": "lifespan.shutdown"}

        async def send(_m):
            pass

        await wrapper({"type": "lifespan"}, receive, send)

    asyncio.run(drive())
    assert events.index("brain_teardown") < events.index("daemon_lock_released"), \
        "writer.lock (brain teardown) must free before daemon.lock"
    assert events.index("daemon_lock_released") < events.index(("forwarded", "lifespan.shutdown")), \
        "daemon.lock frees before the shutdown is forwarded to the app"
