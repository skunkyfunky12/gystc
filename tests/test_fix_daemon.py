"""Regression tests for the daemon audit fixes.

Finding 7: the request-body cap was only enforced via the Content-Length header,
so a chunked request (no Content-Length) could stream an unbounded body past the
guard and OOM the shared daemon.

Finding 12: idle-shutdown vs respawn race -- the dying daemon held daemon.lock
through its whole (slow) teardown, the replacement lost the lock race and exited
before registering, and the launcher never retried, so the client got
"daemon unreachable" although everything would have worked a second later.

Follow-up finding 7 (review round 2): WriterLock.acquire() returned a bare bool,
so run_daemon misdiagnosed 'daemon.lock could not be opened' (transient AV/ACL
error) as "A GYSTC daemon is already running" and exited without any retry.
_acquire_daemon_lock must retry open errors with backoff and print the real
cause, while a genuinely held lock still exits immediately.
"""
import os
import time

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from brain_mcp.daemon.registry import DaemonInfo, write_registry
from brain_mcp.daemon.server import build_guard_middleware


def _guarded_client(handler, max_body):
    app = Starlette(routes=[Route("/mcp", handler, methods=["POST", "GET"])])
    app.add_middleware(build_guard_middleware(
        token="good", allowed_origins=["http://127.0.0.1"], max_body=max_body))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://testserver")


# ---------------------------------------------------------------- finding 7

@pytest.mark.asyncio
async def test_chunked_body_over_cap_rejected_413():
    """A chunked request (no Content-Length) must not bypass the body cap."""
    async def ok(request):
        await request.body()
        return PlainTextResponse("ok")

    async def chunks():  # httpx sends async-iterable content as chunked, no Content-Length
        for _ in range(4):
            yield b"x" * 32  # 128 bytes total > max_body=50

    async with _guarded_client(ok, max_body=50) as c:
        r = await c.post("/mcp", headers={"Authorization": "Bearer good"}, content=chunks())
        assert r.status_code == 413


@pytest.mark.asyncio
async def test_chunked_body_under_cap_reaches_app_intact():
    """An under-cap chunked body must still arrive downstream in full."""
    seen = {}

    async def echo(request):
        seen["body"] = await request.body()
        return PlainTextResponse("ok")

    async def chunks():
        yield b"a" * 20
        yield b"b" * 20  # 40 bytes total < max_body=50

    async with _guarded_client(echo, max_body=50) as c:
        r = await c.post("/mcp", headers={"Authorization": "Bearer good"}, content=chunks())
        assert r.status_code == 200
        assert seen["body"] == b"a" * 20 + b"b" * 20


@pytest.mark.asyncio
async def test_bodyless_get_still_passes_guard():
    """Requests without a body (e.g. GET /health) must not break on the no-CL path."""
    async def ok(request):
        return PlainTextResponse("ok")

    async with _guarded_client(ok, max_body=50) as c:
        r = await c.get("/mcp", headers={"Authorization": "Bearer good"})
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_oversized_chunked_request_rejected_before_unauthorized_check_order_kept():
    """Auth still rejects before any body is consumed (no token -> 401, not 413)."""
    async def ok(request):
        return PlainTextResponse("ok")

    async def chunks():
        yield b"x" * 100

    async with _guarded_client(ok, max_body=50) as c:
        r = await c.post("/mcp", content=chunks())
        assert r.status_code == 401


# --------------------------------------------------------------- finding 12

@pytest.mark.asyncio
async def test_daemon_lock_released_before_inner_lifespan_shutdown():
    """daemon.lock must be released the moment teardown begins, BEFORE the slow
    inner lifespan shutdown (watcher joins, FAISS save) runs."""
    from brain_mcp.daemon.server import _ReleaseLockOnShutdown

    events = []

    class FakeLock:
        def release(self):
            events.append("released")

    async def inner(scope, receive, send):
        assert scope["type"] == "lifespan"
        while True:
            msg = await receive()
            if msg["type"] == "lifespan.startup":
                events.append("startup")
                await send({"type": "lifespan.startup.complete"})
            elif msg["type"] == "lifespan.shutdown":
                events.append("inner-shutdown")  # the slow teardown happens here
                await send({"type": "lifespan.shutdown.complete"})
                return

    app = _ReleaseLockOnShutdown(inner, FakeLock())
    incoming = [{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}]
    sent = []

    async def receive():
        return incoming.pop(0)

    async def send(msg):
        sent.append(msg)

    await app({"type": "lifespan"}, receive, send)
    assert events == ["startup", "released", "inner-shutdown"]
    assert sent == [{"type": "lifespan.startup.complete"},
                    {"type": "lifespan.shutdown.complete"}]


@pytest.mark.asyncio
async def test_release_wrapper_passes_http_scope_through():
    from brain_mcp.daemon.server import _ReleaseLockOnShutdown

    released = []

    class FakeLock:
        def release(self):
            released.append(1)

    calls = []

    async def inner(scope, receive, send):
        calls.append(scope["type"])

    app = _ReleaseLockOnShutdown(inner, FakeLock())
    await app({"type": "http"}, None, None)
    assert calls == ["http"] and released == []


def test_ensure_daemon_retries_when_first_spawn_loses_lock_race(tmp_path, monkeypatch):
    """If the spawned daemon exits before registering (lost the daemon.lock race
    against a daemon that is still tearing down), the launcher must re-spawn
    instead of waiting out the window and giving up."""
    from brain_mcp.daemon import launcher

    reg = tmp_path / "daemon.json"
    spawns = []

    class DeadProc:  # first spawn: lost the lock race, exited immediately
        returncode = 0
        def poll(self):
            return 0

    class LiveProc:
        returncode = None
        def poll(self):
            return None

    def fake_spawn():
        spawns.append(time.monotonic())
        if len(spawns) == 1:
            return DeadProc()
        # second attempt: the old daemon released the lock; daemon registers
        write_registry(reg, DaemonInfo(port=45678, token="t", pid=os.getpid()))
        return LiveProc()

    monkeypatch.setattr(launcher, "_spawn_detached", fake_spawn)
    monkeypatch.setattr(launcher, "is_alive",
                        lambda info, timeout=1.0: info is not None and info.port == 45678)

    start = time.monotonic()
    info = launcher.ensure_daemon(tmp_path)
    elapsed = time.monotonic() - start
    assert info is not None and info.port == 45678
    assert len(spawns) >= 2, "launcher must retry after the spawn died unregistered"
    assert elapsed < 10.0, "retry must happen with backoff inside the wait window"


# ------------------------------------------ follow-up finding 7 (review round 2)

class _ScriptedLock:
    """WriterLock stand-in scripted per acquire() call ('ok'|'held'|'openfail').
    Popping from the script also pins the exact number of acquire() calls."""

    def __init__(self, script):
        self.script = list(script)
        self.open_failed = False
        self.open_error = None

    def acquire(self):
        step = self.script.pop(0)
        if step == "openfail":
            self.open_failed, self.open_error = True, OSError("simulated AV scan")
            return False
        self.open_failed, self.open_error = False, None
        return step == "ok"


def test_daemon_lock_open_error_retries_with_backoff_then_succeeds(capsys):
    from brain_mcp.daemon.server import _acquire_daemon_lock

    sleeps = []
    lock = _ScriptedLock(["openfail", "openfail", "ok"])
    assert _acquire_daemon_lock(lock, sleep=sleeps.append) is True
    assert sleeps == [0.5, 1.0], "open errors must be retried with backoff"
    err = capsys.readouterr().err
    assert "simulated AV scan" in err, "the real cause must be printed"
    assert "already running" not in err, "no phantom 'already running' diagnosis"


def test_daemon_lock_held_exits_immediately_without_retry(capsys):
    from brain_mcp.daemon.server import _acquire_daemon_lock

    sleeps = []
    lock = _ScriptedLock(["held"])
    assert _acquire_daemon_lock(lock, sleep=sleeps.append) is False
    assert sleeps == [], "'held by another daemon' must not be retried"
    assert "already running" in capsys.readouterr().err


def test_daemon_lock_persistent_open_error_gives_up_with_real_cause(capsys):
    from brain_mcp.daemon.server import _acquire_daemon_lock

    sleeps = []
    lock = _ScriptedLock(["openfail"] * 5)
    assert _acquire_daemon_lock(lock, sleep=sleeps.append) is False
    assert sleeps == [0.5, 1.0, 2.0, 4.0], \
        "backoff must stay inside the launcher's 15s respawn window"
    err = capsys.readouterr().err
    assert "simulated AV scan" in err
    assert "already running" not in err


def test_daemon_lock_open_error_then_held_reports_running_daemon(capsys):
    from brain_mcp.daemon.server import _acquire_daemon_lock

    sleeps = []
    lock = _ScriptedLock(["openfail", "held"])
    assert _acquire_daemon_lock(lock, sleep=sleeps.append) is False
    assert sleeps == [0.5]
    assert "already running" in capsys.readouterr().err
