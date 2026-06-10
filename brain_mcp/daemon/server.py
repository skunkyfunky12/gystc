from __future__ import annotations
import hmac, secrets, socket, sys, time
from typing import Callable
from starlette.datastructures import Headers
from starlette.responses import JSONResponse

MAX_BODY_BYTES = 8 * 1024 * 1024  # /mcp request-body cap (note-content cap is 1 MB)


def _idle_expired(last_activity: float, now: float, idle_timeout: float) -> bool:
    """True if idle longer than idle_timeout. idle_timeout <= 0 disables (never expires)."""
    return idle_timeout > 0 and (now - last_activity) > idle_timeout


def build_guard_middleware(token: str, allowed_origins: list[str],
                           max_body: int = MAX_BODY_BYTES,
                           on_request: Callable[[], None] | None = None):
    # Pure ASGI middleware (not BaseHTTPMiddleware) so requests WITHOUT a
    # Content-Length (e.g. Transfer-Encoding: chunked) can be read under an
    # explicit byte budget and replayed downstream. Trusting the header alone
    # let a token-holder stream an unbounded chunked body past the cap.
    class Guard:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                await self.app(scope, receive, send)
                return
            headers = Headers(scope=scope)
            origin = headers.get("origin")
            if origin is not None and origin not in allowed_origins:
                await JSONResponse({"error": "forbidden origin"}, status_code=403)(scope, receive, send)
                return
            auth = headers.get("authorization", "")
            if not (auth and hmac.compare_digest(auth, f"Bearer {token}")):
                await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
                return
            # Bound the body so a token-holder can't OOM the shared daemon.
            cl = headers.get("content-length")
            if cl is not None:
                try:
                    over = int(cl) > max_body
                except ValueError:
                    await JSONResponse({"error": "bad content-length"}, status_code=400)(scope, receive, send)
                    return
                if over:
                    await JSONResponse({"error": "payload too large"}, status_code=413)(scope, receive, send)
                    return
                if on_request is not None:
                    on_request()  # authorized request -> reset the idle timer
                # The server enforces Content-Length framing, so pass through.
                await self.app(scope, receive, send)
                return
            # No Content-Length: read at most max_body bytes (+1 triggers the
            # reject), 413 on oversize, then replay the buffered body downstream.
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return  # client gone; nothing to answer
                body.extend(message.get("body", b""))
                if len(body) > max_body:
                    await JSONResponse({"error": "payload too large"}, status_code=413)(scope, receive, send)
                    return
                if not message.get("more_body", False):
                    break
            if on_request is not None:
                on_request()  # authorized request -> reset the idle timer
            replayed = False

            async def replay():
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            await self.app(scope, replay, send)
    return Guard


class _ReleaseLockOnShutdown:
    """ASGI lifespan wrapper: release `lock` the moment teardown begins.

    The daemon's slow teardown (watcher/reindexer joins, FAISS save) runs inside
    the inner lifespan shutdown and can take seconds. Releasing daemon.lock
    BEFORE forwarding the shutdown message lets a replacement daemon win the
    lock immediately instead of losing the race against a dying daemon and
    exiting unregistered (idle-shutdown vs respawn race). uvicorn has already
    closed the listener at this point, so two daemons never accept at once.
    """

    def __init__(self, app, lock):
        self.app = app
        self.lock = lock

    async def __call__(self, scope, receive, send):
        if scope["type"] != "lifespan":
            await self.app(scope, receive, send)
            return

        async def receive_release_first():
            message = await receive()
            if message["type"] == "lifespan.shutdown":
                print("GYSTC daemon: teardown begins -> releasing daemon.lock", file=sys.stderr)
                try:
                    self.lock.release()
                except OSError as exc:
                    # OS still releases it at process exit; surface, don't hide.
                    print(f"GYSTC daemon: daemon.lock release failed: {exc}", file=sys.stderr)
            return message

        await self.app(scope, receive_release_first, send)

def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _acquire_daemon_lock(lock, attempts: int = 5, initial_delay: float = 0.5,
                         sleep: Callable[[float], None] = time.sleep) -> bool:
    """Take daemon.lock, telling 'held by another daemon' apart from 'the lock
    file could not be opened' (WriterLock.open_failed).

    A transient open error (AV scan / ACL hiccup) is NOT a running daemon.
    Unlike writer.lock there is no promotion loop retrying for us, and the
    launcher discards the spawned daemon's stderr -- so retry briefly with
    backoff here and print the real cause instead of exiting with a phantom
    "already running" diagnosis. Worst-case retry budget (defaults): 7.5s,
    inside the launcher's 15s respawn window.
    """
    delay = initial_delay
    for attempt in range(1, attempts + 1):
        if lock.acquire():
            return True
        if not lock.open_failed:
            print("A GYSTC daemon is already running; exiting.", file=sys.stderr)
            return False
        if attempt < attempts:
            print(f"GYSTC daemon: daemon.lock open failed ({lock.open_error}); "
                  f"retry {attempt}/{attempts - 1} in {delay:.1f}s", file=sys.stderr)
            sleep(delay)
            delay = min(delay * 2, 4.0)
    print(f"GYSTC daemon: could not open daemon.lock after {attempts} attempts "
          f"({lock.open_error}); exiting.", file=sys.stderr)
    return False

def run_daemon(idle_timeout: float = 1800.0) -> None:
    """Run the GYSTC FastMCP server over streamable-http on 127.0.0.1.

    If idle_timeout > 0, the daemon self-terminates after that many seconds with no
    authorized request; the proxy transparently respawns it on the next call. 0 = never.
    """
    import os, time, threading, uvicorn
    os.environ["GYSTC_NO_PARENT_WATCHDOG"] = "1"  # daemon must NOT die with its launcher
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.routing import Route
    from starlette.responses import JSONResponse as JR
    from brain_mcp.server import mcp
    from brain_mcp.config import load_config
    from brain_mcp.daemon.registry import write_registry, DaemonInfo
    from brain_mcp.storage.file_lock import WriterLock

    data_dir = load_config().data_dir
    lock = WriterLock(data_dir / "daemon.lock")
    if not _acquire_daemon_lock(lock):  # prints the actual failure mode
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))                       # do NOT setsockopt SO_REUSEADDR on Windows
    port = sock.getsockname()[1]
    token = secrets.token_urlsafe(32)
    write_registry(data_dir / "daemon.json", DaemonInfo(port=port, token=token, pid=os.getpid()))

    allowed_hosts = [f"127.0.0.1:{port}", f"localhost:{port}"]
    allowed_origins = [f"http://127.0.0.1:{port}", f"http://localhost:{port}"]

    mcp.settings.json_response = True
    mcp.settings.stateless_http = True
    mcp.settings.streamable_http_path = "/mcp"
    mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )
    app = mcp.streamable_http_app()
    activity = [time.monotonic()]
    app.add_middleware(build_guard_middleware(
        token=token, allowed_origins=allowed_origins,
        on_request=lambda: activity.__setitem__(0, time.monotonic())))

    async def health(_req):
        return JR({"ok": True, "pid": os.getpid()})
    app.router.routes.append(Route("/health", health, methods=["GET"]))

    # Release daemon.lock as soon as teardown starts (not at process exit), so
    # a respawned daemon doesn't lose the lock race while this one drains.
    server = uvicorn.Server(uvicorn.Config(_ReleaseLockOnShutdown(app, lock), log_level="warning"))

    def _idle_watch() -> None:
        interval = min(30.0, max(1.0, idle_timeout / 4))
        while not server.should_exit:
            time.sleep(interval)
            if _idle_expired(activity[0], time.monotonic(), idle_timeout):
                print(f"GYSTC daemon idle for >{idle_timeout:.0f}s -> shutting down.", file=sys.stderr)
                server.should_exit = True  # uvicorn stops; daemon.lock is freed at teardown start (_ReleaseLockOnShutdown)
                return
    if idle_timeout > 0:
        threading.Thread(target=_idle_watch, name="gystc-idle", daemon=True).start()

    globals()["_DAEMON_LOCK"] = lock
    globals()["_DAEMON_SOCK"] = sock                  # keep the bound socket alive
    print(f"GYSTC daemon on http://127.0.0.1:{port}/mcp (pid {os.getpid()}; "
          f"idle_timeout={idle_timeout:.0f}s)", file=sys.stderr)
    server.run(sockets=[sock])
