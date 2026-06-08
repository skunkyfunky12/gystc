from __future__ import annotations
import hmac, secrets, socket, sys
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

MAX_BODY_BYTES = 8 * 1024 * 1024  # /mcp request-body cap (note-content cap is 1 MB)


def build_guard_middleware(token: str, allowed_origins: list[str],
                           max_body: int = MAX_BODY_BYTES):
    class Guard(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            origin = request.headers.get("origin")
            if origin is not None and origin not in allowed_origins:
                return JSONResponse({"error": "forbidden origin"}, status_code=403)
            auth = request.headers.get("authorization", "")
            if not (auth and hmac.compare_digest(auth, f"Bearer {token}")):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            # Bound the body so a token-holder can't OOM the shared daemon.
            cl = request.headers.get("content-length")
            if cl is not None:
                try:
                    over = int(cl) > max_body
                except ValueError:
                    return JSONResponse({"error": "bad content-length"}, status_code=400)
                if over:
                    return JSONResponse({"error": "payload too large"}, status_code=413)
            return await call_next(request)
    return Guard

def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def run_daemon() -> None:
    """Run the GYSTC FastMCP server over streamable-http on 127.0.0.1."""
    import os, uvicorn
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
    if not lock.acquire():
        print("A GYSTC daemon is already running; exiting.", file=sys.stderr)
        return

    import socket
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
    app.add_middleware(build_guard_middleware(token=token, allowed_origins=allowed_origins))

    async def health(_req):
        return JR({"ok": True, "pid": os.getpid()})
    app.router.routes.append(Route("/health", health, methods=["GET"]))

    globals()["_DAEMON_LOCK"] = lock
    globals()["_DAEMON_SOCK"] = sock                  # keep the bound socket alive
    print(f"GYSTC daemon on http://127.0.0.1:{port}/mcp (pid {os.getpid()})", file=sys.stderr)
    uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=[sock])
