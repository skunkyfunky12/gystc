from __future__ import annotations
import os
import subprocess
import sys
import time
from pathlib import Path
import httpx
from brain_mcp.daemon.registry import DaemonInfo, read_registry
from brain_mcp.storage.file_lock import WriterLock


def daemon_health_url(info: DaemonInfo) -> str:
    return f"http://127.0.0.1:{info.port}/health"


def is_alive(info, timeout: float = 1.0) -> bool:
    # Never send the bearer token to a port we can't first attribute to our own
    # daemon pid. Fail CLOSED if the pid check can't run (don't probe a possibly
    # reused port) -- a bare best-effort pass would defeat this very guard.
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            if not psutil.pid_exists(info.pid):
                return False
        except Exception:
            return False  # can't verify the pid -> do not leak the token
    try:
        r = httpx.get(daemon_health_url(info),
                      headers={"Authorization": f"Bearer {info.token}"}, timeout=timeout)
        if r.status_code != 200:
            return False
        body = r.json()
        return body.get("ok") is True and body.get("pid") == info.pid
    except Exception:
        return False


def wait_until_alive(info: DaemonInfo, deadline_s: float = 12.0) -> bool:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if is_alive(info):
            return True
        time.sleep(0.15)
    return False


def _spawn_detached() -> subprocess.Popen:
    kwargs = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                  stderr=subprocess.DEVNULL, close_fds=True)
    if os.name == "nt":
        kwargs["creationflags"] = (subprocess.DETACHED_PROCESS
                                   | subprocess.CREATE_NEW_PROCESS_GROUP
                                   | subprocess.CREATE_NO_WINDOW)
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen([sys.executable, "-m", "brain_mcp", "daemon"], **kwargs)


def ensure_daemon(data_dir: Path):
    """Return a live DaemonInfo, starting the daemon if needed. None on failure."""
    data_dir = Path(data_dir)
    reg = data_dir / "daemon.json"
    info = read_registry(reg)
    if info is not None and is_alive(info):
        return info
    # Serialize starters so two proxies don't both spawn a daemon.
    start_lock = WriterLock(data_dir / "daemon-start.lock")
    proc = None
    if start_lock.acquire():
        info = read_registry(reg)
        if info is None or not is_alive(info):
            proc = _spawn_detached()
    try:
        end = time.monotonic() + 15.0
        retry_delay = 0.3
        while time.monotonic() < end:
            info = read_registry(reg)
            if info is not None and is_alive(info):
                return info
            # Our spawn can lose the daemon.lock race against a daemon that is
            # still tearing down (idle shutdown) and exit without registering.
            # Retry with backoff inside the window instead of giving up.
            if proc is not None and proc.poll() is not None:
                print(f"launcher: spawned daemon exited before registering "
                      f"(rc={proc.returncode}); retrying in {retry_delay:.1f}s", file=sys.stderr)
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 2.0)
                proc = _spawn_detached()
            time.sleep(0.15)
        info = read_registry(reg)
        if info is not None and is_alive(info):
            return info
        print("launcher: no live daemon after 15s; giving up", file=sys.stderr)
        return None
    finally:
        start_lock.release()
