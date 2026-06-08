from __future__ import annotations
import os, subprocess, sys, time
from pathlib import Path
import httpx
from brain_mcp.daemon.registry import DaemonInfo, read_registry
from brain_mcp.storage.file_lock import WriterLock


def daemon_health_url(info: DaemonInfo) -> str:
    return f"http://127.0.0.1:{info.port}/health"


def is_alive(info: DaemonInfo, timeout: float = 1.0) -> bool:
    try:
        # The daemon guards every route (including /health) with a Bearer token,
        # so a discovery probe must authenticate with the token from the registry
        # — otherwise the daemon answers 401 and looks dead.
        r = httpx.get(
            daemon_health_url(info),
            timeout=timeout,
            headers={"Authorization": f"Bearer {info.token}"},
        )
        return r.status_code == 200 and r.json().get("ok") is True
    except Exception:
        return False


def wait_until_alive(info: DaemonInfo, deadline_s: float = 12.0) -> bool:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if is_alive(info):
            return True
        time.sleep(0.15)
    return False


def _spawn_detached() -> None:
    kwargs = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                  stderr=subprocess.DEVNULL, close_fds=True)
    if os.name == "nt":
        kwargs["creationflags"] = (subprocess.DETACHED_PROCESS
                                   | subprocess.CREATE_NEW_PROCESS_GROUP
                                   | subprocess.CREATE_NO_WINDOW)
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen([sys.executable, "-m", "brain_mcp", "daemon"], **kwargs)


def ensure_daemon(data_dir: Path):
    """Return a live DaemonInfo, starting the daemon if needed. None on failure."""
    data_dir = Path(data_dir)
    reg = data_dir / "daemon.json"
    info = read_registry(reg)
    if info is not None and is_alive(info):
        return info
    # Serialize starters so two proxies don't both spawn a daemon.
    start_lock = WriterLock(data_dir / "daemon-start.lock")
    spawned = False
    if start_lock.acquire():
        info = read_registry(reg)
        if info is None or not is_alive(info):
            _spawn_detached()
            spawned = True
    try:
        end = time.monotonic() + 15.0
        while time.monotonic() < end:
            info = read_registry(reg)
            if info is not None and is_alive(info):
                return info
            time.sleep(0.15)
        return read_registry(reg)
    finally:
        start_lock.release()
