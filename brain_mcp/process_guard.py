"""Process-lifecycle guard for the stdio MCP server.

stdio MCP is inherently *one server process per client*: this CLI session,
Hermes, and the Command Center each spawn their own ``brain_mcp serve``. The
old approach globally killed every other serve process on startup, which
disconnected those *live* siblings at random ("Connection closed").

Instead, each server watches the client that spawned it (its parent process)
and terminates itself once that client is gone. Orphans clean themselves up;
healthy siblings are never touched.
"""
from __future__ import annotations

import sys
import threading
from collections.abc import Callable

try:  # psutil is a hard dependency, but degrade gracefully if it is missing.
    import psutil
except Exception:  # pragma: no cover - psutil always present in practice
    psutil = None  # type: ignore[assignment]


def get_parent_create_time(ppid: int) -> float | None:
    """Best-effort creation time of ``ppid`` (used to detect PID reuse)."""
    if psutil is None or ppid <= 0:
        return None
    try:
        return psutil.Process(ppid).create_time()
    except Exception:
        return None


def parent_is_alive(ppid: int, create_time: float | None = None) -> bool:
    """Whether process ``ppid`` is still running.

    If ``create_time`` is given, the process must also match it -- so a recycled
    PID belonging to a *different* process counts as "parent gone".

    On genuine uncertainty (psutil missing / access denied) we return ``True``:
    never self-terminate on a false positive.
    """
    if ppid <= 0:
        return False
    if psutil is None:
        return True
    try:
        proc = psutil.Process(ppid)
        if not proc.is_running():
            return False
        if create_time is not None:
            return abs(proc.create_time() - create_time) < 1.0
        return True
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, OSError, psutil.Error):
        return True


class ParentDeathWatchdog:
    """Daemon thread that invokes ``on_parent_death`` once the parent is gone.

    ``is_alive`` is injectable for testing; by default it polls the real parent
    process identity (pid + creation time).
    """

    def __init__(
        self,
        ppid: int,
        on_parent_death: Callable[[], None],
        poll_interval: float = 3.0,
        is_alive: Callable[[], bool] | None = None,
    ) -> None:
        self._ppid = ppid
        self._on_death = on_parent_death
        self._poll = poll_interval
        self._create_time = get_parent_create_time(ppid)
        self._is_alive = is_alive or (
            lambda: parent_is_alive(self._ppid, self._create_time)
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        # Nothing meaningful to watch (e.g. detached process) -> stay running.
        if self._ppid <= 0 or self._stop.is_set():
            return
        self._thread = threading.Thread(
            target=self._run, name="gystc-parent-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self._poll):
            try:
                alive = self._is_alive()
            except Exception as exc:  # never let the watchdog die silently
                print(f"parent-watchdog check failed: {exc}", file=sys.stderr)
                continue
            if not alive:
                self._on_death()
                return
