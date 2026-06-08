"""Process-lifecycle tests (2026-06-08).

Root cause of the random "Connection closed" disconnects: an orphaned server
(client gone) never terminated -- it lingered holding brain.db -- and the old
``_kill_zombie_siblings`` band-aid then globally killed *live* one-server-per-client
siblings (this CLI + Hermes + Command Center each spawn their own stdio server).

The fix replaces the global killer with a parent-death watchdog: each server
watches the client that spawned it and self-terminates when that client is gone.
These tests pin that behavior.
"""
from __future__ import annotations

import os
import threading
import time

import psutil

from brain_mcp.process_guard import ParentDeathWatchdog, parent_is_alive


def _an_unused_pid() -> int:
    """Return a PID that is not currently running."""
    for candidate in range(999999, 900000, -1):
        if not psutil.pid_exists(candidate):
            return candidate
    raise RuntimeError("could not find an unused pid")


def test_parent_is_alive_true_for_running_process():
    # Our own process is definitely running.
    assert parent_is_alive(os.getpid()) is True


def test_parent_is_alive_false_for_nonexistent_pid():
    assert parent_is_alive(_an_unused_pid()) is False


def test_parent_is_alive_false_for_invalid_pid():
    assert parent_is_alive(0) is False
    assert parent_is_alive(-1) is False


def test_parent_is_alive_detects_pid_reuse_via_create_time():
    # Same PID but a create_time that does not match -> treated as a DIFFERENT
    # (reused) process, so the original parent is considered gone.
    real_ct = psutil.Process(os.getpid()).create_time()
    assert parent_is_alive(os.getpid(), create_time=real_ct) is True
    assert parent_is_alive(os.getpid(), create_time=real_ct + 10_000) is False


def test_watchdog_fires_callback_when_parent_dies():
    alive = threading.Event()
    alive.set()
    fired = threading.Event()

    wd = ParentDeathWatchdog(
        ppid=os.getpid(),
        on_parent_death=fired.set,
        poll_interval=0.02,
        is_alive=alive.is_set,
    )
    wd.start()
    try:
        # While "parent" is alive, the callback must NOT fire.
        assert not fired.wait(0.15)
        # Parent dies -> watchdog must fire promptly.
        alive.clear()
        assert fired.wait(1.0), "watchdog did not fire after parent death"
    finally:
        wd.stop()


def test_watchdog_stop_prevents_firing():
    fired = threading.Event()
    wd = ParentDeathWatchdog(
        ppid=os.getpid(),
        on_parent_death=fired.set,
        poll_interval=0.02,
        is_alive=lambda: False,  # parent already "dead"
    )
    wd.stop()  # stop before start -> must never fire
    wd.start()
    assert not fired.wait(0.2)


def test_watchdog_noop_for_invalid_parent():
    fired = threading.Event()
    wd = ParentDeathWatchdog(ppid=0, on_parent_death=fired.set, poll_interval=0.02)
    wd.start()
    assert not fired.wait(0.15), "must not watch an invalid parent pid"
    wd.stop()


def test_kill_zombie_siblings_is_gone():
    """Regression: the global sibling-killer must NOT exist -- it disconnected
    live one-server-per-client siblings. Do not reintroduce it."""
    import brain_mcp.server as server

    assert not hasattr(server, "_kill_zombie_siblings"), (
        "_kill_zombie_siblings was removed because it killed live sibling "
        "servers belonging to other clients; use ParentDeathWatchdog instead."
    )
