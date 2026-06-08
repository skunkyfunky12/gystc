"""Tests for the debounced, serial re-index worker.

Before: the watchdog observer thread re-embedded *synchronously* on every file
change, holding the global DB lock. A burst of changes (e.g. graphify writing
many notes, or a vault sync) serialized a storm of embeds on the observer
thread, contending with live tool calls.

After: file events are submitted to a single background worker that (a) runs
re-index serially off the observer thread and (b) debounces per path, so rapid
repeated edits to the same file collapse into one re-index.
"""
from __future__ import annotations

import threading
import time

from brain_mcp.indexer.reindex_queue import ReindexWorker


def _collector():
    calls: list[tuple[str, str]] = []
    lock = threading.Lock()

    def handler(path: str, event_type: str) -> None:
        with lock:
            calls.append((path, event_type))

    return calls, handler


def test_rapid_events_for_same_path_collapse_to_one(tmp_path):
    calls, handler = _collector()
    w = ReindexWorker(handler, debounce=0.05)
    w.start()
    try:
        for _ in range(6):
            w.submit("a.md", "modified")
        time.sleep(0.4)
        assert calls == [("a.md", "modified")], f"expected single coalesced call, got {calls}"
    finally:
        w.stop()


def test_latest_event_type_wins(tmp_path):
    calls, handler = _collector()
    w = ReindexWorker(handler, debounce=0.05)
    w.start()
    try:
        w.submit("a.md", "created")
        w.submit("a.md", "deleted")
        time.sleep(0.4)
        assert calls == [("a.md", "deleted")]
    finally:
        w.stop()


def test_distinct_paths_each_processed(tmp_path):
    calls, handler = _collector()
    w = ReindexWorker(handler, debounce=0.05)
    w.start()
    try:
        w.submit("a.md", "modified")
        w.submit("b.md", "created")
        time.sleep(0.4)
        assert sorted(calls) == [("a.md", "modified"), ("b.md", "created")]
    finally:
        w.stop()


def test_handlers_never_run_concurrently():
    """A single worker => re-index is serial; the global DB lock is never
    contended by two embeds at once."""
    active = 0
    max_active = 0
    lock = threading.Lock()

    def handler(path: str, event_type: str) -> None:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1

    w = ReindexWorker(handler, debounce=0.01)
    w.start()
    try:
        for i in range(8):
            w.submit(f"n{i}.md", "modified")
        time.sleep(0.6)
        assert max_active == 1, f"handlers ran concurrently (max_active={max_active})"
    finally:
        w.stop()


def test_handler_exception_does_not_kill_worker(tmp_path):
    seen: list[str] = []
    lock = threading.Lock()

    def handler(path: str, event_type: str) -> None:
        with lock:
            seen.append(path)
        if path == "boom.md":
            raise RuntimeError("kaboom")

    w = ReindexWorker(handler, debounce=0.03)
    w.start()
    try:
        w.submit("boom.md", "modified")
        time.sleep(0.2)
        w.submit("after.md", "modified")
        time.sleep(0.3)
        assert "boom.md" in seen
        assert "after.md" in seen, "worker died on a handler exception"
    finally:
        w.stop()


def test_stop_then_join_lets_worker_thread_exit(tmp_path):
    calls, handler = _collector()
    w = ReindexWorker(handler, debounce=0.05)
    w.start()
    w.stop()
    w.join(timeout=2.0)
    assert w._thread is None or not w._thread.is_alive()
    # Submitting with the worker stopped + plenty of elapsed time -> still no work.
    w.submit("a.md", "modified")
    time.sleep(0.15)  # > debounce, so only the stopped worker prevents processing
    assert calls == []


def test_join_drains_inflight_handler(tmp_path):
    started = threading.Event()
    finished = threading.Event()

    def slow_handler(path: str, event_type: str) -> None:
        started.set()
        time.sleep(0.3)
        finished.set()

    w = ReindexWorker(slow_handler, debounce=0.01)
    w.start()
    try:
        w.submit("a.md", "modified")
        assert started.wait(1.0), "handler never started"
        # Stop while the handler is mid-flight; join must wait for it to finish
        # (so os._exit/db.close cannot kill it mid-write).
        w.stop()
        w.join(timeout=2.0)
        assert finished.is_set(), "join() returned before the in-flight handler finished"
    finally:
        w.stop()


def test_continuously_edited_path_is_eventually_indexed(tmp_path):
    """Starvation guard: a path that keeps changing faster than the debounce
    must still be re-indexed within max_wait, not deferred forever."""
    calls, handler = _collector()
    w = ReindexWorker(handler, debounce=0.05, max_wait=0.25)
    w.start()
    try:
        t0 = time.monotonic()
        # Hammer the same path faster than debounce for longer than max_wait.
        while time.monotonic() - t0 < 0.5:
            w.submit("hot.md", "modified")
            time.sleep(0.02)
        time.sleep(0.1)
        assert any(p == "hot.md" for p, _ in calls), (
            "continuously-edited path was starved (never indexed)"
        )
    finally:
        w.stop()


def test_worker_is_restartable_after_stop(tmp_path):
    calls, handler = _collector()
    w = ReindexWorker(handler, debounce=0.05)
    w.start()
    w.stop()
    w.join(timeout=2.0)
    # Restarting must produce a live consumer again, not silently drop events.
    w.start()
    try:
        w.submit("a.md", "modified")
        time.sleep(0.3)
        assert calls == [("a.md", "modified")]
    finally:
        w.stop()
