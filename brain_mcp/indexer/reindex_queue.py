"""Debounced, serial re-index worker.

Decouples the watchdog observer thread from the (expensive) embedding work.
The observer only ``submit``s a path; a single background thread re-indexes
serially and debounces per path, so a burst of file changes -- or many rapid
edits to the same note -- collapses into one re-index instead of a storm of
synchronous embeds contending on the global DB lock.

A per-path ``max_wait`` cap guarantees a continuously-edited path is still
re-indexed eventually (debounce alone would defer it forever). ``stop()`` +
``join()`` let a caller drain the in-flight unit before closing the DB or
exiting the process, so a re-index is never killed mid-write.
"""
from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable


class ReindexWorker:
    def __init__(
        self,
        handler: Callable[[str, str], None],
        debounce: float = 0.5,
        max_wait: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._handler = handler
        self._debounce = debounce
        # Force-claim a path that has been pending this long even if it keeps
        # changing, so an actively-edited note is never starved.
        self._max_wait = max_wait if max_wait is not None else max(debounce * 10, 5.0)
        self._clock = clock
        # path -> (event_type, last_seen, first_seen)
        self._pending: dict[str, tuple[str, float, float]] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        # Allow restart after a previous stop()/join().
        self._stop.clear()
        self._wake.clear()
        self._thread = threading.Thread(
            target=self._run, name="gystc-reindex", daemon=True
        )
        self._thread.start()

    def submit(self, path: str, event_type: str) -> None:
        """Record the latest event for ``path`` and wake the worker."""
        now = self._clock()
        with self._lock:
            prev = self._pending.get(path)
            first_seen = prev[2] if prev is not None else now
            self._pending[path] = (event_type, now, first_seen)
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the worker thread (and its in-flight handler) to finish."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _claim_due(self) -> list[tuple[str, str]]:
        """Remove and return (path, event_type) for paths that are due: quiet
        for >= debounce, OR pending for >= max_wait (starvation guard)."""
        now = self._clock()
        due: list[tuple[str, str]] = []
        with self._lock:
            for path, (event_type, last_seen, first_seen) in list(self._pending.items()):
                if now - last_seen >= self._debounce or now - first_seen >= self._max_wait:
                    due.append((path, event_type))
                    del self._pending[path]
        return due

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=self._debounce)
            self._wake.clear()
            for path, event_type in self._claim_due():
                try:
                    self._handler(path, event_type)
                except Exception as exc:  # one bad note must not kill the worker
                    print(
                        f"reindex worker: handler failed for {path}: {exc}",
                        file=sys.stderr,
                    )
