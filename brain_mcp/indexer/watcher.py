from __future__ import annotations

import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable

from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

PENDING_WRITE_TTL = 2.0
PENDING_WRITE_CLEANUP = 5.0


class _Handler(FileSystemEventHandler):
    def __init__(self, on_change: Callable[[str, str], None], pending: dict[str, float], lock: threading.Lock):
        self._on_change = on_change
        self._pending = pending
        self._lock = lock

    def _is_pending(self, path: str) -> bool:
        resolved = str(Path(path).resolve())
        with self._lock:
            ts = self._pending.get(resolved)
            if ts is not None and (time.time() - ts) < PENDING_WRITE_TTL:
                return True
            self._cleanup()
        return False

    def _cleanup(self) -> None:
        now = time.time()
        stale = [k for k, v in self._pending.items() if now - v > PENDING_WRITE_CLEANUP]
        for k in stale:
            del self._pending[k]

    def _dispatch(self, path: str, event_type: str) -> None:
        try:
            self._on_change(path, event_type)
        except Exception as exc:
            print(f"Watcher callback error for {path}: {exc}\n{traceback.format_exc()}", file=sys.stderr)

    def on_created(self, event: FileCreatedEvent) -> None:
        if not event.is_directory and event.src_path.endswith(".md"):
            if not self._is_pending(event.src_path):
                self._dispatch(event.src_path, "created")

    def on_modified(self, event: FileModifiedEvent) -> None:
        if not event.is_directory and event.src_path.endswith(".md"):
            if not self._is_pending(event.src_path):
                self._dispatch(event.src_path, "modified")

    def on_deleted(self, event: FileDeletedEvent) -> None:
        if not event.is_directory and event.src_path.endswith(".md"):
            self._dispatch(event.src_path, "deleted")

    def on_moved(self, event: FileMovedEvent) -> None:
        if event.is_directory:
            return
        if event.src_path.endswith(".md"):
            self._dispatch(event.src_path, "deleted")
        if event.dest_path.endswith(".md"):
            if not self._is_pending(event.dest_path):
                self._dispatch(event.dest_path, "created")


class BrainWatcher:
    def __init__(self, vault_path: Path, on_change: Callable[[str, str], None]):
        self._vault_path = vault_path
        self._lock = threading.Lock()
        self._pending_writes: dict[str, float] = {}
        self._handler = _Handler(on_change, self._pending_writes, self._lock)
        self._observer = Observer()
        self._observer.daemon = True

    @property
    def is_running(self) -> bool:
        return self._observer.is_alive()

    def add_pending_write(self, resolved_path: str) -> None:
        with self._lock:
            self._pending_writes[resolved_path] = time.time()

    def start(self) -> None:
        self._observer.schedule(self._handler, str(self._vault_path), recursive=True)
        self._observer.start()
        print(f"Watcher started on {self._vault_path}", file=sys.stderr)

    def stop(self) -> None:
        try:
            self._observer.stop()
            self._observer.join(timeout=5)
        except Exception as exc:
            print(f"Watcher stop error: {exc}", file=sys.stderr)
        print("Watcher stopped.", file=sys.stderr)
