"""Cross-process advisory exclusive lock backed by a lock file.

Used to elect a single "writer" GYSTC instance that owns index.faiss and the
faiss_idx column. Other instances run read-only. The OS releases the lock when
the holding process exits (even via os._exit or a crash), so a dead writer
never blocks a future one.

Tradeoff: if the lock file cannot even be opened (ACL/AV/read-only data dir),
acquire() fails CLOSED -- the instance stays a reader and retries via the
promotion loop. A vault that temporarily doesn't index beats two instances
both assuming the writer role and silently clobbering the FAISS index (the
historical dual-writer corruption this lock exists to prevent).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


class WriterLock:
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._fd: int | None = None
        self._held = False

    def acquire(self) -> bool:
        """Try to take the lock. Returns True if this process got it."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fd = os.open(str(self._path), os.O_RDWR | os.O_CREAT, 0o644)
        except OSError as exc:
            # Can't even open the lock file -> fail CLOSED (stay a reader).
            # Failing open here let two instances both believe they were the
            # writer and clobber index.faiss/faiss_idx; silent index corruption
            # is strictly worse than an instance that doesn't index (the
            # promotion loop keeps retrying acquire() every ~20s anyway).
            print(
                f"writer-lock: open failed ({exc}); failing CLOSED -> running "
                "read-only, will retry via promotion loop",
                file=sys.stderr,
            )
            self._fd = None
            self._held = False
            return False

        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._held = True
            return True
        except OSError:
            # Held by another process -> we are a reader.
            os.close(self._fd)
            self._fd = None
            self._held = False
            return False

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        if self._fd is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                try:
                    msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl
                try:
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
