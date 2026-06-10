# tests/test_fix_storage_filelock.py
"""WriterLock fail-closed + crash-release regression tests (finding 19).

The old behavior FAILED OPEN: an OSError while opening the lock file made the
instance assume the writer role with no lock held -> two concurrent writers
clobbering index.faiss (the exact corruption the lock exists to prevent).
acquire() must fail CLOSED (reader) and be loud about it; the promotion loop
retries anyway. Also verifies the docstring's load-bearing claim that the OS
releases the lock when the holding process dies.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import brain_mcp.storage.file_lock as file_lock_mod
from brain_mcp.storage.file_lock import WriterLock

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_acquire_fails_closed_on_oserror(tmp_path, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise OSError("simulated ACL/AV failure")

    monkeypatch.setattr(file_lock_mod.os, "open", boom)
    lk = WriterLock(tmp_path / "writer.lock")
    assert lk.acquire() is False, "acquire must fail CLOSED (reader) on OSError"
    assert lk._held is False
    assert "writer-lock" in capsys.readouterr().err, "degradation must be loud"
    lk.release()  # must not raise


_CHILD = """
import sys, time
from brain_mcp.storage.file_lock import WriterLock
lk = WriterLock(sys.argv[1])
print("LOCKED" if lk.acquire() else "DENIED", flush=True)
time.sleep(60)
"""


def test_crash_release_lets_next_writer_in(tmp_path):
    """Child process takes the lock, parent is denied; after the child is
    killed (crash), the parent must be able to acquire -- the OS-level release
    guarantee the class docstring relies on."""
    lock_path = tmp_path / "writer.lock"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(lock_path)],
        stdout=subprocess.PIPE, text=True, env=env, cwd=str(REPO_ROOT),
    )
    parent = WriterLock(lock_path)
    try:
        line = proc.stdout.readline().strip()
        assert line == "LOCKED", f"child failed to take the lock: {line!r}"
        assert parent.acquire() is False, "lock must be held cross-process"

        proc.kill()
        proc.wait(timeout=10)

        deadline = time.time() + 5
        got = False
        while time.time() < deadline:
            if parent.acquire():
                got = True
                break
            time.sleep(0.2)
        assert got, "lock was not released after the holder crashed"
    finally:
        parent.release()
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        proc.stdout.close()
