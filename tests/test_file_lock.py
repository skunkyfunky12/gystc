"""Single-writer file lock.

stdio MCP runs one server per client (CLI + Hermes + Command Center). Only one
of them may own index.faiss / the faiss_idx column, or coexisting servers
clobber each other's vectors. WriterLock elects exactly one writer; the rest
run read-only.
"""
from __future__ import annotations

from brain_mcp.storage.file_lock import WriterLock


def test_first_acquirer_wins_second_is_reader(tmp_path):
    lock_path = tmp_path / "writer.lock"
    a = WriterLock(lock_path)
    b = WriterLock(lock_path)
    try:
        assert a.acquire() is True, "first acquirer should become the writer"
        assert b.acquire() is False, "second acquirer must be denied (a holds it)"
    finally:
        a.release()
        b.release()


def test_lock_is_reacquirable_after_release(tmp_path):
    lock_path = tmp_path / "writer.lock"
    a = WriterLock(lock_path)
    assert a.acquire() is True
    a.release()
    b = WriterLock(lock_path)
    try:
        assert b.acquire() is True, "lock should be free again after release"
    finally:
        b.release()


def test_release_is_idempotent(tmp_path):
    a = WriterLock(tmp_path / "writer.lock")
    a.acquire()
    a.release()
    a.release()  # must not raise


def test_release_without_acquire_is_safe(tmp_path):
    WriterLock(tmp_path / "writer.lock").release()  # must not raise
