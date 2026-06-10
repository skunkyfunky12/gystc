# tests/test_fix_storage_database.py
"""fts_search must never swallow sqlite3.OperationalError silently (finding 28).

OperationalError is not just "bad MATCH query" (already prevented by
_sanitize_fts_query) -- it is also 'database is locked' after busy_timeout and
a missing/corrupt notes_fts table. FTS is the ONLY search path while the model
loads, so a silently-empty result reads as "the brain has no memory of this".
"""
from __future__ import annotations

import sqlite3

from brain_mcp.storage.database import BrainDB


def _make_db(tmp_path):
    db = BrainDB(tmp_path / "t.db")
    db.upsert_note(
        path="a.md", title="Routing", content="routing middleware text",
        content_hash="h1", region_idx=9, tags=[], word_count=3,
        created_at="2026-01-01", modified_at="2026-01-01T00:00:00",
    )
    return db


def test_fts_search_logs_operational_error_to_stderr(tmp_path, capsys):
    db = _make_db(tmp_path)
    db.execute("DROP TABLE notes_fts")  # simulate corrupt/missing FTS index
    assert db.fts_search("routing") == []  # degraded fallback, not a crash
    err = capsys.readouterr().err
    assert "fts_search" in err, "FTS failure must be logged, never silent"
    assert "routing" in err, "log line must carry the query context"
    db.close()


class _LockedOnceConn:
    """Raises 'database is locked' on the first execute, then delegates."""

    def __init__(self, real):
        self._real = real
        self.raised = 0

    def execute(self, *args, **kwargs):
        if self.raised == 0:
            self.raised += 1
            raise sqlite3.OperationalError("database is locked")
        return self._real.execute(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_fts_search_retries_once_when_locked(tmp_path):
    db = _make_db(tmp_path)
    real = db._conn
    db._conn = _LockedOnceConn(real)
    rows = db.fts_search("routing")
    db._conn = real
    assert len(rows) == 1, "a single transient 'database is locked' must be retried"
    db.close()


class _AlwaysLockedConn:
    def __init__(self, real):
        self._real = real

    def execute(self, *args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_fts_search_locked_twice_logs_and_returns_empty(tmp_path, capsys):
    db = _make_db(tmp_path)
    real = db._conn
    db._conn = _AlwaysLockedConn(real)
    assert db.fts_search("routing") == []
    db._conn = real
    assert "locked" in capsys.readouterr().err, "persistent lock must be surfaced"
    db.close()


# -----------------------------------------------------------------------
# Final-verification round: pin the displaced-stamp contract (R2#6) on the
# single-row variant without mocks, plus clear_faiss_idx (detach-then-embed).
# -----------------------------------------------------------------------

def test_set_faiss_idx_returns_displaced_and_clear_detaches(tmp_path):
    db = _make_db(tmp_path)
    nid = db.get_note_by_path("a.md")["id"]
    assert db.set_faiss_idx(nid, 5) is None, "first stamp displaces nothing"
    assert db.set_faiss_idx(nid, 5) is None, "unchanged stamp is not displaced"
    assert db.set_faiss_idx(nid, 9) == 5, "overwriting must report the displaced stamp"
    assert db.clear_faiss_idx(nid) == 9, "clear must report the detached stamp"
    row = db.get_note_by_path("a.md")
    assert row["faiss_idx"] is None and row["embedded_at"] is None
    assert db.clear_faiss_idx(nid) is None, "clearing a NULL row reports nothing"
    db.close()


# -----------------------------------------------------------------------
# Final-verification round (R2#2): the retry backoff must sleep OUTSIDE the
# global DB RLock -- sleeping inside stalls every in-process DB call,
# including the 'instant' brain_status/brain_recent.
# -----------------------------------------------------------------------

def test_fts_retry_backoff_releases_db_lock(tmp_path, monkeypatch):
    import threading

    import brain_mcp.storage.database as dbmod

    db = _make_db(tmp_path)
    db._conn = _LockedOnceConn(db._conn)  # attempt 1 raises 'database is locked'
    probe_results = []

    def probing_sleep(_secs):
        # During the backoff, ANOTHER thread must be able to take the DB lock.
        def probe():
            got = db._lock.acquire(blocking=False)
            if got:
                db._lock.release()
            probe_results.append(got)

        t = threading.Thread(target=probe)
        t.start()
        t.join(timeout=2)

    monkeypatch.setattr(dbmod.time, "sleep", probing_sleep)
    rows = db.fts_search("routing")
    assert probe_results == [True], \
        "the DB RLock must be free during the retry backoff (sleep outside the lock)"
    assert len(rows) == 1, "the retry itself must still succeed"
    db.close()
