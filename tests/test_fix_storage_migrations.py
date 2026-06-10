# tests/test_fix_storage_migrations.py
"""Upgrade-path regression tests for run_migrations (finding 16).

Pins three guarantees:
1. An existing v1 database upgrades to the latest version with data intact.
2. The historical crash window (v2 DDL committed, version bump lost) no longer
   bricks startup with 'duplicate column name' -- re-runs are idempotent.
3. A migration that fails mid-way rolls back atomically: no partial DDL, no
   version bump, and a loud stderr line.
"""
from __future__ import annotations

import sqlite3

import pytest

import brain_mcp.storage.migrations as migrations
from brain_mcp.storage.database import BrainDB
from brain_mcp.storage.migrations import MIGRATIONS, run_migrations


def _build_v1_db(path):
    """Create a database exactly as a v1 install would have left it."""
    conn = sqlite3.connect(str(path))
    conn.executescript(MIGRATIONS[0])
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (1, 'test')")
    conn.execute(
        "INSERT INTO notes (path, title, content, content_hash) VALUES ('a.md', 'A', 'body', 'h1')"
    )
    conn.execute(
        "INSERT INTO notes (path, title, content, content_hash) VALUES ('b.md', 'B', 'body2', 'h2')"
    )
    conn.execute("INSERT INTO edges (source_id, target_id, link_text) VALUES (1, 2, 'B')")
    conn.commit()
    conn.close()


def test_upgrade_from_v1_preserves_data(tmp_path):
    p = tmp_path / "old.db"
    _build_v1_db(p)
    db = BrainDB(p)
    assert db.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == len(MIGRATIONS)
    assert db.get_note_by_path("a.md")["title"] == "A"
    edge = db.execute("SELECT edge_type, weight FROM edges").fetchone()
    assert edge["edge_type"] == "backlink"  # v2 column with default applied
    db.close()


def test_rerun_after_crash_window_is_idempotent(tmp_path):
    """Historical crash: v2 ALTERs committed but the version bump was lost.
    A re-run must NOT die with 'duplicate column name' (it permanently bricked
    server startup on the real brain.db)."""
    p = tmp_path / "old.db"
    _build_v1_db(p)
    conn = sqlite3.connect(str(p))
    conn.executescript(MIGRATIONS[1])  # v2 DDL applied, version row never written
    conn.commit()
    conn.close()

    db = BrainDB(p)  # must not raise
    assert db.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == len(MIGRATIONS)
    db.close()


def test_failing_migration_rolls_back_atomically(tmp_path, monkeypatch, capsys):
    p = tmp_path / "new.db"
    bad = MIGRATIONS + [
        "CREATE TABLE half_applied (x INTEGER);\nINSERT INTO does_not_exist VALUES (1);"
    ]
    monkeypatch.setattr(migrations, "MIGRATIONS", bad)
    conn = sqlite3.connect(str(p))
    with pytest.raises(sqlite3.OperationalError):
        run_migrations(conn)

    names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    assert "half_applied" not in names, "partial migration DDL leaked (not transactional)"
    version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    assert version == len(MIGRATIONS), "version bumped despite failed migration"
    assert "migration" in capsys.readouterr().err.lower(), "failure must be loud"
    conn.close()

    # Recovery: with the bad migration removed, the db opens normally again.
    monkeypatch.setattr(migrations, "MIGRATIONS", MIGRATIONS)
    db = BrainDB(p)
    db.close()


def test_fresh_db_reaches_latest_version_with_tombstone_table(tmp_path):
    db = BrainDB(tmp_path / "fresh.db")
    assert db.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == len(MIGRATIONS)
    names = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    assert "archived_note_versions" in names  # v4 tombstone (finding 10)
    db.close()
