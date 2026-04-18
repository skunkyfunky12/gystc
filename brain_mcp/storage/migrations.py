# brain_mcp/storage/migrations.py
from __future__ import annotations
import sqlite3
from datetime import datetime, timezone

MIGRATIONS: list[str] = [
    # v1: initial schema (schema_version is created by run_migrations, NOT here)
    """
    CREATE TABLE IF NOT EXISTS notes (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        path        TEXT UNIQUE NOT NULL,
        title       TEXT NOT NULL,
        content     TEXT DEFAULT '',
        content_hash TEXT NOT NULL,
        region_idx  INTEGER NOT NULL DEFAULT 9,
        tags        TEXT DEFAULT '[]',
        word_count  INTEGER DEFAULT 0,
        created_at  TEXT,
        modified_at TEXT,
        embedded_at TEXT,
        faiss_idx   INTEGER
    );
    CREATE TABLE IF NOT EXISTS regions (
        idx         INTEGER PRIMARY KEY,
        name        TEXT NOT NULL,
        color       TEXT NOT NULL,
        description TEXT DEFAULT '',
        position    TEXT DEFAULT '[0,0,0]'
    );
    CREATE TABLE IF NOT EXISTS edges (
        source_id   INTEGER REFERENCES notes(id) ON DELETE CASCADE,
        target_id   INTEGER REFERENCES notes(id) ON DELETE CASCADE,
        link_text   TEXT DEFAULT '',
        PRIMARY KEY (source_id, target_id)
    );
    CREATE INDEX IF NOT EXISTS idx_notes_region ON notes(region_idx);
    CREATE INDEX IF NOT EXISTS idx_notes_modified ON notes(modified_at);
    CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
    CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
    """,
]

DEFAULT_REGIONS = [
    (0, "Praefrontaler Cortex", "#3498DB", "Architecture, decisions, planning", "[0.0, 110.0, -145.0]"),
    (1, "Motorischer Cortex", "#E74C3C", "API writes, actions, execution", "[-55.0, 125.0, -70.0]"),
    (2, "Sensorischer Cortex", "#2ECC71", "Data intake, references, input", "[55.0, 125.0, -35.0]"),
    (3, "Hippocampus", "#F39C12", "Memory, sessions, personal notes", "[-90.0, 0.0, 35.0]"),
    (4, "Kleinhirn", "#9B59B6", "Precision algorithms", "[75.0, -40.0, 125.0]"),
    (5, "Nucleus Accumbens", "#1ABC9C", "Subscriptions, pricing, rewards", "[0.0, 25.0, -55.0]"),
    (6, "Broca-Areal", "#E67E22", "AI, prompts, agents, language", "[-70.0, 55.0, -90.0]"),
    (7, "Visueller Cortex", "#8E44AD", "UI, themes, design", "[55.0, 35.0, 125.0]"),
    (8, "Thalamus", "#16A085", "Index, MOC, data relay", "[0.0, 40.0, 0.0]"),
    (9, "Stammhirn", "#95A5A6", "Config, infrastructure", "[0.0, -110.0, 90.0]"),
    (10, "Basalganglien", "#D35400", "Pipelines, ETL, background", "[-35.0, 30.0, -15.0]"),
    (11, "Amygdala", "#C0392B", "Auth, team, social interaction", "[-40.0, -25.0, -55.0]"),
]


def run_migrations(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    current = row[0] if row[0] is not None else 0
    for i, sql in enumerate(MIGRATIONS, start=1):
        if i > current:
            conn.executescript(sql)
            now = datetime.now(timezone.utc).isoformat()
            conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (?, ?)", (i, now))
            conn.commit()
    existing = conn.execute("SELECT COUNT(*) FROM regions").fetchone()[0]
    if existing == 0:
        conn.executemany(
            "INSERT INTO regions (idx, name, color, description, position) VALUES (?, ?, ?, ?, ?)",
            DEFAULT_REGIONS,
        )
        conn.commit()
    try:
        conn.execute("SELECT * FROM notes_fts LIMIT 0")
    except sqlite3.OperationalError:
        conn.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
                title, content,
                content='notes', content_rowid='id',
                tokenize='unicode61'
            );
        """)
        conn.commit()
