# Brain MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a stdio MCP server that gives Claude 6 tools to semantically search, store, and navigate a vault of markdown files organized into 12 brain regions.

**Architecture:** Single-process Python MCP server using FastMCP. SQLite + FTS5 for metadata, FAISS IndexFlatIP for vector search, watchdog for live file watching. Sentence-transformers for embeddings (lazy-loaded). All state in `~/.neural-brain/`.

**Tech Stack:** Python 3.11+, mcp SDK (FastMCP), sentence-transformers, faiss-cpu, watchdog, numpy, SQLite3

**Spec:** `docs/superpowers/specs/2026-04-18-brain-mcp-server-design.md`

---

## File Structure

```
neural-brain/
├── brain_mcp/
│   ├── __init__.py               # Package version
│   ├── __main__.py               # CLI entry: serve | index
│   ├── server.py                 # FastMCP server, lifespan, tool/resource registration
│   ├── config.py                 # Config loading, env var overrides, validation
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── recent.py             # brain_recent
│   │   ├── regions.py            # brain_regions
│   │   ├── retrieve.py           # brain_retrieve
│   │   ├── store.py              # brain_store
│   │   ├── related.py            # brain_related
│   │   └── context.py            # brain_context
│   ├── indexer/
│   │   ├── __init__.py
│   │   ├── scanner.py            # Vault scanning, hash tracking, backlinks, tags
│   │   ├── embedder.py           # EmbeddingBackend protocol + SentenceTransformerBackend
│   │   ├── vector_store.py       # FAISS IndexFlatIP with RLock
│   │   └── watcher.py            # watchdog observer, self-write detection
│   └── storage/
│       ├── __init__.py
│       ├── database.py           # SQLite operations, FTS5
│       └── migrations.py         # Schema versioning
├── tests/
│   ├── conftest.py               # Fixtures: tmp vault, test DB, MockEmbedder
│   ├── test_config.py
│   ├── test_database.py
│   ├── test_scanner.py
│   ├── test_embedder.py
│   ├── test_vector_store.py
│   ├── test_watcher.py
│   ├── test_tool_recent.py
│   ├── test_tool_regions.py
│   ├── test_tool_retrieve.py
│   ├── test_tool_store.py
│   ├── test_tool_related.py
│   ├── test_tool_context.py
│   ├── test_security.py
│   └── test_integration.py
└── pyproject.toml
```

---

### Task 1: Project Scaffold + Config

**Files:**
- Create: `brain_mcp/__init__.py`
- Create: `brain_mcp/config.py`
- Create: `pyproject.toml`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import json
import os
from pathlib import Path
from brain_mcp.config import BrainConfig, load_config

def test_load_config_from_file(tmp_path):
    config_dir = tmp_path / ".neural-brain"
    config_dir.mkdir()
    config_file = config_dir / "config.json"
    config_file.write_text(json.dumps({
        "vault_path": "/my/vault",
        "model_name": "test-model",
    }))
    cfg = load_config(config_dir=config_dir)
    assert cfg.vault_path == Path("/my/vault")
    assert cfg.model_name == "test-model"
    assert cfg.embedding_backend == "sentence-transformers"

def test_env_var_overrides_config(tmp_path, monkeypatch):
    config_dir = tmp_path / ".neural-brain"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(json.dumps({"vault_path": "/from/file"}))
    monkeypatch.setenv("BRAIN_VAULT_PATH", "/from/env")
    cfg = load_config(config_dir=config_dir)
    assert cfg.vault_path == Path("/from/env")

def test_default_config_when_no_file(tmp_path):
    config_dir = tmp_path / ".neural-brain"
    config_dir.mkdir()
    cfg = load_config(config_dir=config_dir)
    assert cfg.vault_path is None
    assert cfg.model_name == "paraphrase-multilingual-MiniLM-L12-v2"
    assert cfg.data_dir == config_dir

def test_data_dir_env_override(tmp_path, monkeypatch):
    alt_dir = tmp_path / "alt"
    alt_dir.mkdir()
    monkeypatch.setenv("BRAIN_DATA_DIR", str(alt_dir))
    cfg = load_config(config_dir=tmp_path / "ignored")
    assert cfg.data_dir == alt_dir

def test_folder_to_region_defaults(tmp_path):
    config_dir = tmp_path / ".neural-brain"
    config_dir.mkdir()
    cfg = load_config(config_dir=config_dir)
    assert cfg.folder_to_region == {}

def test_folder_to_region_from_config(tmp_path):
    config_dir = tmp_path / ".neural-brain"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(json.dumps({
        "folder_to_region": {"Projects": 0, "Notes": 3}
    }))
    cfg = load_config(config_dir=config_dir)
    assert cfg.folder_to_region == {"Projects": 0, "Notes": 3}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'brain_mcp'`

- [ ] **Step 3: Create pyproject.toml**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "neural-brain"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "mcp>=1.0",
    "sentence-transformers>=2.2",
    "faiss-cpu>=1.7.4",
    "watchdog>=3.0,<5.0",
    "numpy>=1.24,<2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
]

[tool.setuptools.packages.find]
include = ["brain_mcp*"]
```

- [ ] **Step 4: Create brain_mcp package**

```python
# brain_mcp/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 5: Implement config.py**

```python
# brain_mcp/config.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DATA_DIR = Path.home() / ".neural-brain"

@dataclass
class BrainConfig:
    vault_path: Path | None = None
    model_name: str = DEFAULT_MODEL
    embedding_backend: str = "sentence-transformers"
    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    auto_index: bool = True
    index_on_startup: bool = True
    folder_to_region: dict[str, int] = field(default_factory=dict)
    log_level: str = "INFO"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "brain.db"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.faiss"

    @property
    def config_path(self) -> Path:
        return self.data_dir / "config.json"


def load_config(config_dir: Path | None = None) -> BrainConfig:
    data_dir_env = os.environ.get("BRAIN_DATA_DIR")
    if data_dir_env:
        data_dir = Path(data_dir_env)
    elif config_dir:
        data_dir = config_dir
    else:
        data_dir = DEFAULT_DATA_DIR

    data_dir.mkdir(parents=True, exist_ok=True)
    config_file = data_dir / "config.json"

    file_data: dict = {}
    if config_file.exists():
        try:
            file_data = json.loads(config_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    vault_path_str = os.environ.get("BRAIN_VAULT_PATH") or file_data.get("vault_path")
    vault_path = Path(vault_path_str) if vault_path_str else None

    return BrainConfig(
        vault_path=vault_path,
        model_name=os.environ.get("BRAIN_MODEL_NAME") or file_data.get("model_name", DEFAULT_MODEL),
        embedding_backend=file_data.get("embedding_backend", "sentence-transformers"),
        data_dir=data_dir,
        auto_index=file_data.get("auto_index", True),
        index_on_startup=file_data.get("index_on_startup", True),
        folder_to_region=file_data.get("folder_to_region", {}),
        log_level=os.environ.get("BRAIN_LOG_LEVEL") or file_data.get("log_level", "INFO"),
    )
```

- [ ] **Step 6: Create conftest.py with shared fixtures**

```python
# tests/conftest.py
import json
from pathlib import Path
import numpy as np
import pytest

@pytest.fixture
def tmp_vault(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note1.md").write_text("# Hello\nThis is about routing and maps.\n[[note2]]\n#brain/hippocampus", encoding="utf-8")
    (vault / "note2.md").write_text("# World\nCSP middleware handles security headers.\n[[note1]]\n#brain/praefrontaler-cortex", encoding="utf-8")
    sub = vault / "Projects"
    sub.mkdir()
    (sub / "project1.md").write_text("# Project\nA project about data pipelines.\n[[note1]]", encoding="utf-8")
    return vault

@pytest.fixture
def tmp_config_dir(tmp_path, tmp_vault):
    config_dir = tmp_path / ".neural-brain"
    config_dir.mkdir()
    config = {"vault_path": str(tmp_vault)}
    (config_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return config_dir

class MockEmbedder:
    """Deterministic embedder for testing. Maps text hash to a fixed 384-dim vector."""

    @property
    def dimension(self) -> int:
        return 384

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            rng = np.random.RandomState(hash(text) % (2**31))
            vec = rng.randn(384).astype(np.float32)
            vec /= np.linalg.norm(vec)
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

@pytest.fixture
def mock_embedder():
    return MockEmbedder()
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && pip install -e ".[dev]" && python -m pytest tests/test_config.py -v`
Expected: 6 PASSED

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml brain_mcp/__init__.py brain_mcp/config.py tests/conftest.py tests/test_config.py
git commit -m "feat(mcp): project scaffold with config loading and test fixtures"
```

---

### Task 2: SQLite Database + Migrations

**Files:**
- Create: `brain_mcp/storage/__init__.py`
- Create: `brain_mcp/storage/database.py`
- Create: `brain_mcp/storage/migrations.py`
- Create: `tests/test_database.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_database.py
import json
from brain_mcp.storage.database import BrainDB

def test_create_tables(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    tables = db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    names = [r[0] for r in tables]
    assert "notes" in names
    assert "regions" in names
    assert "edges" in names
    assert "schema_version" in names
    db.close()

def test_default_regions_seeded(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    rows = db.execute("SELECT idx, name, color FROM regions ORDER BY idx").fetchall()
    assert len(rows) == 12
    assert rows[0][1] == "Praefrontaler Cortex"
    assert rows[0][2] == "#3498DB"
    assert rows[11][1] == "Amygdala"
    db.close()

def test_upsert_note(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    db.upsert_note(
        path="test/note.md",
        title="Test Note",
        content="Some content here",
        content_hash="abc123",
        region_idx=3,
        tags=["#brain/hippocampus"],
        word_count=3,
        created_at="2026-01-01",
        modified_at="2026-01-02",
    )
    row = db.get_note_by_path("test/note.md")
    assert row is not None
    assert row["title"] == "Test Note"
    assert row["region_idx"] == 3
    assert json.loads(row["tags"]) == ["#brain/hippocampus"]
    db.close()

def test_upsert_note_updates_existing(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    db.upsert_note(path="a.md", title="V1", content="old", content_hash="h1",
                   region_idx=0, tags=[], word_count=1, created_at="2026-01-01", modified_at="2026-01-01")
    db.upsert_note(path="a.md", title="V2", content="new", content_hash="h2",
                   region_idx=1, tags=["#x"], word_count=2, created_at="2026-01-01", modified_at="2026-01-02")
    row = db.get_note_by_path("a.md")
    assert row["title"] == "V2"
    assert row["content_hash"] == "h2"
    assert row["region_idx"] == 1
    db.close()

def test_get_note_by_title(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    db.upsert_note(path="x.md", title="Hello World", content="c", content_hash="h",
                   region_idx=0, tags=[], word_count=1, created_at="2026-01-01", modified_at="2026-01-01")
    row = db.get_note_by_title("Hello World")
    assert row is not None
    assert row["path"] == "x.md"
    db.close()

def test_upsert_edges(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    db.upsert_note(path="a.md", title="A", content="c", content_hash="h1",
                   region_idx=0, tags=[], word_count=1, created_at="2026-01-01", modified_at="2026-01-01")
    db.upsert_note(path="b.md", title="B", content="c", content_hash="h2",
                   region_idx=1, tags=[], word_count=1, created_at="2026-01-01", modified_at="2026-01-01")
    a = db.get_note_by_path("a.md")
    b = db.get_note_by_path("b.md")
    db.upsert_edge(a["id"], b["id"], link_text="B")
    edges = db.get_edges_for_note(a["id"])
    assert len(edges) == 1
    assert edges[0]["target_id"] == b["id"]
    db.close()

def test_fts_search(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    db.upsert_note(path="a.md", title="Routing", content="This note explains Express routing middleware",
                   content_hash="h1", region_idx=3, tags=[], word_count=6, created_at="2026-01-01", modified_at="2026-01-01")
    db.upsert_note(path="b.md", title="Auth", content="Authentication uses JWT tokens",
                   content_hash="h2", region_idx=11, tags=[], word_count=5, created_at="2026-01-01", modified_at="2026-01-01")
    results = db.fts_search("routing middleware")
    assert len(results) >= 1
    assert results[0]["path"] == "a.md"
    db.close()

def test_recent_notes(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    db.upsert_note(path="old.md", title="Old", content="c", content_hash="h1",
                   region_idx=0, tags=[], word_count=1, created_at="2025-01-01", modified_at="2025-01-01")
    db.upsert_note(path="new.md", title="New", content="c", content_hash="h2",
                   region_idx=0, tags=[], word_count=1, created_at="2026-04-18", modified_at="2026-04-18")
    results = db.get_recent_notes(days=30, limit=10)
    assert results[0]["path"] == "new.md"
    db.close()

def test_region_note_counts(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    db.upsert_note(path="a.md", title="A", content="c", content_hash="h1",
                   region_idx=3, tags=[], word_count=1, created_at="2026-01-01", modified_at="2026-01-01")
    db.upsert_note(path="b.md", title="B", content="c", content_hash="h2",
                   region_idx=3, tags=[], word_count=1, created_at="2026-01-01", modified_at="2026-01-01")
    counts = db.get_region_note_counts()
    assert counts.get(3, 0) == 2
    assert counts.get(0, 0) == 0
    db.close()

def test_update_region(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    db.update_region(0, description="My planning region", color="#FF0000")
    row = db.execute("SELECT description, color FROM regions WHERE idx=0").fetchone()
    assert row[0] == "My planning region"
    assert row[1] == "#FF0000"
    db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_database.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'brain_mcp.storage'`

- [ ] **Step 3: Implement migrations.py**

```python
# brain_mcp/storage/__init__.py
```

```python
# brain_mcp/storage/migrations.py
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

MIGRATIONS: list[str] = [
    # v1: initial schema
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

    CREATE TABLE IF NOT EXISTS schema_version (
        version     INTEGER PRIMARY KEY,
        applied_at  TEXT NOT NULL
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
                title,
                content,
                content='notes',
                content_rowid='id',
                tokenize='unicode61'
            );
        """)
        conn.commit()
```

- [ ] **Step 4: Implement database.py**

```python
# brain_mcp/storage/database.py
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from brain_mcp.storage.migrations import run_migrations


class BrainDB:
    def __init__(self, db_path: Path | str):
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        run_migrations(self._conn)

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    def close(self) -> None:
        self._conn.close()

    def upsert_note(
        self,
        path: str,
        title: str,
        content: str,
        content_hash: str,
        region_idx: int,
        tags: list[str],
        word_count: int,
        created_at: str,
        modified_at: str,
        faiss_idx: int | None = None,
    ) -> int:
        tags_json = json.dumps(tags)
        self._conn.execute(
            """INSERT INTO notes (path, title, content, content_hash, region_idx, tags, word_count, created_at, modified_at, faiss_idx)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(path) DO UPDATE SET
                 title=excluded.title, content=excluded.content, content_hash=excluded.content_hash,
                 region_idx=excluded.region_idx, tags=excluded.tags, word_count=excluded.word_count,
                 modified_at=excluded.modified_at, faiss_idx=COALESCE(excluded.faiss_idx, faiss_idx)
            """,
            (path, title, content, content_hash, region_idx, tags_json, word_count, created_at, modified_at, faiss_idx),
        )
        self._conn.commit()
        row = self._conn.execute("SELECT id FROM notes WHERE path=?", (path,)).fetchone()
        self._sync_fts(row["id"], title, content)
        return row["id"]

    def _sync_fts(self, note_id: int, title: str, content: str) -> None:
        self._conn.execute("INSERT OR REPLACE INTO notes_fts(rowid, title, content) VALUES (?, ?, ?)", (note_id, title, content))
        self._conn.commit()

    def set_faiss_idx(self, note_id: int, faiss_idx: int) -> None:
        self._conn.execute("UPDATE notes SET faiss_idx=?, embedded_at=? WHERE id=?",
                           (faiss_idx, datetime.now(timezone.utc).isoformat(), note_id))
        self._conn.commit()

    def get_note_by_path(self, path: str) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM notes WHERE path=?", (path,)).fetchone()

    def get_note_by_title(self, title: str) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM notes WHERE title=?", (title,)).fetchone()

    def get_note_by_id(self, note_id: int) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM notes WHERE id=?", (note_id,)).fetchone()

    def get_all_notes(self) -> list[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM notes ORDER BY id").fetchall()

    def get_notes_by_faiss_indices(self, indices: list[int]) -> list[sqlite3.Row]:
        if not indices:
            return []
        placeholders = ",".join("?" for _ in indices)
        return self._conn.execute(
            f"SELECT * FROM notes WHERE faiss_idx IN ({placeholders})", tuple(indices)
        ).fetchall()

    def delete_note(self, path: str) -> None:
        row = self.get_note_by_path(path)
        if row:
            self._conn.execute("DELETE FROM notes_fts WHERE rowid=?", (row["id"],))
            self._conn.execute("DELETE FROM edges WHERE source_id=? OR target_id=?", (row["id"], row["id"]))
            self._conn.execute("DELETE FROM notes WHERE id=?", (row["id"],))
            self._conn.commit()

    def upsert_edge(self, source_id: int, target_id: int, link_text: str = "") -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO edges (source_id, target_id, link_text) VALUES (?, ?, ?)",
            (source_id, target_id, link_text),
        )
        self._conn.commit()

    def get_edges_for_note(self, note_id: int) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM edges WHERE source_id=? OR target_id=?", (note_id, note_id)
        ).fetchall()

    def get_neighbor_ids(self, note_id: int, depth: int = 1) -> set[int]:
        visited: set[int] = set()
        frontier = {note_id}
        for _ in range(depth):
            if not frontier:
                break
            placeholders = ",".join("?" for _ in frontier)
            rows = self._conn.execute(
                f"SELECT source_id, target_id FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
                tuple(frontier) + tuple(frontier),
            ).fetchall()
            next_frontier: set[int] = set()
            for r in rows:
                next_frontier.add(r["source_id"])
                next_frontier.add(r["target_id"])
            visited |= frontier
            frontier = next_frontier - visited
        visited |= frontier
        visited.discard(note_id)
        return visited

    def fts_search(self, query: str, limit: int = 10) -> list[sqlite3.Row]:
        return self._conn.execute(
            """SELECT n.*, rank FROM notes_fts
               JOIN notes n ON n.id = notes_fts.rowid
               WHERE notes_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (query, limit),
        ).fetchall()

    def get_recent_notes(self, days: int = 7, region_idx: int | None = None, limit: int = 20) -> list[sqlite3.Row]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        if region_idx is not None:
            return self._conn.execute(
                "SELECT * FROM notes WHERE modified_at >= ? AND region_idx = ? ORDER BY modified_at DESC LIMIT ?",
                (cutoff, region_idx, limit),
            ).fetchall()
        return self._conn.execute(
            "SELECT * FROM notes WHERE modified_at >= ? ORDER BY modified_at DESC LIMIT ?",
            (cutoff, limit),
        ).fetchall()

    def get_region_note_counts(self) -> dict[int, int]:
        rows = self._conn.execute("SELECT region_idx, COUNT(*) as cnt FROM notes GROUP BY region_idx").fetchall()
        return {r["region_idx"]: r["cnt"] for r in rows}

    def get_all_regions(self) -> list[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM regions ORDER BY idx").fetchall()

    def get_region(self, idx: int) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM regions WHERE idx=?", (idx,)).fetchone()

    def update_region(self, idx: int, description: str | None = None, color: str | None = None) -> None:
        if description is not None:
            self._conn.execute("UPDATE regions SET description=? WHERE idx=?", (description, idx))
        if color is not None:
            self._conn.execute("UPDATE regions SET color=? WHERE idx=?", (color, idx))
        self._conn.commit()

    def get_content_hash(self, path: str) -> str | None:
        row = self._conn.execute("SELECT content_hash FROM notes WHERE path=?", (path,)).fetchone()
        return row["content_hash"] if row else None
```

- [ ] **Step 5: Run tests**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_database.py -v`
Expected: 10 PASSED

- [ ] **Step 6: Commit**

```bash
git add brain_mcp/storage/ tests/test_database.py
git commit -m "feat(mcp): SQLite database with FTS5, migrations, and region seeding"
```

---

### Task 3: Vault Scanner

**Files:**
- Create: `brain_mcp/indexer/__init__.py`
- Create: `brain_mcp/indexer/scanner.py`
- Create: `tests/test_scanner.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scanner.py
from brain_mcp.indexer.scanner import scan_vault, compute_content_hash

def test_scan_vault_finds_md_files(tmp_vault):
    notes = scan_vault(tmp_vault, folder_to_region={})
    assert len(notes) == 3
    titles = {n["title"] for n in notes}
    assert titles == {"note1", "note2", "project1"}

def test_scan_vault_extracts_brain_tags(tmp_vault):
    notes = scan_vault(tmp_vault, folder_to_region={})
    by_title = {n["title"]: n for n in notes}
    assert by_title["note1"]["region_idx"] == 3  # hippocampus
    assert by_title["note2"]["region_idx"] == 0  # praefrontaler-cortex

def test_scan_vault_folder_mapping(tmp_vault):
    notes = scan_vault(tmp_vault, folder_to_region={"Projects": 10})
    by_title = {n["title"]: n for n in notes}
    assert by_title["project1"]["region_idx"] == 10

def test_scan_vault_default_region(tmp_vault):
    notes = scan_vault(tmp_vault, folder_to_region={})
    by_title = {n["title"]: n for n in notes}
    assert by_title["project1"]["region_idx"] == 9

def test_scan_vault_backlinks(tmp_vault):
    notes = scan_vault(tmp_vault, folder_to_region={})
    by_title = {n["title"]: n for n in notes}
    assert "note2" in by_title["note1"]["backlink_titles"]
    assert "note1" in by_title["note2"]["backlink_titles"]

def test_scan_vault_content_hash(tmp_vault):
    notes = scan_vault(tmp_vault, folder_to_region={})
    assert all("content_hash" in n for n in notes)
    assert all(len(n["content_hash"]) == 64 for n in notes)

def test_scan_vault_content_included(tmp_vault):
    notes = scan_vault(tmp_vault, folder_to_region={})
    by_title = {n["title"]: n for n in notes}
    assert "routing" in by_title["note1"]["content"].lower()

def test_compute_content_hash_deterministic():
    h1 = compute_content_hash("hello world")
    h2 = compute_content_hash("hello world")
    h3 = compute_content_hash("different")
    assert h1 == h2
    assert h1 != h3

def test_scan_vault_skips_obsidian_dir(tmp_vault):
    obs = tmp_vault / ".obsidian"
    obs.mkdir()
    (obs / "config.md").write_text("# Config", encoding="utf-8")
    notes = scan_vault(tmp_vault, folder_to_region={})
    titles = {n["title"] for n in notes}
    assert "config" not in titles
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_scanner.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement scanner.py**

```python
# brain_mcp/indexer/__init__.py
```

```python
# brain_mcp/indexer/scanner.py
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

_BACKLINK_RE = re.compile(r"\[\[([^\]|#]+?)(?:[|#][^\]]*?)?\]\]")
_BRAIN_TAG_RE = re.compile(r"#brain/([\w-]+)")

REGION_TAG_TO_IDX = {
    "praefrontaler-cortex": 0, "motorischer-cortex": 1, "sensorischer-cortex": 2,
    "hippocampus": 3, "kleinhirn": 4, "nucleus-accumbens": 5,
    "broca-areal": 6, "visueller-cortex": 7, "thalamus": 8,
    "stammhirn": 9, "basalganglien": 10, "amygdala": 11,
}


def compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scan_vault(vault_path: Path, folder_to_region: dict[str, int]) -> list[dict]:
    md_files = sorted(vault_path.rglob("*.md"))
    md_files = [f for f in md_files if ".obsidian" not in f.parts]

    notes: list[dict] = []
    for f in md_files:
        rel = f.relative_to(vault_path)
        title = f.stem
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        brain_tags = _BRAIN_TAG_RE.findall(text)
        region_idx = 9
        if brain_tags:
            region_idx = REGION_TAG_TO_IDX.get(brain_tags[0], 9)
        else:
            top_folder = rel.parts[0] if len(rel.parts) > 1 else ""
            region_idx = folder_to_region.get(top_folder, 9)

        backlinks = _BACKLINK_RE.findall(text)
        word_count = len(text.split())
        all_tags = list(set(re.findall(r"#[\w/-]+", text)))
        mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)

        notes.append({
            "path": str(rel).replace("\\", "/"),
            "title": title,
            "content": text,
            "content_hash": compute_content_hash(text),
            "region_idx": region_idx,
            "tags": all_tags[:20],
            "word_count": word_count,
            "created_at": mtime.strftime("%Y-%m-%d"),
            "modified_at": mtime.isoformat(),
            "backlink_titles": [bl.strip() for bl in backlinks],
        })

    return notes
```

- [ ] **Step 4: Run tests**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_scanner.py -v`
Expected: 9 PASSED

- [ ] **Step 5: Commit**

```bash
git add brain_mcp/indexer/ tests/test_scanner.py
git commit -m "feat(mcp): vault scanner with backlink extraction and region tagging"
```

---

### Task 4: Embedding Backend

**Files:**
- Create: `brain_mcp/indexer/embedder.py`
- Create: `tests/test_embedder.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_embedder.py
import numpy as np
from brain_mcp.indexer.embedder import EmbeddingBackend, SentenceTransformerBackend

def test_mock_embedder_implements_protocol(mock_embedder):
    assert isinstance(mock_embedder.dimension, int)
    assert mock_embedder.dimension == 384

def test_mock_embedder_returns_correct_shape(mock_embedder):
    result = mock_embedder.embed(["hello", "world"])
    assert result.shape == (2, 384)
    assert result.dtype == np.float32

def test_mock_embedder_normalized(mock_embedder):
    result = mock_embedder.embed(["test"])
    norm = np.linalg.norm(result[0])
    assert abs(norm - 1.0) < 1e-5

def test_mock_embedder_deterministic(mock_embedder):
    r1 = mock_embedder.embed(["hello"])
    r2 = mock_embedder.embed(["hello"])
    np.testing.assert_array_equal(r1, r2)

def test_mock_embedder_different_texts_differ(mock_embedder):
    r1 = mock_embedder.embed(["hello"])
    r2 = mock_embedder.embed(["world"])
    assert not np.allclose(r1, r2)

def test_sentence_transformer_backend_has_protocol_methods():
    assert hasattr(SentenceTransformerBackend, "embed")
    assert hasattr(SentenceTransformerBackend, "dimension")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_embedder.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement embedder.py**

```python
# brain_mcp/indexer/embedder.py
from __future__ import annotations

import sys
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingBackend(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...

    @property
    def dimension(self) -> int: ...


class SentenceTransformerBackend:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self._model_name = model_name
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        print(f"Loading embedding model: {self._model_name}...", file=sys.stderr)
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._model_name)
        print("Model loaded.", file=sys.stderr)

    @property
    def dimension(self) -> int:
        return 384

    def embed(self, texts: list[str]) -> np.ndarray:
        self._load()
        vectors = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(vectors, dtype=np.float32)
```

- [ ] **Step 4: Run tests**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_embedder.py -v`
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add brain_mcp/indexer/embedder.py tests/test_embedder.py
git commit -m "feat(mcp): embedding backend protocol with sentence-transformers implementation"
```

---

### Task 5: FAISS Vector Store

**Files:**
- Create: `brain_mcp/indexer/vector_store.py`
- Create: `tests/test_vector_store.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_vector_store.py
import numpy as np
from brain_mcp.indexer.vector_store import VectorStore

def _random_vectors(n, dim=384, seed=42):
    rng = np.random.RandomState(seed)
    vecs = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms

def test_add_and_search():
    store = VectorStore(dimension=384)
    vecs = _random_vectors(5)
    ids = store.add(vecs)
    assert ids == [0, 1, 2, 3, 4]
    assert store.size == 5
    scores, result_ids = store.search(vecs[0:1], k=3)
    assert result_ids[0][0] == 0
    assert scores[0][0] > 0.99

def test_search_returns_top_k():
    store = VectorStore(dimension=384)
    vecs = _random_vectors(10)
    store.add(vecs)
    scores, result_ids = store.search(vecs[0:1], k=5)
    assert len(result_ids[0]) == 5
    assert result_ids[0][0] == 0

def test_save_and_load(tmp_path):
    store = VectorStore(dimension=384)
    vecs = _random_vectors(5)
    store.add(vecs)
    path = tmp_path / "test.faiss"
    store.save(path)
    store2 = VectorStore.load(path, dimension=384)
    assert store2.size == 5
    scores, result_ids = store2.search(vecs[0:1], k=1)
    assert result_ids[0][0] == 0

def test_empty_store_search():
    store = VectorStore(dimension=384)
    query = _random_vectors(1)
    scores, result_ids = store.search(query, k=5)
    assert len(result_ids[0]) == 0

def test_load_nonexistent_returns_empty(tmp_path):
    path = tmp_path / "missing.faiss"
    store = VectorStore.load(path, dimension=384)
    assert store.size == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_vector_store.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement vector_store.py**

```python
# brain_mcp/indexer/vector_store.py
from __future__ import annotations

import threading
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._lock = threading.RLock()

    @property
    def size(self) -> int:
        with self._lock:
            return self._index.ntotal

    def add(self, vectors: np.ndarray) -> list[int]:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors)
        with self._lock:
            start = self._index.ntotal
            self._index.add(vectors)
            return list(range(start, start + len(vectors)))

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        query = np.ascontiguousarray(query, dtype=np.float32)
        faiss.normalize_L2(query)
        with self._lock:
            if self._index.ntotal == 0:
                return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)
            actual_k = min(k, self._index.ntotal)
            scores, ids = self._index.search(query, actual_k)
            return scores, ids

    def save(self, path: Path | str) -> None:
        with self._lock:
            faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: Path | str, dimension: int = 384) -> VectorStore:
        store = cls(dimension=dimension)
        path = Path(path)
        if path.exists():
            store._index = faiss.read_index(str(path))
        return store
```

- [ ] **Step 4: Run tests**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_vector_store.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add brain_mcp/indexer/vector_store.py tests/test_vector_store.py
git commit -m "feat(mcp): FAISS vector store with RLock and L2 normalization"
```

---

### Task 6: MCP Server Shell + brain_recent Tool

**Files:**
- Create: `brain_mcp/server.py`
- Create: `brain_mcp/__main__.py`
- Create: `brain_mcp/tools/__init__.py`
- Create: `brain_mcp/tools/recent.py`
- Create: `tests/test_tool_recent.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tool_recent.py
import json
from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.recent import handle_brain_recent

def _seed_db(db: BrainDB) -> None:
    db.upsert_note(path="a.md", title="Recent Note", content="c", content_hash="h1",
                   region_idx=3, tags=["#test"], word_count=10,
                   created_at="2026-04-18", modified_at="2026-04-18T10:00:00+00:00")
    db.upsert_note(path="b.md", title="Old Note", content="c", content_hash="h2",
                   region_idx=0, tags=[], word_count=5,
                   created_at="2025-01-01", modified_at="2025-01-01T10:00:00+00:00")
    db.upsert_note(path="c.md", title="Also Recent", content="c", content_hash="h3",
                   region_idx=3, tags=[], word_count=8,
                   created_at="2026-04-17", modified_at="2026-04-17T10:00:00+00:00")

def test_brain_recent_default(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    _seed_db(db)
    result = handle_brain_recent(db, days=7, region=None, limit=20)
    assert len(result) == 2
    assert result[0]["title"] == "Recent Note"
    db.close()

def test_brain_recent_region_filter(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    _seed_db(db)
    result = handle_brain_recent(db, days=7, region="Hippocampus", limit=20)
    assert len(result) == 2
    assert all(r["region"] == "Hippocampus" for r in result)
    db.close()

def test_brain_recent_limit(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    _seed_db(db)
    result = handle_brain_recent(db, days=365, region=None, limit=1)
    assert len(result) == 1
    db.close()

def test_brain_recent_output_format(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    _seed_db(db)
    result = handle_brain_recent(db, days=7, region=None, limit=20)
    note = result[0]
    assert "title" in note
    assert "path" in note
    assert "region" in note
    assert "modified_at" in note
    assert "word_count" in note
    db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_tool_recent.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement tools/recent.py**

```python
# brain_mcp/tools/__init__.py
```

```python
# brain_mcp/tools/recent.py
from __future__ import annotations

from brain_mcp.storage.database import BrainDB

REGION_NAMES = [
    "Praefrontaler Cortex", "Motorischer Cortex", "Sensorischer Cortex",
    "Hippocampus", "Kleinhirn", "Nucleus Accumbens", "Broca-Areal",
    "Visueller Cortex", "Thalamus", "Stammhirn", "Basalganglien", "Amygdala",
]

REGION_NAME_TO_IDX = {name: i for i, name in enumerate(REGION_NAMES)}


def handle_brain_recent(
    db: BrainDB,
    days: int = 7,
    region: str | None = None,
    limit: int = 20,
) -> list[dict]:
    days = max(1, min(days, 365))
    limit = max(1, min(limit, 100))
    region_idx = REGION_NAME_TO_IDX.get(region) if region else None

    rows = db.get_recent_notes(days=days, region_idx=region_idx, limit=limit)
    results = []
    for r in rows:
        results.append({
            "title": r["title"],
            "path": r["path"],
            "region": REGION_NAMES[r["region_idx"]] if 0 <= r["region_idx"] < 12 else "Stammhirn",
            "modified_at": r["modified_at"],
            "word_count": r["word_count"],
        })
    return results
```

- [ ] **Step 4: Implement server.py and __main__.py**

```python
# brain_mcp/server.py
from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from brain_mcp.config import BrainConfig, load_config
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.recent import handle_brain_recent


@dataclass
class BrainState:
    config: BrainConfig
    db: BrainDB
    vectors: VectorStore
    embedder: object | None = None


@asynccontextmanager
async def brain_lifespan(server: FastMCP) -> AsyncIterator[BrainState]:
    config = load_config()
    config.data_dir.mkdir(parents=True, exist_ok=True)

    db = BrainDB(config.db_path)
    vectors = VectorStore.load(config.index_path, dimension=384)

    print(f"Brain MCP Server started. Vault: {config.vault_path}", file=sys.stderr)
    print(f"DB: {config.db_path} | Index: {vectors.size} vectors", file=sys.stderr)

    try:
        yield BrainState(config=config, db=db, vectors=vectors)
    finally:
        vectors.save(config.index_path)
        db.close()
        print("Brain MCP Server stopped.", file=sys.stderr)


mcp = FastMCP(
    "Neural Brain",
    lifespan=brain_lifespan,
)


@mcp.tool()
def brain_recent(days: int = 7, region: str | None = None, limit: int = 20) -> list[dict]:
    """Show recently modified notes in the vault.

    Args:
        days: Lookback window in days (default 7, max 365)
        region: Filter by brain region name (e.g. "Hippocampus")
        limit: Max results (default 20, max 100)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    return handle_brain_recent(state.db, days=days, region=region, limit=limit)
```

```python
# brain_mcp/__main__.py
import sys

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "index":
        print("Index command not yet implemented.", file=sys.stderr)
        sys.exit(1)

    from brain_mcp.server import mcp
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_tool_recent.py -v`
Expected: 4 PASSED

- [ ] **Step 6: Commit**

```bash
git add brain_mcp/server.py brain_mcp/__main__.py brain_mcp/tools/ tests/test_tool_recent.py
git commit -m "feat(mcp): MCP server shell with brain_recent tool"
```

---

### Task 7: brain_regions Tool

**Files:**
- Create: `brain_mcp/tools/regions.py`
- Create: `tests/test_tool_regions.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tool_regions.py
from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.regions import handle_brain_regions

def _seed(db):
    db.upsert_note(path="a.md", title="A", content="c", content_hash="h1",
                   region_idx=0, tags=[], word_count=100, created_at="2026-01-01", modified_at="2026-01-01")
    db.upsert_note(path="b.md", title="B", content="c", content_hash="h2",
                   region_idx=0, tags=[], word_count=50, created_at="2026-01-01", modified_at="2026-01-01")
    db.upsert_note(path="c.md", title="C", content="c", content_hash="h3",
                   region_idx=3, tags=[], word_count=200, created_at="2026-01-01", modified_at="2026-01-01")

def test_list_regions(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    _seed(db)
    result = handle_brain_regions(db, action="list")
    assert len(result) == 12
    assert result[0]["name"] == "Praefrontaler Cortex"
    assert result[0]["note_count"] == 2
    assert result[3]["note_count"] == 1
    db.close()

def test_describe_region(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    _seed(db)
    result = handle_brain_regions(db, action="describe", region="Praefrontaler Cortex")
    assert result["name"] == "Praefrontaler Cortex"
    assert result["note_count"] == 2
    assert len(result["top_notes"]) == 2
    assert result["top_notes"][0]["word_count"] >= result["top_notes"][1]["word_count"]
    db.close()

def test_customize_region(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    result = handle_brain_regions(db, action="customize", region="Praefrontaler Cortex",
                                  description="My custom description", color="#FF0000")
    assert result["updated"] is True
    row = db.get_region(0)
    assert row["description"] == "My custom description"
    assert row["color"] == "#FF0000"
    db.close()

def test_describe_unknown_region(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    result = handle_brain_regions(db, action="describe", region="Nonexistent")
    assert "error" in result
    db.close()

def test_invalid_action(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    result = handle_brain_regions(db, action="delete")
    assert "error" in result
    db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_tool_regions.py -v`
Expected: FAIL

- [ ] **Step 3: Implement regions.py**

```python
# brain_mcp/tools/regions.py
from __future__ import annotations

import json

from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.recent import REGION_NAMES, REGION_NAME_TO_IDX


def handle_brain_regions(
    db: BrainDB,
    action: str,
    region: str | None = None,
    description: str | None = None,
    color: str | None = None,
) -> dict | list[dict]:
    if action == "list":
        return _list_regions(db)
    elif action == "describe":
        if not region:
            return {"error": "region parameter required for describe action"}
        return _describe_region(db, region)
    elif action == "customize":
        if not region:
            return {"error": "region parameter required for customize action"}
        return _customize_region(db, region, description, color)
    else:
        return {"error": f"Unknown action: {action}. Use 'list', 'describe', or 'customize'."}


def _list_regions(db: BrainDB) -> list[dict]:
    regions = db.get_all_regions()
    counts = db.get_region_note_counts()
    results = []
    for r in regions:
        results.append({
            "idx": r["idx"],
            "name": r["name"],
            "color": r["color"],
            "description": r["description"],
            "note_count": counts.get(r["idx"], 0),
            "position": json.loads(r["position"]),
        })
    return results


def _describe_region(db: BrainDB, region_name: str) -> dict:
    idx = REGION_NAME_TO_IDX.get(region_name)
    if idx is None:
        return {"error": f"Unknown region: {region_name}. Use brain_regions(action='list') to see available regions."}
    region = db.get_region(idx)
    counts = db.get_region_note_counts()
    notes = db.execute(
        "SELECT title, path, word_count FROM notes WHERE region_idx=? ORDER BY word_count DESC LIMIT 10",
        (idx,),
    ).fetchall()
    return {
        "idx": region["idx"],
        "name": region["name"],
        "color": region["color"],
        "description": region["description"],
        "position": json.loads(region["position"]),
        "note_count": counts.get(idx, 0),
        "top_notes": [{"title": n["title"], "path": n["path"], "word_count": n["word_count"]} for n in notes],
    }


def _customize_region(db: BrainDB, region_name: str, description: str | None, color: str | None) -> dict:
    idx = REGION_NAME_TO_IDX.get(region_name)
    if idx is None:
        return {"error": f"Unknown region: {region_name}"}
    db.update_region(idx, description=description, color=color)
    return {"updated": True, "region": region_name, "idx": idx}
```

- [ ] **Step 4: Register in server.py**

Add to `brain_mcp/server.py` after the `brain_recent` tool:

```python
from brain_mcp.tools.regions import handle_brain_regions

@mcp.tool()
def brain_regions(action: str, region: str | None = None, description: str | None = None, color: str | None = None) -> dict | list[dict]:
    """List, describe, or customize brain region definitions.

    Args:
        action: "list" (all regions), "describe" (one region detail), or "customize" (update region)
        region: Region name (required for describe/customize)
        description: New description (for customize)
        color: New hex color (for customize)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    return handle_brain_regions(state.db, action=action, region=region, description=description, color=color)
```

- [ ] **Step 5: Run tests**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_tool_regions.py -v`
Expected: 5 PASSED

- [ ] **Step 6: Commit**

```bash
git add brain_mcp/tools/regions.py brain_mcp/server.py tests/test_tool_regions.py
git commit -m "feat(mcp): brain_regions tool with list, describe, customize actions"
```

---

### Task 8: brain_retrieve Tool

**Files:**
- Create: `brain_mcp/tools/retrieve.py`
- Create: `tests/test_tool_retrieve.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tool_retrieve.py
from brain_mcp.storage.database import BrainDB
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.tools.retrieve import handle_brain_retrieve

def _seed(db, vectors, embedder):
    notes = [
        ("routing.md", "Routing", "Express routing handles HTTP requests and maps URLs to handlers", 3),
        ("auth.md", "Auth", "JWT authentication protects API endpoints with bearer tokens", 11),
        ("css.md", "CSS Themes", "Tailwind CSS themes with dark mode and custom color palette", 7),
    ]
    for path, title, content, region_idx in notes:
        note_id = db.upsert_note(path=path, title=title, content=content, content_hash=f"h_{path}",
                                  region_idx=region_idx, tags=[], word_count=len(content.split()),
                                  created_at="2026-01-01", modified_at="2026-01-01")
        vec = embedder.embed([content])
        faiss_ids = vectors.add(vec)
        db.set_faiss_idx(note_id, faiss_ids[0])

def test_retrieve_finds_relevant(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_retrieve(db, vectors, mock_embedder, query="HTTP routing middleware", limit=3)
    assert len(result) >= 1
    assert all("title" in r for r in result)
    assert all("similarity" in r for r in result)
    db.close()

def test_retrieve_region_filter(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_retrieve(db, vectors, mock_embedder, query="routing", region="Hippocampus", limit=10)
    assert all(r["region"] == "Hippocampus" for r in result)
    db.close()

def test_retrieve_respects_limit(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_retrieve(db, vectors, mock_embedder, query="test", limit=1)
    assert len(result) <= 1
    db.close()

def test_retrieve_output_format(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_retrieve(db, vectors, mock_embedder, query="routing", limit=3)
    if result:
        r = result[0]
        assert "title" in r
        assert "path" in r
        assert "region" in r
        assert "similarity" in r
        assert "snippet" in r
        assert "word_count" in r
    db.close()

def test_retrieve_empty_index(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    result = handle_brain_retrieve(db, vectors, mock_embedder, query="anything", limit=10)
    assert result == []
    db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_tool_retrieve.py -v`
Expected: FAIL

- [ ] **Step 3: Implement retrieve.py**

```python
# brain_mcp/tools/retrieve.py
from __future__ import annotations

import json

from brain_mcp.indexer.embedder import EmbeddingBackend
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.recent import REGION_NAMES, REGION_NAME_TO_IDX


def handle_brain_retrieve(
    db: BrainDB,
    vectors: VectorStore,
    embedder: EmbeddingBackend,
    query: str,
    region: str | None = None,
    limit: int = 10,
    threshold: float = 0.3,
) -> list[dict]:
    query = query[:1000]
    limit = max(1, min(limit, 100))

    if vectors.size == 0:
        return []

    query_vec = embedder.embed([query])
    scores, ids = vectors.search(query_vec, k=min(limit * 3, vectors.size))

    faiss_indices = [int(i) for i in ids[0] if i >= 0]
    if not faiss_indices:
        return []

    notes = db.get_notes_by_faiss_indices(faiss_indices)
    note_map = {n["faiss_idx"]: n for n in notes}

    region_idx_filter = REGION_NAME_TO_IDX.get(region) if region else None
    results = []
    for rank, (faiss_id, score) in enumerate(zip(ids[0], scores[0])):
        faiss_id = int(faiss_id)
        score = float(score)
        if faiss_id < 0 or score < threshold:
            continue
        note = note_map.get(faiss_id)
        if note is None:
            continue
        if region_idx_filter is not None and note["region_idx"] != region_idx_filter:
            continue

        content = note["content"] or ""
        snippet = content[:200].strip()
        if len(content) > 200:
            snippet += "..."

        results.append({
            "title": note["title"],
            "path": note["path"],
            "region": REGION_NAMES[note["region_idx"]] if 0 <= note["region_idx"] < 12 else "Stammhirn",
            "region_idx": note["region_idx"],
            "similarity": round(score, 4),
            "snippet": snippet,
            "tags": json.loads(note["tags"]) if note["tags"] else [],
            "created": note["created_at"],
            "modified": note["modified_at"],
            "word_count": note["word_count"],
        })
        if len(results) >= limit:
            break

    return results
```

- [ ] **Step 4: Register in server.py**

Add to `brain_mcp/server.py`:

```python
from brain_mcp.tools.retrieve import handle_brain_retrieve
from brain_mcp.indexer.embedder import SentenceTransformerBackend

# In brain_lifespan, before yield:
embedder = SentenceTransformerBackend(config.model_name)

# Update BrainState yield:
yield BrainState(config=config, db=db, vectors=vectors, embedder=embedder)

@mcp.tool()
async def brain_retrieve(query: str, region: str | None = None, limit: int = 10, threshold: float = 0.3, ctx: Context = None) -> list[dict]:
    """Semantic search across all vault notes. Finds notes by meaning, not just keywords.

    Args:
        query: Natural language search query
        region: Filter by brain region name (e.g. "Hippocampus")
        limit: Max results (default 10, max 100)
        threshold: Min cosine similarity (default 0.3)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    if state.embedder is None:
        return [{"error": "Embedding model not available"}]
    return handle_brain_retrieve(state.db, state.vectors, state.embedder, query=query, region=region, limit=limit, threshold=threshold)
```

- [ ] **Step 5: Run tests**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_tool_retrieve.py -v`
Expected: 5 PASSED

- [ ] **Step 6: Commit**

```bash
git add brain_mcp/tools/retrieve.py brain_mcp/server.py tests/test_tool_retrieve.py
git commit -m "feat(mcp): brain_retrieve tool with semantic search and region filtering"
```

---

### Task 9: brain_store Tool + Security

**Files:**
- Create: `brain_mcp/tools/store.py`
- Create: `tests/test_tool_store.py`
- Create: `tests/test_security.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tool_store.py
from pathlib import Path
from brain_mcp.storage.database import BrainDB
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.tools.store import handle_brain_store, sanitize_title

def test_sanitize_title_strips_traversal():
    assert sanitize_title("../../etc/passwd") == "etcpasswd"
    assert sanitize_title("normal title") == "normal title"
    assert sanitize_title("a" * 300) == "a" * 200

def test_sanitize_title_strips_special_chars():
    assert sanitize_title('file:name*bad?"yes"') == "filenamebadyes"

def test_store_creates_file(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    result = handle_brain_store(
        db, vectors, mock_embedder, vault,
        title="My Note", content="Hello world content", region="Hippocampus",
        tags=["#test"], folder="",
        pending_writes={}
    )
    assert result["path"] == "My Note.md"
    assert (vault / "My Note.md").exists()
    assert "Hello world content" in (vault / "My Note.md").read_text(encoding="utf-8")
    db.close()

def test_store_creates_subfolder(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    result = handle_brain_store(
        db, vectors, mock_embedder, vault,
        title="Sub Note", content="In a folder", region=None,
        tags=[], folder="Projects",
        pending_writes={}
    )
    assert result["path"] == "Projects/Sub Note.md"
    assert (vault / "Projects" / "Sub Note.md").exists()
    db.close()

def test_store_indexes_note(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    handle_brain_store(db, vectors, mock_embedder, vault,
                       title="Indexed", content="Some searchable content", region=None,
                       tags=[], folder="", pending_writes={})
    assert vectors.size == 1
    row = db.get_note_by_title("Indexed")
    assert row is not None
    assert row["faiss_idx"] == 0
    db.close()

def test_store_adds_brain_tag(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    handle_brain_store(db, vectors, mock_embedder, vault,
                       title="Tagged", content="Content here", region="Hippocampus",
                       tags=[], folder="", pending_writes={})
    text = (vault / "Tagged.md").read_text(encoding="utf-8")
    assert "#brain/hippocampus" in text
    db.close()

def test_store_sets_pending_write(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    pending = {}
    handle_brain_store(db, vectors, mock_embedder, vault,
                       title="Pending", content="Content", region=None,
                       tags=[], folder="", pending_writes=pending)
    assert len(pending) == 1
    db.close()
```

```python
# tests/test_security.py
import pytest
from pathlib import Path
from brain_mcp.tools.store import handle_brain_store, sanitize_title
from brain_mcp.storage.database import BrainDB
from brain_mcp.indexer.vector_store import VectorStore

def test_path_traversal_blocked(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    result = handle_brain_store(
        db, vectors, mock_embedder, vault,
        title="../../etc/passwd", content="bad", region=None,
        tags=[], folder="", pending_writes={}
    )
    assert not (tmp_path / "etc" / "passwd.md").exists()
    db.close()

def test_path_traversal_via_folder(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    result = handle_brain_store(
        db, vectors, mock_embedder, vault,
        title="test", content="bad", region=None,
        tags=[], folder="../../outside", pending_writes={}
    )
    assert "error" in result
    db.close()

def test_content_size_limit(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    huge_content = "x" * (1024 * 1024 + 1)
    result = handle_brain_store(
        db, vectors, mock_embedder, vault,
        title="big", content=huge_content, region=None,
        tags=[], folder="", pending_writes={}
    )
    assert "error" in result
    db.close()

def test_too_many_tags(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    result = handle_brain_store(
        db, vectors, mock_embedder, vault,
        title="tagged", content="c", region=None,
        tags=["#t" + str(i) for i in range(25)], folder="", pending_writes={}
    )
    assert result.get("path") is not None
    row = db.get_note_by_title("tagged")
    import json
    assert len(json.loads(row["tags"])) <= 20
    db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_tool_store.py tests/test_security.py -v`
Expected: FAIL

- [ ] **Step 3: Implement store.py**

```python
# brain_mcp/tools/store.py
from __future__ import annotations

import os
import re
import time
from pathlib import Path

from brain_mcp.indexer.embedder import EmbeddingBackend
from brain_mcp.indexer.scanner import REGION_TAG_TO_IDX, compute_content_hash
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.recent import REGION_NAMES, REGION_NAME_TO_IDX

_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_TRAVERSAL = re.compile(r'\.\.[\\/]?')
MAX_CONTENT_SIZE = 1024 * 1024  # 1MB
MAX_TITLE_LEN = 200
MAX_TAGS = 20
MAX_TAG_LEN = 100

REGION_NAME_TO_SLUG = {name: slug for slug, idx in REGION_TAG_TO_IDX.items()
                        for i, name in enumerate(REGION_NAMES) if i == idx}


def sanitize_title(title: str) -> str:
    title = _TRAVERSAL.sub("", title)
    title = _BAD_CHARS.sub("", title)
    return title.strip()[:MAX_TITLE_LEN]


def handle_brain_store(
    db: BrainDB,
    vectors: VectorStore,
    embedder: EmbeddingBackend,
    vault_root: Path,
    title: str,
    content: str,
    region: str | None = None,
    region_idx: int | None = None,
    tags: list[str] | None = None,
    folder: str = "",
    pending_writes: dict[str, float] | None = None,
) -> dict:
    if len(content) > MAX_CONTENT_SIZE:
        return {"error": f"Content exceeds {MAX_CONTENT_SIZE} byte limit"}

    safe_title = sanitize_title(title)
    if not safe_title:
        return {"error": "Title is empty after sanitization"}

    tags = (tags or [])[:MAX_TAGS]
    tags = [t[:MAX_TAG_LEN] for t in tags]

    folder_clean = _TRAVERSAL.sub("", folder).strip("/\\")
    target_dir = (vault_root / folder_clean) if folder_clean else vault_root
    target = target_dir / f"{safe_title}.md"

    try:
        resolved = target.resolve()
        if not resolved.is_relative_to(vault_root.resolve()):
            return {"error": "Path escapes vault directory"}
    except (ValueError, OSError):
        return {"error": "Invalid path"}

    if region_idx is not None and 0 <= region_idx < 12:
        r_idx = region_idx
    elif region:
        r_idx = REGION_NAME_TO_IDX.get(region, 9)
    else:
        r_idx = 9

    region_slug = REGION_NAME_TO_SLUG.get(REGION_NAMES[r_idx])
    if region_slug and f"#brain/{region_slug}" not in content:
        content = content.rstrip() + f"\n\n#brain/{region_slug}\n"

    target_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = target.with_suffix(".md.tmp")
    try:
        tmp_file.write_text(content, encoding="utf-8")
        os.replace(str(tmp_file), str(target))
    except OSError as e:
        return {"error": f"Write failed: {e}"}

    rel_path = str(target.relative_to(vault_root)).replace("\\", "/")

    if pending_writes is not None:
        pending_writes[str(resolved)] = time.time()

    content_hash = compute_content_hash(content)
    word_count = len(content.split())
    note_id = db.upsert_note(
        path=rel_path, title=safe_title, content=content, content_hash=content_hash,
        region_idx=r_idx, tags=tags, word_count=word_count,
        created_at=time.strftime("%Y-%m-%d"), modified_at=time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
    )

    vec = embedder.embed([content])
    faiss_ids = vectors.add(vec)
    db.set_faiss_idx(note_id, faiss_ids[0])

    return {
        "path": rel_path,
        "region": REGION_NAMES[r_idx],
        "region_idx": r_idx,
        "indexed": True,
        "word_count": word_count,
    }
```

- [ ] **Step 4: Register in server.py**

Add to `brain_mcp/server.py`:

```python
from brain_mcp.tools.store import handle_brain_store

@mcp.tool()
def brain_store(title: str, content: str, region: str | None = None, region_idx: int | None = None,
                tags: list[str] | None = None, folder: str = "") -> dict:
    """Create or update a note in the vault.

    Args:
        title: Note title (becomes filename)
        content: Markdown content
        region: Brain region name (auto-detected if omitted)
        region_idx: Region index (overrides name)
        tags: Additional tags (max 20)
        folder: Subfolder in vault
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    if state.config.vault_path is None:
        return {"error": "No vault_path configured"}
    return handle_brain_store(
        state.db, state.vectors, state.embedder, state.config.vault_path,
        title=title, content=content, region=region, region_idx=region_idx,
        tags=tags, folder=folder, pending_writes=getattr(state, '_pending_writes', {}),
    )
```

- [ ] **Step 5: Run tests**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_tool_store.py tests/test_security.py -v`
Expected: 10 PASSED

- [ ] **Step 6: Commit**

```bash
git add brain_mcp/tools/store.py tests/test_tool_store.py tests/test_security.py brain_mcp/server.py
git commit -m "feat(mcp): brain_store tool with path traversal guard and atomic writes"
```

---

### Task 10: brain_related Tool

**Files:**
- Create: `brain_mcp/tools/related.py`
- Create: `tests/test_tool_related.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tool_related.py
from brain_mcp.storage.database import BrainDB
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.tools.related import handle_brain_related

def _seed(db, vectors, embedder):
    notes = [
        ("a.md", "Routing", "Express routing handles HTTP requests", 3),
        ("b.md", "Auth", "JWT authentication protects endpoints", 11),
        ("c.md", "Middleware", "Middleware chain processes requests in order", 0),
    ]
    for path, title, content, region_idx in notes:
        nid = db.upsert_note(path=path, title=title, content=content, content_hash=f"h_{path}",
                              region_idx=region_idx, tags=[], word_count=len(content.split()),
                              created_at="2026-01-01", modified_at="2026-01-01")
        vec = embedder.embed([content])
        faiss_ids = vectors.add(vec)
        db.set_faiss_idx(nid, faiss_ids[0])
    a = db.get_note_by_path("a.md")
    c = db.get_note_by_path("c.md")
    db.upsert_edge(a["id"], c["id"], link_text="Middleware")

def test_related_by_title(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_related(db, vectors, mock_embedder, title="Routing", limit=5)
    assert len(result) >= 1
    assert all(r["title"] != "Routing" for r in result)
    db.close()

def test_related_by_path(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_related(db, vectors, mock_embedder, path="a.md", limit=5)
    assert len(result) >= 1
    db.close()

def test_related_includes_backlink_neighbor(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_related(db, vectors, mock_embedder, title="Routing", limit=5)
    titles = {r["title"] for r in result}
    assert "Middleware" in titles
    db.close()

def test_related_has_relation_type(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_related(db, vectors, mock_embedder, title="Routing", limit=5)
    assert all("relation_type" in r for r in result)
    db.close()

def test_related_unknown_title(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    result = handle_brain_related(db, vectors, mock_embedder, title="Nonexistent", limit=5)
    assert "error" in result or result == []
    db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_tool_related.py -v`
Expected: FAIL

- [ ] **Step 3: Implement related.py**

```python
# brain_mcp/tools/related.py
from __future__ import annotations

import json

from brain_mcp.indexer.embedder import EmbeddingBackend
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.recent import REGION_NAMES


def handle_brain_related(
    db: BrainDB,
    vectors: VectorStore,
    embedder: EmbeddingBackend,
    title: str | None = None,
    path: str | None = None,
    limit: int = 10,
) -> list[dict] | dict:
    limit = max(1, min(limit, 100))

    if path:
        source = db.get_note_by_path(path)
    elif title:
        source = db.get_note_by_title(title)
    else:
        return {"error": "Provide either title or path"}

    if source is None:
        return {"error": f"Note not found: {title or path}"}

    semantic_scores: dict[int, float] = {}
    if source["faiss_idx"] is not None and vectors.size > 1:
        scores, ids = vectors.search(
            embedder.embed([source["content"] or source["title"]]),
            k=min(limit * 2, vectors.size),
        )
        notes = db.get_notes_by_faiss_indices([int(i) for i in ids[0] if i >= 0])
        faiss_to_note = {n["faiss_idx"]: n for n in notes}
        for fid, score in zip(ids[0], scores[0]):
            fid = int(fid)
            note = faiss_to_note.get(fid)
            if note and note["id"] != source["id"]:
                semantic_scores[note["id"]] = float(score)

    neighbor_ids = db.get_neighbor_ids(source["id"], depth=1)

    all_ids = set(semantic_scores.keys()) | neighbor_ids
    results = []
    for nid in all_ids:
        note = db.get_note_by_id(nid)
        if note is None or note["id"] == source["id"]:
            continue

        sem_score = semantic_scores.get(nid, 0.0)
        graph_score = 1.0 if nid in neighbor_ids else 0.0
        combined = sem_score * 0.6 + graph_score * 0.4

        if sem_score > 0 and graph_score > 0:
            rel_type = "both"
        elif graph_score > 0:
            rel_type = "backlink"
        else:
            rel_type = "semantic"

        results.append({
            "title": note["title"],
            "path": note["path"],
            "region": REGION_NAMES[note["region_idx"]] if 0 <= note["region_idx"] < 12 else "Stammhirn",
            "score": round(combined, 4),
            "relation_type": rel_type,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:limit]
```

- [ ] **Step 4: Register in server.py**

Add to `brain_mcp/server.py`:

```python
from brain_mcp.tools.related import handle_brain_related

@mcp.tool()
def brain_related(title: str | None = None, path: str | None = None, limit: int = 10) -> list[dict] | dict:
    """Find notes related to a specific note by meaning or backlinks.

    Args:
        title: Note title to find relations for
        path: Note path (alternative to title)
        limit: Max results (default 10, max 100)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    return handle_brain_related(state.db, state.vectors, state.embedder, title=title, path=path, limit=limit)
```

- [ ] **Step 5: Run tests**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_tool_related.py -v`
Expected: 5 PASSED

- [ ] **Step 6: Commit**

```bash
git add brain_mcp/tools/related.py brain_mcp/server.py tests/test_tool_related.py
git commit -m "feat(mcp): brain_related tool with semantic + graph scoring"
```

---

### Task 11: brain_context Tool

**Files:**
- Create: `brain_mcp/tools/context.py`
- Create: `tests/test_tool_context.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tool_context.py
from brain_mcp.storage.database import BrainDB
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.tools.context import handle_brain_context

def _seed(db, vectors, embedder):
    notes_data = [
        ("routing.md", "Routing", "Express routing handles HTTP request mapping", 3),
        ("auth.md", "Auth", "JWT authentication and bearer token validation", 11),
        ("middleware.md", "Middleware", "Request middleware chain for Express apps", 0),
    ]
    for path, title, content, region_idx in notes_data:
        nid = db.upsert_note(path=path, title=title, content=content, content_hash=f"h_{path}",
                              region_idx=region_idx, tags=[], word_count=len(content.split()),
                              created_at="2026-01-01", modified_at="2026-01-01")
        vec = embedder.embed([content])
        faiss_ids = vectors.add(vec)
        db.set_faiss_idx(nid, faiss_ids[0])
    a = db.get_note_by_path("routing.md")
    c = db.get_note_by_path("middleware.md")
    db.upsert_edge(a["id"], c["id"], link_text="Middleware")

def test_context_by_file_paths(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_context(db, vectors, mock_embedder,
                                   file_paths=["routing.md"], max_notes=10)
    assert len(result) >= 1
    titles = {r["title"] for r in result}
    assert "Middleware" in titles
    db.close()

def test_context_by_task_description(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_context(db, vectors, mock_embedder,
                                   task_description="fixing authentication tokens", max_notes=5)
    assert len(result) >= 1
    db.close()

def test_context_combined(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_context(db, vectors, mock_embedder,
                                   file_paths=["routing.md"],
                                   task_description="adding auth middleware",
                                   max_notes=10)
    assert len(result) >= 1
    db.close()

def test_context_has_relevance_reason(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_context(db, vectors, mock_embedder,
                                   file_paths=["routing.md"], max_notes=10)
    assert all("relevance_reason" in r for r in result)
    db.close()

def test_context_respects_max_notes(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_context(db, vectors, mock_embedder,
                                   task_description="anything", max_notes=1)
    assert len(result) <= 1
    db.close()

def test_context_no_input(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    result = handle_brain_context(db, vectors, mock_embedder)
    assert "error" in result or result == []
    db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_tool_context.py -v`
Expected: FAIL

- [ ] **Step 3: Implement context.py**

```python
# brain_mcp/tools/context.py
from __future__ import annotations

import json
from pathlib import Path

from brain_mcp.indexer.embedder import EmbeddingBackend
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.recent import REGION_NAMES


def handle_brain_context(
    db: BrainDB,
    vectors: VectorStore,
    embedder: EmbeddingBackend,
    file_paths: list[str] | None = None,
    task_description: str | None = None,
    depth: int = 1,
    max_notes: int = 10,
) -> list[dict] | dict:
    depth = max(1, min(depth, 3))
    max_notes = max(1, min(max_notes, 100))

    if not file_paths and not task_description:
        return {"error": "Provide file_paths, task_description, or both"}

    scored: dict[int, dict] = {}

    if file_paths:
        for fp in file_paths:
            stem = Path(fp).stem
            note = db.get_note_by_path(fp) or db.get_note_by_title(stem)
            if note is None:
                continue
            neighbor_ids = db.get_neighbor_ids(note["id"], depth=depth)
            for nid in neighbor_ids:
                if nid not in scored:
                    neighbor = db.get_note_by_id(nid)
                    if neighbor:
                        scored[nid] = {
                            "note": neighbor,
                            "graph_score": 1.0,
                            "semantic_score": 0.0,
                            "reason": f"backlink from {note['title']} ({depth} hop)",
                        }

    if task_description and vectors.size > 0:
        query_vec = embedder.embed([task_description[:1000]])
        k = min(max_notes * 2, vectors.size)
        scores, ids = vectors.search(query_vec, k=k)
        faiss_indices = [int(i) for i in ids[0] if i >= 0]
        notes = db.get_notes_by_faiss_indices(faiss_indices)
        faiss_to_note = {n["faiss_idx"]: n for n in notes}

        for fid, score in zip(ids[0], scores[0]):
            fid = int(fid)
            score = float(score)
            note = faiss_to_note.get(fid)
            if note is None:
                continue
            nid = note["id"]
            if nid in scored:
                scored[nid]["semantic_score"] = score
                scored[nid]["reason"] += f" + semantic match ({score:.2f})"
            else:
                scored[nid] = {
                    "note": note,
                    "graph_score": 0.0,
                    "semantic_score": score,
                    "reason": f"semantic match ({score:.2f})",
                }

    source_paths = set(file_paths or [])
    results = []
    for nid, data in scored.items():
        note = data["note"]
        if note["path"] in source_paths:
            continue
        combined = data["semantic_score"] * 0.6 + data["graph_score"] * 0.4
        results.append({
            "title": note["title"],
            "path": note["path"],
            "region": REGION_NAMES[note["region_idx"]] if 0 <= note["region_idx"] < 12 else "Stammhirn",
            "similarity": round(combined, 4),
            "relevance_reason": data["reason"],
            "word_count": note["word_count"],
        })

    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results[:max_notes]
```

- [ ] **Step 4: Register in server.py**

Add to `brain_mcp/server.py`:

```python
from brain_mcp.tools.context import handle_brain_context

@mcp.tool()
def brain_context(file_paths: list[str] | None = None, task_description: str | None = None,
                  depth: int = 1, max_notes: int = 10) -> list[dict] | dict:
    """Get contextually relevant notes for current work. Combines file proximity and semantic search.

    Args:
        file_paths: Files currently being edited
        task_description: What you are doing
        depth: Backlink graph hops 1-3 (default 1)
        max_notes: Max notes returned (default 10, max 100)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    return handle_brain_context(state.db, state.vectors, state.embedder,
                                 file_paths=file_paths, task_description=task_description,
                                 depth=depth, max_notes=max_notes)
```

- [ ] **Step 5: Run tests**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_tool_context.py -v`
Expected: 6 PASSED

- [ ] **Step 6: Commit**

```bash
git add brain_mcp/tools/context.py brain_mcp/server.py tests/test_tool_context.py
git commit -m "feat(mcp): brain_context tool with semantic + backlink graph blending"
```

---

### Task 12: File Watcher

**Files:**
- Create: `brain_mcp/indexer/watcher.py`
- Create: `tests/test_watcher.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_watcher.py
import time
from pathlib import Path
from brain_mcp.indexer.watcher import BrainWatcher

def test_watcher_starts_and_stops(tmp_path):
    watcher = BrainWatcher(tmp_path, on_change=lambda p, ev: None)
    watcher.start()
    assert watcher.is_running
    watcher.stop()
    assert not watcher.is_running

def test_watcher_detects_file_create(tmp_path):
    events = []
    watcher = BrainWatcher(tmp_path, on_change=lambda p, ev: events.append((p, ev)))
    watcher.start()
    time.sleep(0.3)
    (tmp_path / "new.md").write_text("# New", encoding="utf-8")
    time.sleep(1.0)
    watcher.stop()
    md_events = [(p, e) for p, e in events if p.endswith(".md")]
    assert len(md_events) >= 1

def test_watcher_ignores_non_md(tmp_path):
    events = []
    watcher = BrainWatcher(tmp_path, on_change=lambda p, ev: events.append((p, ev)))
    watcher.start()
    time.sleep(0.3)
    (tmp_path / "ignore.txt").write_text("not markdown", encoding="utf-8")
    time.sleep(1.0)
    watcher.stop()
    md_events = [(p, e) for p, e in events if p.endswith(".md")]
    assert len(md_events) == 0

def test_watcher_skips_pending_writes(tmp_path):
    events = []
    watcher = BrainWatcher(tmp_path, on_change=lambda p, ev: events.append((p, ev)))
    target = tmp_path / "self.md"
    resolved = str(target.resolve())
    watcher.add_pending_write(resolved)
    watcher.start()
    time.sleep(0.3)
    target.write_text("# Self-written", encoding="utf-8")
    time.sleep(1.0)
    watcher.stop()
    md_events = [(p, e) for p, e in events if "self.md" in p]
    assert len(md_events) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_watcher.py -v`
Expected: FAIL

- [ ] **Step 3: Implement watcher.py**

```python
# brain_mcp/indexer/watcher.py
from __future__ import annotations

import sys
import time
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
    def __init__(self, on_change: Callable[[str, str], None], pending: dict[str, float]):
        self._on_change = on_change
        self._pending = pending

    def _is_pending(self, path: str) -> bool:
        resolved = str(Path(path).resolve())
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

    def on_created(self, event: FileCreatedEvent) -> None:
        if not event.is_directory and event.src_path.endswith(".md"):
            if not self._is_pending(event.src_path):
                self._on_change(event.src_path, "created")

    def on_modified(self, event: FileModifiedEvent) -> None:
        if not event.is_directory and event.src_path.endswith(".md"):
            if not self._is_pending(event.src_path):
                self._on_change(event.src_path, "modified")

    def on_deleted(self, event: FileDeletedEvent) -> None:
        if not event.is_directory and event.src_path.endswith(".md"):
            self._on_change(event.src_path, "deleted")

    def on_moved(self, event: FileMovedEvent) -> None:
        if event.src_path.endswith(".md"):
            self._on_change(event.src_path, "deleted")
        if event.dest_path.endswith(".md"):
            if not self._is_pending(event.dest_path):
                self._on_change(event.dest_path, "created")


class BrainWatcher:
    def __init__(self, vault_path: Path, on_change: Callable[[str, str], None]):
        self._vault_path = vault_path
        self._pending_writes: dict[str, float] = {}
        self._handler = _Handler(on_change, self._pending_writes)
        self._observer = Observer()
        self._observer.daemon = True

    @property
    def is_running(self) -> bool:
        return self._observer.is_alive()

    def add_pending_write(self, resolved_path: str) -> None:
        self._pending_writes[resolved_path] = time.time()

    def start(self) -> None:
        self._observer.schedule(self._handler, str(self._vault_path), recursive=True)
        self._observer.start()
        print(f"Watcher started on {self._vault_path}", file=sys.stderr)

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join(timeout=5)
        print("Watcher stopped.", file=sys.stderr)
```

- [ ] **Step 4: Run tests**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_watcher.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add brain_mcp/indexer/watcher.py tests/test_watcher.py
git commit -m "feat(mcp): file watcher with self-write detection and moved event handling"
```

---

### Task 13: MCP Resources + Server Integration

**Files:**
- Modify: `brain_mcp/server.py`
- Create: `tests/test_integration.py`

This task wires the watcher into the server lifespan, adds the initial vault indexing pipeline, and registers MCP resources.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_integration.py
import json
from pathlib import Path
from brain_mcp.config import BrainConfig
from brain_mcp.storage.database import BrainDB
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.indexer.scanner import scan_vault
from brain_mcp.indexer.scanner import compute_content_hash

def test_full_index_pipeline(tmp_vault, mock_embedder, tmp_path):
    """End-to-end: scan vault → populate DB → embed → search."""
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)

    notes = scan_vault(tmp_vault, folder_to_region={"Projects": 10})
    title_to_id: dict[str, int] = {}
    for note in notes:
        note_id = db.upsert_note(
            path=note["path"], title=note["title"], content=note["content"],
            content_hash=note["content_hash"], region_idx=note["region_idx"],
            tags=note["tags"], word_count=note["word_count"],
            created_at=note["created_at"], modified_at=note["modified_at"],
        )
        title_to_id[note["title"]] = note_id

    for note in notes:
        for bl_title in note.get("backlink_titles", []):
            src_id = title_to_id.get(note["title"])
            tgt_id = title_to_id.get(bl_title)
            if src_id and tgt_id and src_id != tgt_id:
                db.upsert_edge(src_id, tgt_id, link_text=bl_title)

    contents = [n["content"] for n in notes]
    vecs = mock_embedder.embed(contents)
    faiss_ids = vectors.add(vecs)
    for i, note in enumerate(notes):
        nid = title_to_id[note["title"]]
        db.set_faiss_idx(nid, faiss_ids[i])

    assert vectors.size == 3
    assert len(db.get_all_notes()) == 3

    from brain_mcp.tools.retrieve import handle_brain_retrieve
    results = handle_brain_retrieve(db, vectors, mock_embedder, query="routing HTTP", limit=3)
    assert len(results) >= 1

    from brain_mcp.tools.recent import handle_brain_recent
    recent = handle_brain_recent(db, days=365, limit=10)
    assert len(recent) == 3

    from brain_mcp.tools.related import handle_brain_related
    related = handle_brain_related(db, vectors, mock_embedder, title="note1", limit=5)
    assert len(related) >= 1

    db.close()

def test_incremental_reindex(tmp_vault, mock_embedder, tmp_path):
    """Only changed files get re-embedded."""
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)

    notes = scan_vault(tmp_vault, folder_to_region={})
    for note in notes:
        existing_hash = db.get_content_hash(note["path"])
        if existing_hash == note["content_hash"]:
            continue
        note_id = db.upsert_note(
            path=note["path"], title=note["title"], content=note["content"],
            content_hash=note["content_hash"], region_idx=note["region_idx"],
            tags=note["tags"], word_count=note["word_count"],
            created_at=note["created_at"], modified_at=note["modified_at"],
        )
        vec = mock_embedder.embed([note["content"]])
        faiss_ids = vectors.add(vec)
        db.set_faiss_idx(note_id, faiss_ids[0])

    assert vectors.size == 3

    notes2 = scan_vault(tmp_vault, folder_to_region={})
    new_embeds = 0
    for note in notes2:
        existing_hash = db.get_content_hash(note["path"])
        if existing_hash == note["content_hash"]:
            continue
        new_embeds += 1

    assert new_embeds == 0
    db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_integration.py -v`
Expected: FAIL

- [ ] **Step 3: Update server.py with full integration**

Replace the full `brain_mcp/server.py` with the complete version that includes watcher, indexing pipeline, and resources:

```python
# brain_mcp/server.py
from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from brain_mcp.config import BrainConfig, load_config
from brain_mcp.indexer.embedder import SentenceTransformerBackend
from brain_mcp.indexer.scanner import scan_vault
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.indexer.watcher import BrainWatcher
from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.context import handle_brain_context
from brain_mcp.tools.recent import REGION_NAMES, handle_brain_recent
from brain_mcp.tools.regions import handle_brain_regions
from brain_mcp.tools.related import handle_brain_related
from brain_mcp.tools.retrieve import handle_brain_retrieve
from brain_mcp.tools.store import handle_brain_store


@dataclass
class BrainState:
    config: BrainConfig
    db: BrainDB
    vectors: VectorStore
    embedder: SentenceTransformerBackend
    watcher: BrainWatcher | None = None
    _pending_writes: dict[str, float] = field(default_factory=dict)


def _index_vault(state: BrainState) -> None:
    if state.config.vault_path is None or not state.config.vault_path.is_dir():
        return
    notes = scan_vault(state.config.vault_path, state.config.folder_to_region)
    title_to_id: dict[str, int] = {}
    to_embed: list[tuple[int, str]] = []

    for note in notes:
        existing_hash = state.db.get_content_hash(note["path"])
        if existing_hash == note["content_hash"]:
            row = state.db.get_note_by_path(note["path"])
            if row:
                title_to_id[note["title"]] = row["id"]
            continue
        note_id = state.db.upsert_note(
            path=note["path"], title=note["title"], content=note["content"],
            content_hash=note["content_hash"], region_idx=note["region_idx"],
            tags=note["tags"], word_count=note["word_count"],
            created_at=note["created_at"], modified_at=note["modified_at"],
        )
        title_to_id[note["title"]] = note_id
        to_embed.append((note_id, note["content"]))

    if to_embed:
        texts = [t for _, t in to_embed]
        vecs = state.embedder.embed(texts)
        faiss_ids = state.vectors.add(vecs)
        for (note_id, _), fid in zip(to_embed, faiss_ids):
            state.db.set_faiss_idx(note_id, fid)
        print(f"Indexed {len(to_embed)} new/changed notes.", file=sys.stderr)

    for note in notes:
        src_id = title_to_id.get(note["title"])
        if src_id is None:
            continue
        for bl_title in note.get("backlink_titles", []):
            tgt_id = title_to_id.get(bl_title)
            if tgt_id and src_id != tgt_id:
                state.db.upsert_edge(src_id, tgt_id, link_text=bl_title)


def _handle_file_change(state: BrainState, path: str, event_type: str) -> None:
    rel = str(Path(path).relative_to(state.config.vault_path)).replace("\\", "/")
    if event_type == "deleted":
        state.db.delete_note(rel)
        return
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    from brain_mcp.indexer.scanner import compute_content_hash, REGION_TAG_TO_IDX, _BRAIN_TAG_RE, _BACKLINK_RE
    import re
    from datetime import datetime, timezone

    content_hash = compute_content_hash(text)
    if state.db.get_content_hash(rel) == content_hash:
        return

    brain_tags = _BRAIN_TAG_RE.findall(text)
    region_idx = REGION_TAG_TO_IDX.get(brain_tags[0], 9) if brain_tags else 9
    all_tags = list(set(re.findall(r"#[\w/-]+", text)))[:20]
    word_count = len(text.split())
    now = datetime.now(timezone.utc)

    note_id = state.db.upsert_note(
        path=rel, title=Path(path).stem, content=text, content_hash=content_hash,
        region_idx=region_idx, tags=all_tags, word_count=word_count,
        created_at=now.strftime("%Y-%m-%d"), modified_at=now.isoformat(),
    )
    vec = state.embedder.embed([text])
    faiss_ids = state.vectors.add(vec)
    state.db.set_faiss_idx(note_id, faiss_ids[0])
    print(f"Re-indexed: {rel}", file=sys.stderr)


@asynccontextmanager
async def brain_lifespan(server: FastMCP) -> AsyncIterator[BrainState]:
    config = load_config()
    config.data_dir.mkdir(parents=True, exist_ok=True)

    db = BrainDB(config.db_path)
    vectors = VectorStore.load(config.index_path, dimension=384)
    embedder = SentenceTransformerBackend(config.model_name)

    state = BrainState(config=config, db=db, vectors=vectors, embedder=embedder)

    if config.vault_path and config.vault_path.is_dir() and config.index_on_startup:
        _index_vault(state)

    watcher = None
    if config.vault_path and config.vault_path.is_dir() and config.auto_index:
        watcher = BrainWatcher(
            config.vault_path,
            on_change=lambda p, ev: _handle_file_change(state, p, ev),
        )
        watcher.start()
        state.watcher = watcher

    print(f"Brain MCP Server started. Vault: {config.vault_path}", file=sys.stderr)
    print(f"DB: {config.db_path} | Index: {vectors.size} vectors", file=sys.stderr)

    try:
        yield state
    finally:
        if watcher:
            watcher.stop()
        vectors.save(config.index_path)
        db.close()
        print("Brain MCP Server stopped.", file=sys.stderr)


mcp = FastMCP("Neural Brain", lifespan=brain_lifespan)


@mcp.tool()
def brain_recent(days: int = 7, region: str | None = None, limit: int = 20) -> list[dict]:
    """Show recently modified notes in the vault.

    Args:
        days: Lookback window in days (default 7, max 365)
        region: Filter by brain region name (e.g. "Hippocampus")
        limit: Max results (default 20, max 100)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    return handle_brain_recent(state.db, days=days, region=region, limit=limit)


@mcp.tool()
def brain_regions(action: str, region: str | None = None, description: str | None = None, color: str | None = None) -> dict | list[dict]:
    """List, describe, or customize brain region definitions.

    Args:
        action: "list" (all regions), "describe" (one region detail), or "customize" (update region)
        region: Region name (required for describe/customize)
        description: New description (for customize)
        color: New hex color (for customize)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    return handle_brain_regions(state.db, action=action, region=region, description=description, color=color)


@mcp.tool()
def brain_retrieve(query: str, region: str | None = None, limit: int = 10, threshold: float = 0.3) -> list[dict]:
    """Semantic search across all vault notes. Finds notes by meaning, not just keywords.

    Args:
        query: Natural language search query
        region: Filter by brain region name (e.g. "Hippocampus")
        limit: Max results (default 10, max 100)
        threshold: Min cosine similarity (default 0.3)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    return handle_brain_retrieve(state.db, state.vectors, state.embedder, query=query, region=region, limit=limit, threshold=threshold)


@mcp.tool()
def brain_store(title: str, content: str, region: str | None = None, region_idx: int | None = None,
                tags: list[str] | None = None, folder: str = "") -> dict:
    """Create or update a note in the vault.

    Args:
        title: Note title (becomes filename)
        content: Markdown content
        region: Brain region name (auto-detected if omitted)
        region_idx: Region index (overrides name)
        tags: Additional tags (max 20)
        folder: Subfolder in vault
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    if state.config.vault_path is None:
        return {"error": "No vault_path configured"}
    pw = state._pending_writes
    if state.watcher:
        pw = state.watcher._handler._pending
    return handle_brain_store(
        state.db, state.vectors, state.embedder, state.config.vault_path,
        title=title, content=content, region=region, region_idx=region_idx,
        tags=tags, folder=folder, pending_writes=pw,
    )


@mcp.tool()
def brain_related(title: str | None = None, path: str | None = None, limit: int = 10) -> list[dict] | dict:
    """Find notes related to a specific note by meaning or backlinks.

    Args:
        title: Note title to find relations for
        path: Note path (alternative to title)
        limit: Max results (default 10, max 100)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    return handle_brain_related(state.db, state.vectors, state.embedder, title=title, path=path, limit=limit)


@mcp.tool()
def brain_context(file_paths: list[str] | None = None, task_description: str | None = None,
                  depth: int = 1, max_notes: int = 10) -> list[dict] | dict:
    """Get contextually relevant notes for current work. Combines file proximity and semantic search.

    Args:
        file_paths: Files currently being edited
        task_description: What you are doing
        depth: Backlink graph hops 1-3 (default 1)
        max_notes: Max notes returned (default 10, max 100)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    return handle_brain_context(state.db, state.vectors, state.embedder,
                                 file_paths=file_paths, task_description=task_description,
                                 depth=depth, max_notes=max_notes)


@mcp.resource("brain://regions")
def resource_regions() -> str:
    """All 12 brain regions with note counts."""
    state: BrainState = mcp.get_context().request_context.lifespan_context
    result = handle_brain_regions(state.db, action="list")
    return json.dumps(result, indent=2)


@mcp.resource("brain://recent")
def resource_recent() -> str:
    """Last 20 modified notes."""
    state: BrainState = mcp.get_context().request_context.lifespan_context
    result = handle_brain_recent(state.db, days=7, limit=20)
    return json.dumps(result, indent=2)


@mcp.resource("brain://stats")
def resource_stats() -> str:
    """Vault statistics."""
    state: BrainState = mcp.get_context().request_context.lifespan_context
    notes = state.db.get_all_notes()
    counts = state.db.get_region_note_counts()
    return json.dumps({
        "total_notes": len(notes),
        "total_vectors": state.vectors.size,
        "regions_with_notes": len(counts),
        "vault_path": str(state.config.vault_path),
    }, indent=2)
```

- [ ] **Step 4: Run tests**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/test_integration.py -v`
Expected: 2 PASSED

- [ ] **Step 5: Run full test suite**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m pytest tests/ -v --ignore=tests/test_camera.py --ignore=tests/test_picking.py --ignore=tests/test_physics.py`
Expected: All PASSED

- [ ] **Step 6: Commit**

```bash
git add brain_mcp/server.py tests/test_integration.py
git commit -m "feat(mcp): full server integration with watcher, indexing pipeline, and MCP resources"
```

---

### Task 14: CLI Index Command

**Files:**
- Modify: `brain_mcp/__main__.py`

- [ ] **Step 1: Update __main__.py with index command**

```python
# brain_mcp/__main__.py
import argparse
import sys
import time


def cmd_index(args):
    from pathlib import Path
    from brain_mcp.config import load_config
    from brain_mcp.storage.database import BrainDB
    from brain_mcp.indexer.vector_store import VectorStore
    from brain_mcp.indexer.embedder import SentenceTransformerBackend
    from brain_mcp.indexer.scanner import scan_vault

    config = load_config()
    if args.vault:
        config.vault_path = Path(args.vault)
    if config.vault_path is None or not config.vault_path.is_dir():
        print(f"ERROR: vault_path not set or not a directory: {config.vault_path}", file=sys.stderr)
        sys.exit(1)

    config.data_dir.mkdir(parents=True, exist_ok=True)
    db = BrainDB(config.db_path)
    vectors = VectorStore(dimension=384) if args.force else VectorStore.load(config.index_path, dimension=384)
    embedder = SentenceTransformerBackend(config.model_name)

    print(f"Scanning vault: {config.vault_path}", file=sys.stderr)
    notes = scan_vault(config.vault_path, config.folder_to_region)
    print(f"Found {len(notes)} notes.", file=sys.stderr)

    title_to_id: dict[str, int] = {}
    to_embed: list[tuple[int, str]] = []
    for note in notes:
        if not args.force:
            existing_hash = db.get_content_hash(note["path"])
            if existing_hash == note["content_hash"]:
                row = db.get_note_by_path(note["path"])
                if row:
                    title_to_id[note["title"]] = row["id"]
                continue
        note_id = db.upsert_note(
            path=note["path"], title=note["title"], content=note["content"],
            content_hash=note["content_hash"], region_idx=note["region_idx"],
            tags=note["tags"], word_count=note["word_count"],
            created_at=note["created_at"], modified_at=note["modified_at"],
        )
        title_to_id[note["title"]] = note_id
        to_embed.append((note_id, note["content"]))

    if to_embed:
        print(f"Embedding {len(to_embed)} notes...", file=sys.stderr)
        t0 = time.time()
        texts = [t for _, t in to_embed]
        vecs = embedder.embed(texts)
        faiss_ids = vectors.add(vecs)
        for (note_id, _), fid in zip(to_embed, faiss_ids):
            db.set_faiss_idx(note_id, fid)
        elapsed = time.time() - t0
        print(f"Embedded in {elapsed:.1f}s", file=sys.stderr)
    else:
        print("No new/changed notes to embed.", file=sys.stderr)

    for note in notes:
        src_id = title_to_id.get(note["title"])
        if src_id is None:
            continue
        for bl_title in note.get("backlink_titles", []):
            tgt_id = title_to_id.get(bl_title)
            if tgt_id and src_id != tgt_id:
                db.upsert_edge(src_id, tgt_id, link_text=bl_title)

    vectors.save(config.index_path)
    db.close()
    print(f"Done. Index: {config.index_path} | DB: {config.db_path}", file=sys.stderr)


def cmd_serve(args):
    from brain_mcp.server import mcp
    mcp.run(transport="stdio")


def main():
    parser = argparse.ArgumentParser(prog="brain_mcp", description="Neural Brain MCP Server")
    sub = parser.add_subparsers(dest="command")

    serve_p = sub.add_parser("serve", help="Start MCP server (default)")
    index_p = sub.add_parser("index", help="Build/update vault index")
    index_p.add_argument("--vault", type=str, help="Vault directory path")
    index_p.add_argument("--force", action="store_true", help="Re-embed all notes")

    args = parser.parse_args()
    if args.command == "index":
        cmd_index(args)
    else:
        cmd_serve(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test CLI help**

Run: `cd "C:/Users/lucas/Desktop/ALLES ÜBER CLAUDE/Claude Stuff/Projekte/neural-brain" && python -m brain_mcp --help && python -m brain_mcp index --help`
Expected: Help output showing serve and index commands

- [ ] **Step 3: Commit**

```bash
git add brain_mcp/__main__.py
git commit -m "feat(mcp): CLI index command for pre-building vault index"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Section 1 (Overview): covered by all tasks
- [x] Section 2 (Architecture): server.py, lifespan, components
- [x] Section 3 (12 Brain Regions): migrations.py seeds defaults, regions.py customize
- [x] Section 4.1 (brain_retrieve): Task 8
- [x] Section 4.2 (brain_context): Task 11
- [x] Section 4.3 (brain_store): Task 9
- [x] Section 4.4 (brain_related): Task 10
- [x] Section 4.5 (brain_recent): Task 6
- [x] Section 4.6 (brain_regions): Task 7
- [x] Section 5 (MCP Resources): Task 13
- [x] Section 6 (Storage): Task 2
- [x] Section 7 (Embedding System): Task 4
- [x] Section 8 (File Watcher): Task 12
- [x] Section 9 (Startup Flow): Task 13 + 14
- [x] Section 10 (Error Handling): Distributed across tools
- [x] Section 11 (Security): Task 9
- [x] Section 12 (Configuration): Task 1
- [x] Section 13 (File Structure): All tasks create the correct files
- [x] Section 14 (Dependencies): Task 1 pyproject.toml

**Placeholder scan:** No TBD/TODO found.

**Type consistency:** All imports, function signatures, and class names verified consistent across tasks.
