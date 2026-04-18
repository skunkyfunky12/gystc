# Brain MCP Server — Design Spec

**Date:** 2026-04-18
**Status:** Approved
**Goal:** Turn the Obsidian vault into Claude's semantic working memory via an MCP Server with retrieval, storage, and live indexing.

---

## 1. Overview

The Brain MCP Server is a standalone stdio MCP server that provides Claude with 6 tools to read, write, search, and navigate a vault of markdown files organized into 12 brain regions. It maintains a FAISS vector index for semantic search, a SQLite database for metadata, and a watchdog file watcher for live re-indexing.

**Key properties:**
- No Obsidian dependency — vault is any folder of `.md` files
- Single process, lazy model loading, live file watching
- Installable via `pip install neural-brain`, one `mcp.json` entry
- Review-hardened: path traversal guards, atomic writes, thread-safe FAISS, FTS5 snippets

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Brain MCP Server                     │
│                  (stdio, single process)              │
│                                                       │
│  ┌────────────┐  ┌─────────────┐  ┌───────────────┐  │
│  │   Tools     │  │   Indexer   │  │   Watcher     │  │
│  │ (6 tools)   │──│  Embedder   │  │ (watchdog)    │  │
│  │ + Resources │  │  Scanner    │──│ Background    │  │
│  └─────┬───────┘  └──────┬──────┘  └───────────────┘  │
│        │                 │                             │
│  ┌─────┴─────────────────┴───────┐                    │
│  │        Storage Layer          │                    │
│  │  SQLite + FTS5 + FAISS Index  │                    │
│  │  threading.RLock protection   │                    │
│  └───────────────────────────────┘                    │
└──────────────────────────────────────────────────────┘
          │                              │
     ~/.neural-brain/               ~/vault/
     brain.db + index.faiss         (any .md folder)
```

### Components

| Component | Responsibility |
|-----------|---------------|
| **server.py** | MCP protocol handler, tool dispatch, resource endpoints, progress notifications |
| **tools/** | 6 tool modules — each validates input, calls storage/indexer, formats output |
| **indexer/scanner.py** | Vault scan, file hash tracking, backlink extraction, tag parsing |
| **indexer/embedder.py** | sentence-transformers wrapper, lazy model load, batch encoding, swappable interface |
| **indexer/vector_store.py** | FAISS IndexFlatIP management, L2 normalization, RLock-protected read/write |
| **indexer/watcher.py** | watchdog observer, self-write detection, FileMovedEvent handling (Windows) |
| **storage/database.py** | SQLite operations, FTS5 virtual table, parameterized queries |
| **storage/migrations.py** | Schema versioning for upgrades |
| **config.py** | Config loading, env var fallbacks, path validation |

---

## 3. The 12 Brain Regions

Default regions ship with the server. New users customize via `brain_regions` tool.

| Idx | Name | Color | Function |
|-----|------|-------|----------|
| 0 | Präfrontaler Cortex | #3498DB | Architecture, decisions, planning |
| 1 | Motorischer Cortex | #E74C3C | API writes, actions, execution |
| 2 | Sensorischer Cortex | #2ECC71 | Data intake, references, input |
| 3 | Hippocampus | #F39C12 | Memory, sessions, personal notes |
| 4 | Kleinhirn | #9B59B6 | Precision algorithms |
| 5 | Nucleus Accumbens | #1ABC9C | Subscriptions, pricing, rewards |
| 6 | Broca-Areal | #E67E22 | AI, prompts, agents, language |
| 7 | Visueller Cortex | #8E44AD | UI, themes, design |
| 8 | Thalamus | #16A085 | Index, MOC, data relay |
| 9 | Stammhirn | #95A5A6 | Config, infrastructure |
| 10 | Basalganglien | #D35400 | Pipelines, ETL, background |
| 11 | Amygdala | #C0392B | Auth, team, social interaction |

Region assignment priority:
1. Explicit `#brain/<region-slug>` tag in note content
2. Folder-to-region mapping (configurable in config.json)
3. Default: Stammhirn (idx 9)

---

## 4. Tool Specifications

### 4.1 brain_retrieve — Semantic Search

**Purpose:** Find notes by meaning, not just keywords.

**Input:**
| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| query | string | yes | — | Natural language search query |
| region | string | no | — | Filter by brain region name |
| limit | int | no | 10 | Max results |
| threshold | float | no | 0.3 | Min cosine similarity |

**Logic:**
1. Embed query with sentence-transformers
2. `faiss.normalize_L2()` on query vector
3. Acquire RLock → FAISS `search(k=limit)` → release
4. Filter by threshold
5. If region specified: filter by region_idx from SQLite
6. Join with SQLite for metadata
7. FTS5 snippet extraction for matched passages

**Output:**
```json
[{
  "title": "CSP Middleware",
  "path": "02 Projekte/D2D-Scout/graphify/CSP Middleware.md",
  "region": "Präfrontaler Cortex",
  "region_idx": 0,
  "similarity": 0.87,
  "snippet": "Die CSP Middleware setzt Content-Security-Policy Headers für alle Responses...",
  "tags": ["#brain/praefrontaler-cortex", "#middleware"],
  "created": "2026-03-15",
  "modified": "2026-04-10",
  "word_count": 342
}]
```

### 4.2 brain_context — Context for Current Work

**Purpose:** Get relevant notes for files currently being worked on.

**Input:**
| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| file_paths | string[] | no | — | Files currently being edited |
| task_description | string | no | — | What the user is doing |
| depth | int | no | 1 | Backlink graph hops (1-3) |
| max_notes | int | no | 10 | Max notes returned |

**Logic:**
1. If `file_paths`: match vault notes by filename/stem
2. Get their backlinks (up to `depth` hops) from `edges` table
3. If `task_description`: semantic search, weighted blend
4. Scoring: semantic similarity × 0.6 + backlink proximity × 0.4
5. Deduplicate and rank, cap at `max_notes`

**Output:** Same as brain_retrieve, plus:
```json
{ "relevance_reason": "backlink from Router Setup (1 hop)" }
```

### 4.3 brain_store — Create/Update Notes

**Purpose:** Write knowledge to the vault.

**Input:**
| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| title | string | yes | — | Note title (becomes filename) |
| content | string | yes | — | Markdown content |
| region | string | no | auto-detect | Brain region name |
| region_idx | int | no | — | Region index (overrides name) |
| tags | string[] | no | [] | Additional tags |
| folder | string | no | "" | Subfolder in vault |

**Security:**
1. Sanitize title: strip `../`, `\`, control chars, limit to 200 chars
2. Enforce `.md` extension — reject all other extensions
3. Resolve final path, assert `Path.resolve().is_relative_to(vault_root)`
4. Atomic write: write to `.tmp` file, then `os.replace()` to final path
5. Add path to `_pending_writes` set (self-write detection, 2s TTL)

**Logic:**
1. Validate inputs (path traversal guard)
2. Add `#brain/<region>` tag to content if region specified
3. Atomic write `.md` file to vault
4. Embed content → `faiss.normalize_L2()` → add to FAISS (under RLock)
5. Upsert metadata in SQLite
6. Return path + region assignment

**Output:**
```json
{
  "path": "02 Projekte/new-note.md",
  "region": "Präfrontaler Cortex",
  "region_idx": 0,
  "indexed": true,
  "word_count": 156
}
```

### 4.4 brain_related — Find Related Notes

**Purpose:** Discover notes connected by meaning or backlinks.

**Input:**
| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| title | string | no* | — | Note title to find relations for |
| path | string | no* | — | Note path (alternative to title) |
| limit | int | no | 10 | Max results |

*One of title or path is required.

**Logic:**
1. Find note in SQLite by title or path
2. Get embedding → FAISS nearest neighbors (semantic)
3. Get backlinks from `edges` table (graph)
4. Merge: semantic_score × 0.6 + graph_proximity × 0.4
5. Deduplicate, exclude the source note

**Output:** Same format as brain_retrieve, plus:
```json
{ "relation_type": "semantic" | "backlink" | "both" }
```

### 4.5 brain_recent — Recently Modified Notes

**Purpose:** Show recent vault activity. No embedding model needed.

**Input:**
| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| days | int | no | 7 | Lookback window |
| region | string | no | — | Filter by region |
| limit | int | no | 20 | Max results |

**Logic:** Pure SQLite query on `modified_at`, filtered by region_idx if given. Ordered by modified_at DESC.

**Output:**
```json
[{
  "title": "Router Setup",
  "path": "02 Projekte/D2D-Scout/graphify/Router Setup.md",
  "region": "Hippocampus",
  "modified_at": "2026-04-18T14:30:00Z",
  "word_count": 523
}]
```

### 4.6 brain_regions — Manage Brain Regions

**Purpose:** List, describe, and customize brain region definitions. Essential for product — new users need to define their Claude's brain structure.

**Input:**
| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| action | string | yes | — | "list", "describe", or "customize" |
| region | string | no | — | Region name (for describe/customize) |
| description | string | no | — | New description (for customize) |
| color | string | no | — | New hex color (for customize) |

**Logic:**
- `list`: Query all 12 regions from SQLite `regions` table, include note counts via JOIN
- `describe`: Single region detail + top 10 notes by word_count + connected regions via edges
- `customize`: UPDATE region in SQLite + update config.json for persistence across restarts

**Output (list):**
```json
[{
  "idx": 0,
  "name": "Präfrontaler Cortex",
  "color": "#3498DB",
  "description": "Architecture, decisions, planning",
  "note_count": 58,
  "position": [0.0, 110.0, -145.0]
}]
```

---

## 5. MCP Resources

In addition to tools, the server exposes read-only MCP Resources for browsing:

| URI | Description |
|-----|-------------|
| `brain://regions` | All 12 regions with note counts |
| `brain://region/{name}` | Notes in a specific region |
| `brain://recent` | Last 20 modified notes |
| `brain://stats` | Vault statistics (total notes, edges, index freshness) |

Resources are application-driven (Claude Code UI), tools are model-driven (Claude queries).

---

## 6. Storage

### 6.1 SQLite Schema (`~/.neural-brain/brain.db`)

```sql
-- Core note metadata
CREATE TABLE notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT UNIQUE NOT NULL,
    title       TEXT NOT NULL,
    content     TEXT DEFAULT '',            -- raw markdown for FTS5 snippets
    content_hash TEXT NOT NULL,
    region_idx  INTEGER NOT NULL DEFAULT 9,
    tags        TEXT DEFAULT '[]',       -- JSON array
    word_count  INTEGER DEFAULT 0,
    created_at  TEXT,
    modified_at TEXT,
    embedded_at TEXT,
    faiss_idx   INTEGER                  -- position in FAISS index
);

-- Full-text search on note content
CREATE VIRTUAL TABLE notes_fts USING fts5(
    title,
    content,
    content='notes',
    content_rowid='id',
    tokenize='unicode61'
);

-- Brain region definitions (customizable)
CREATE TABLE regions (
    idx         INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    color       TEXT NOT NULL,            -- hex color
    description TEXT DEFAULT '',
    position    TEXT DEFAULT '[0,0,0]'    -- JSON [x,y,z]
);

-- Backlink edges between notes
CREATE TABLE edges (
    source_id   INTEGER REFERENCES notes(id) ON DELETE CASCADE,
    target_id   INTEGER REFERENCES notes(id) ON DELETE CASCADE,
    link_text   TEXT DEFAULT '',           -- [[Note|Display Text]] display part
    PRIMARY KEY (source_id, target_id)
);

-- Schema version tracking
CREATE TABLE schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL
);

CREATE INDEX idx_notes_region ON notes(region_idx);
CREATE INDEX idx_notes_modified ON notes(modified_at);
CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
```

### 6.2 FAISS Index (`~/.neural-brain/index.faiss`)

- **Type:** IndexFlatIP (inner product on L2-normalized vectors = cosine similarity)
- **Dimensions:** 384 (MiniLM output)
- **Normalization:** All vectors L2-normalized before `add()` and `search()`
- **Concurrency:** `threading.RLock` wraps all `add()`, `remove()`, `search()`, `write_index()`, `read_index()` calls
- **ID mapping:** `notes.faiss_idx` column maps SQLite rows to FAISS positions
- **Scale:** IndexFlatIP is optimal up to ~100k notes. At 5000 notes: ~7.3MB RAM, <5ms search.

### 6.3 File Layout

```
~/.neural-brain/
├── config.json          # vault_path, model_name, regions overrides
├── brain.db             # SQLite database
├── index.faiss          # FAISS vector index
└── .gitignore           # Exclude from sync (embeddings contain note content)
```

---

## 7. Embedding System

### 7.1 Model

- **Default:** `paraphrase-multilingual-MiniLM-L12-v2`
- **Dimensions:** 384
- **Size:** ~120MB download, ~500MB RAM when loaded
- **Language:** Multilingual (German vault content works natively)
- **Loading:** Lazy — first semantic query triggers load. Progress notifications sent via MCP.

### 7.2 Swappable Interface

```python
class EmbeddingBackend(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts, return (n, 384) float32 array, L2-normalized."""
        ...

    @property
    def dimension(self) -> int: ...
```

**Implementations:**
- `SentenceTransformerBackend` — default, local model
- `OllamaBackend` — future: connect to local Ollama instance

Config selects backend:
```json
{
  "embedding_backend": "sentence-transformers",
  "model_name": "paraphrase-multilingual-MiniLM-L12-v2"
}
```

### 7.3 Batch Processing

Initial vault indexing uses batch encoding:
```python
model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
```

At 555 notes: ~28 seconds initial index build.
At 5000 notes: ~250 seconds (use `python -m brain_mcp index` pre-build).

---

## 8. File Watcher

### 8.1 watchdog Observer

- Runs as daemon thread in the MCP server process
- Watches vault directory recursively for `.md` file changes
- Events: `FileCreatedEvent`, `FileModifiedEvent`, `FileDeletedEvent`, `FileMovedEvent`

### 8.2 Self-Write Detection

When `brain_store` writes a file:
1. Add path to `_pending_writes: dict[str, float]` with `time.time()` timestamp
2. Watcher checks `_pending_writes` before processing an event
3. If path in set and age < 2 seconds → skip (it's our own write)
4. Cleanup: remove entries older than 5 seconds periodically

### 8.3 Windows Considerations

- `watchdog` uses `ReadDirectoryChangesW` on Windows
- `FileMovedEvent` fired for renames (not delete+create)
- Handler: treat `src_path` as delete, `dest_path` as create
- Symlinks: resolve all paths before comparing

### 8.4 Re-indexing Pipeline

```
File event → filter (.md only) → self-write check → read content →
compute hash → compare with DB → if changed: embed → update FAISS (RLock) →
update SQLite → update FTS5
```

---

## 9. Startup Flow

### 9.1 Normal Start (warm index)

1. Load config from `~/.neural-brain/config.json` (or env vars)
2. Validate vault_path exists
3. Open SQLite DB, run pending migrations
4. Load FAISS index from disk
5. Start watchdog observer
6. Enter MCP stdio loop
7. **Model loads lazily on first semantic query**

Target: <2 seconds to first tool response.

### 9.2 Pre-Build Command

```bash
python -m brain_mcp index [--vault PATH] [--force]
```

- Scans entire vault
- Loads model, embeds all notes in batch
- Writes brain.db + index.faiss
- Progress bar on stderr
- `--force`: re-embed everything (ignore hashes)

Use case: first install, or after major vault reorganization.

### 9.3 Cold Start (no pre-build)

1. Steps 1-5 same as normal
2. First semantic query triggers:
   a. Model download + load (5-10s, progress notifications)
   b. Scan vault, embed all notes (28s for 555 notes)
   c. Write index to disk
3. Subsequent queries: <100ms

---

## 10. Error Handling

| Scenario | Behavior |
|----------|----------|
| Model load fails | `brain_recent` + `brain_regions` still work. Semantic tools return clear error: "Embedding model not available" |
| Vault path invalid | Server starts, all tools return: "Vault not found at {path}. Set BRAIN_VAULT_PATH or update config.json" |
| SQLite locked | Retry with exponential backoff (100ms, 200ms, 400ms). Max 3 attempts, then error. |
| Watcher crash | Log to stderr, fall back to hash-check-on-query (graceful degradation) |
| Single note embed fails | Skip note, log warning, continue with rest |
| FAISS index corrupt | Delete and rebuild from SQLite content_hash records |
| Disk full | Atomic write fails at rename step, original file preserved |

All errors logged to **stderr only** (stdout is JSON-RPC).

---

## 11. Security

### 11.1 Path Traversal Prevention

Every file operation validates:
```python
resolved = Path(vault_root, folder, f"{sanitized_title}.md").resolve()
if not resolved.is_relative_to(Path(vault_root).resolve()):
    raise ValueError("Path escapes vault directory")
```

Title sanitization: strip `../`, `\`, `:`, `*`, `?`, `"`, `<`, `>`, `|`, control chars. Max 200 chars.

### 11.2 File Extension Whitelist

Only `.md` files are read, written, or indexed. All other extensions rejected.

### 11.3 Atomic Writes

```python
tmp = target.with_suffix('.md.tmp')
tmp.write_text(content, encoding='utf-8')
os.replace(str(tmp), str(target))  # atomic on same filesystem
```

### 11.4 SQL Injection Prevention

All SQLite queries use parameterized statements. No string interpolation in SQL.

### 11.5 Input Size Limits

| Input | Limit |
|-------|-------|
| title | 200 chars |
| content | 1MB |
| tags array | 20 items, 100 chars each |
| query | 1000 chars |
| limit param | max 100 |
| depth param | max 3 |

### 11.6 Config File Security

- `~/.neural-brain/config.json` should be owner-readable only
- Document: `chmod 600` (Linux/Mac), `icacls` owner-only (Windows)
- Index directory excluded from sync tools (embeddings contain note content)

---

## 12. Configuration

### 12.1 Config File (`~/.neural-brain/config.json`)

```json
{
    "vault_path": "/path/to/vault",
    "embedding_backend": "sentence-transformers",
    "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
    "auto_index": true,
    "index_on_startup": true,
    "folder_to_region": {
        "00 Index": 8,
        "01 Lucas": 3,
        "02 Projekte": 0,
        "03 Agenten": 6,
        "04 Sessions": 3,
        "05 Referenzen": 2
    }
}
```

### 12.2 Environment Variables (override config)

| Variable | Description |
|----------|-------------|
| `BRAIN_VAULT_PATH` | Vault directory path |
| `BRAIN_MODEL_NAME` | Override embedding model |
| `BRAIN_DATA_DIR` | Override `~/.neural-brain/` location |
| `BRAIN_LOG_LEVEL` | Logging verbosity (DEBUG/INFO/WARNING) |

### 12.3 MCP Registration (`~/.claude/mcp.json`)

```json
{
    "mcpServers": {
        "brain": {
            "command": "python",
            "args": ["-m", "brain_mcp"],
            "env": {
                "BRAIN_VAULT_PATH": "/path/to/vault"
            }
        }
    }
}
```

---

## 13. File Structure

```
neural-brain/
├── brain_mcp/
│   ├── __init__.py
│   ├── __main__.py               # CLI: serve (default) | index | version
│   ├── server.py                 # MCP protocol, tool dispatch, resources, progress
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── retrieve.py           # brain_retrieve
│   │   ├── context.py            # brain_context
│   │   ├── store.py              # brain_store
│   │   ├── related.py            # brain_related
│   │   ├── recent.py             # brain_recent
│   │   └── regions.py            # brain_regions
│   ├── indexer/
│   │   ├── __init__.py
│   │   ├── scanner.py            # Vault scan, hash tracking, backlink/tag extraction
│   │   ├── embedder.py           # EmbeddingBackend protocol + SentenceTransformerBackend
│   │   ├── vector_store.py       # FAISS index with RLock, normalize, add/search/remove
│   │   └── watcher.py            # watchdog observer, self-write detection, Windows compat
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── database.py           # SQLite operations, FTS5, parameterized queries
│   │   └── migrations.py         # Schema versioning
│   └── config.py                 # Config loading, env var fallbacks, validation
├── tests/
│   ├── conftest.py               # Fixtures: temp vault, test DB, mock embedder
│   ├── test_scanner.py
│   ├── test_embedder.py
│   ├── test_vector_store.py
│   ├── test_watcher.py
│   ├── test_database.py
│   ├── test_tools_retrieve.py
│   ├── test_tools_store.py
│   ├── test_tools_context.py
│   ├── test_tools_related.py
│   ├── test_tools_recent.py
│   ├── test_tools_regions.py
│   ├── test_security.py          # Path traversal, input limits, injection tests
│   └── test_integration.py       # End-to-end MCP protocol tests
├── pyproject.toml
├── README.md
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-04-18-brain-mcp-server-design.md  (this file)
```

---

## 14. Dependencies

```toml
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
    "pytest-asyncio",
]
```

---

## 15. Integration with Existing Dashboard

The Brain MCP Server and the Neural Brain Dashboard share:
- `~/.neural-brain/config.json` — same config file
- `data/regions.py` — same 12 region definitions (extracted to shared module)
- Vault data — dashboard reads the same `.md` files for visualization

The dashboard remains a PyQt6 app. The MCP server is a separate process. They don't communicate directly — they share the vault as source of truth.

---

## 16. Product Setup (New User)

```bash
# 1. Install
pip install neural-brain

# 2. Pre-build index (optional but recommended)
python -m brain_mcp index --vault ~/my-notes

# 3. Register with Claude Code
# Add to ~/.claude/mcp.json:
{
    "mcpServers": {
        "brain": {
            "command": "python",
            "args": ["-m", "brain_mcp"],
            "env": { "BRAIN_VAULT_PATH": "~/my-notes" }
        }
    }
}

# 4. Done. Claude now has brain_retrieve, brain_store, etc.
# New user can customize regions:
# Claude: brain_regions(action="customize", region="Präfrontaler Cortex", description="My planning notes")
```

No Obsidian. No extra daemon. No cron job.
