# Changelog

## v1.3.3 — Multi-client stability (2026-06-08)

Fixes the random `Connection closed` / "MCP stopped working" disconnects and makes
multiple GYSTC servers (Claude CLI + Hermes + Command Center) coexist safely on one vault.

### Fixed — disconnects & process lifecycle
- **Removed the global `_kill_zombie_siblings` killer.** stdio MCP runs one server per
  client; the old startup routine killed *every* other `brain_mcp serve` process, so each
  new/restarted instance disconnected the others at random. This was the root cause of the
  recurring disconnects.
- **Parent-death watchdog** (`process_guard.py`): a server now self-terminates when the
  client that spawned it is gone, instead of lingering as an orphan holding `brain.db`.
- **Prompt, clean exit** on client disconnect (`__main__.py` / `_shutdown`), so a server
  can't linger mid-embed.
- **Single-winner, idempotent shutdown**: the lifespan teardown and the watchdog can no
  longer race on `save()`/`close()`; the loser waits for the winner to finish before any
  `os._exit`, so an in-flight re-index is drained and the final index save isn't dropped.

### Fixed — multi-writer index safety
- **Single-writer file lock** (`file_lock.py`): exactly one instance owns `index.faiss` and
  the `faiss_idx` column. Other instances run read-only (fully usable via FTS + the shared
  DB + on-disk index) and never persist the index — preventing the silent FAISS clobber /
  `faiss_idx` desync that coexisting writers would cause.
- **Reader→writer promotion**: a read-only instance takes over indexing if the writer exits,
  so reader writes don't pile up un-indexed.
- Read-only `brain_store`/rollback write the `.md` file (source of truth) only; the writer's
  watcher indexes it. The writer always runs an incremental reconcile scan on startup.

### Fixed — indexing correctness
- **Debounced, serial re-index worker** (`reindex_queue.py`): file changes are re-indexed
  off the watchdog observer thread, coalesced per path, with a starvation cap so an
  actively-edited note is never deferred forever.
- **Watcher path now refreshes chunk vectors** (and removes them on delete) — previously it
  updated only the note-level vector, so edited notes returned stale search snippets and
  leaked orphaned chunk vectors.
- Re-embeds notes whose DB row exists but has no vector (`faiss_idx IS NULL`).

### Packaging
- `requirements.txt` now installs a runnable server (`-e .[dashboard]`); it previously
  listed only GUI deps and omitted the core MCP runtime dependencies. pyproject.toml remains
  the single source of truth.

### Tests
- 220 passing (+31): process guard, file lock, reindex worker, shutdown idempotency,
  reader/writer split, and watcher chunk-refresh.
