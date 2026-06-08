# Changelog

## v1.3.5 — Daemon idle-shutdown + security hardening (2026-06-08)

### Added
- **Daemon idle-shutdown**: the shared daemon now self-terminates after `--idle` seconds (default 1800)
  with no authorized request, freeing the embedding model + index from memory. The proxy transparently
  respawns it on the next call, so this is invisible to clients. `--idle 0` keeps it resident.

### Security (hardening from a full audit + semgrep; all defense-in-depth, no critical/high)
- `is_alive` now fails **closed** on a pid-check error instead of probing (and leaking the bearer token
  to) a possibly-reused port.
- `/mcp` request bodies are capped at 8 MB (413) so a token-holder can't OOM the shared daemon — parity
  with the GUI `/api`.
- Network-facing dependencies gained upper bounds (`mcp<2`, `httpx<1`, …) against silent / supply-chain
  major upgrades; distributed binaries remain the CI-frozen PyInstaller build.

## v1.3.4 — Shared daemon: load the model once (2026-06-08)

All MCP clients (Claude CLI, Hermes, Command Center) now share ONE long-lived daemon that loads
the embedding model + index once, instead of each client spawning its own ~400 MB server.

### Added
- **Shared daemon** (`brain_mcp daemon`): the GYSTC server over streamable-HTTP on 127.0.0.1,
  secured with a per-run bearer token + Origin allow-list (DNS-rebinding protection) and a held
  ephemeral port (no port TOCTOU / token leak). Single-instance via a lock file.
- **`brain_mcp serve` is now a thin stdio↔HTTP proxy**: it auto-starts the daemon if needed and
  forwards JSON-RPC to it. The model is loaded once for the whole machine — Hermes no longer pays
  a ~6 s cold start per run. If the daemon dies mid-session the proxy respawns it and never hangs
  the client (returns a JSON-RPC error as a last resort).
- **`serve --direct`**: escape hatch that runs the previous in-process stdio server unchanged.

### Notes
- Deferred to a follow-up: daemon idle-shutdown, a logon autostart task for warm starts, and
  retiring the now-redundant per-process writer-lock. The daemon stays resident once started.
- 239 tests passing; verified end-to-end, including a mid-session daemon kill (client never hangs).

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
