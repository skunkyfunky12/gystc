# Changelog

## Unreleased - the released binary can actually search

### Fixed
- **The packaged dashboard shipped without its own runtime** (external review 2026-09-03,
  finding 6 — severity raised on verification: the review called it a build risk, measuring the
  artifact showed it was already broken). The release workflow installed a hand-written list —
  `pyinstaller PyQt6 PyQt6-WebEngine numpy scipy requests` — and never `pip install .`, while
  `gystc.spec` bundles `brain_mcp.indexer.vector_store` and `.embedder`. Reading the ZIP central
  directory of the v1.4.3 asset (2085 entries) finds **no** faiss, torch, sentence-transformers,
  transformers, mcp, watchdog, psutil or httpx. Since `brain/web_widget.py` imports `brain_mcp`
  inside its request handlers, wrapped in `except Exception -> 500`, the shipped app opened,
  drew the graph, and answered every `/api/stats`, `/api/search` and `/api/reindex` with
  `No module named 'faiss'`. The build stayed green because "Verify build output" only checked
  that `dist/` existed. The build now installs `.[dashboard]`, so the artifact contains what the
  spec says it contains.
- **Semantic search could never work in a packaged install**: `_load()` sets `HF_HUB_OFFLINE=1`,
  so `SentenceTransformer("<hub-name>")` resolves only from a local Hugging Face cache — and
  nothing in the project ever created one. A fresh install therefore degraded silently to
  keyword-only search, with the reason buried on stderr. The model now travels *inside* the
  bundle (`scripts/bundle_model.py` saves it to `assets/model` at build time,
  `brain_mcp/indexer/bundled_model.py` resolves it at load time), so offline stays on and there
  is nothing left to fetch. This adds about 465 MB to the download; the app makes no network
  request either way. The bundle records *which* model it holds (`gystc-model.json`) and is used
  only when that matches the configured `model_name` — a bundled default silently replacing a
  model the user chose would embed the vault with the wrong model, and on a dimension mismatch
  `VectorStore.load` deletes the existing index. `GYSTC_MODEL_DIR` still overrides everything,
  since pointing at a directory is an explicit choice.

### Added
- **`GYSTC Dashboard --selfcheck [report.json]`**: a headless check that imports every runtime
  dependency, loads the bundled model, then indexes, embeds and retrieves a throwaway note in a
  temporary directory — it never touches the user's vault, config, database or index. The
  release build runs it against the built binary and refuses to package a build that fails it.
  A directory listing cannot tell whether a binary can search; running it can. It blocks the
  hub and empties `HF_HOME` *before* the first import, because `huggingface_hub` freezes that
  flag into a constant at import time — otherwise the check would pass on the networked build
  machine by downloading what the artifact is missing, and fail on the user's offline one.
- Regression tests (`tests/test_release_bundle.py`) that fail at pull-request time if the build
  stops installing the project, stops bundling the model, stops running the self-check before
  packaging, or if a new runtime dependency is added that the self-check does not import.

## v1.4.3 - external review follow-up: index integrity, dashboard API auth, CI gates (2026-08-25)

### Fixed
- **Reading the stats could delete the vector index** (external review 2026-08-23, finding 4 —
  severity raised on verification): `VectorStore.load()` deletes an index file whose dimension
  differs from the one passed in, which is right for the writer paths that rebuild afterwards
  and fatal for a caller that only wants a number. Two display-only callers passed a hardcoded
  `384`, so with any non-384 embedding model configured (`BRAIN_MODEL_NAME`, e.g.
  `all-mpnet-base-v2` at 768) they destroyed `index.faiss` just by being run: the dashboard's
  `/api/stats`, which `brain-app.js` fetches on every page load, and `brain_mcp config show` —
  the command a user runs *because* search stopped working. Both failures were silent
  (`except Exception: pass` on one side, a "was corrupt" line on the other), and since the DB
  kept its `faiss_idx` stamps nothing re-embedded afterwards either: semantic search stayed
  dead while keyword search kept answering and covered it up. Both now use the new
  `read_index_stats()`, which reads `(ntotal, dimension)` without writing and without guessing
  a dimension. `config show` also prints the index dimension, since a mismatch with the
  configured model is exactly what that command is run to diagnose.
- **`/api/stats` no longer swallows failures**: the edge-count fallback logged nothing, and the
  raw `sqlite3` connection had no busy timeout — a live writer could turn the panel into a
  bare "database is locked". Now `PRAGMA busy_timeout = 5000` and a stderr line.

- **Rejected POSTs answered with an RST instead of a status on macOS**: found by the new CI
  gate on its first run. A handler that answers 401/403/404 without reading the request body
  leaves the received bytes in the socket; `BaseHTTPRequestHandler` then closes the connection,
  and on BSD/macOS closing a socket that still holds unread data sends an RST rather than a FIN.
  The client raises `ConnectionResetError` instead of seeing the status just written — a hook
  post rejected for a stale token reported "connection reset by peer", pointing debugging at the
  network instead of at the token. Windows and Linux tolerate the same code, which is why three
  long-standing activity-feed tests only went red once the suite started running on macOS. All
  three rejection paths (activity feed 401/403, dashboard 404) now drain the body first, bounded
  by `MAX_DRAIN_BODY`.

- **Dashboard API endpoints now require the session token** (finding 1, the review's only HIGH):
  `/api/config` (POST rewrites the configuration), `/api/reindex`, `/api/search` and
  `/api/vault/<path>` (serves any note or attachment in the vault) ran on an ephemeral
  127.0.0.1 port with no token and no origin check — while the activity feed on the *fixed*
  port 9500 has had both since the last audit. Any local process could scan for the port and
  read vault contents over HTTP, and a browser tab could reach the state-changing endpoints
  (the body is parsed as JSON regardless of Content-Type, so `text/plain` skips the preflight).
  All `/api/*` routes now require a per-session bearer token, injected into the dashboard's own
  page at document creation via `QWebEngineScript`, so a tab that guesses the port still cannot
  obtain it. Foreign origins are rejected on top of that; the page's own origin is allowed
  because Chrome sends `Origin` on same-origin POSTs. Static files stay open — the shell has to
  load before any script can send a token, and it carries no vault data.
- **A reindex no longer loads a second embedding model** (finding 3): `/api/reindex` built its
  own `SentenceTransformerBackend` and `VectorStore` next to the ones the search cache already
  held — two models in one process. It now reuses the cached triple, which also fixes a
  correctness bug: rebuilding into a private store left the cached one (the store search answers
  from) holding the pre-reindex index until the dashboard restarted. Closing the widget releases
  the cached database handle, which nothing did before.
- **Windows device names in note titles** (finding 5): `sanitize_title` stripped bad characters
  but let `CON`, `NUL`, `COM1` … through, and Windows resolves those to devices regardless of
  extension or directory — so a note titled `CON` could not be written. The device stem now gets
  a trailing underscore (`CON_.md`). Applied everywhere, not only on Windows: a vault written on
  macOS is routinely opened on Windows.
- **The watcher ignored `.MD`** (finding 6) and its delete path checked neither the exclude list
  nor the pending-write registry — archiving a folder pulled exactly the notes the exclude list
  exists to keep out back through the change path.
- **`created_at` was the modification time** (finding 7): now taken from `st_birthtime`
  (macOS/BSD) or `st_ctime` (Windows, where it is the creation time), clamped to never exceed
  the modification time; Linux exposes neither, so mtime remains the fallback there.
- **The proxy passed HTTP errors off as JSON-RPC responses** (finding 9): `forward_one` returned
  a 401 body verbatim, so the client saw an error object where a result belongs — and the real
  cause (a stale registry token, which respawning the daemon repairs) stayed hidden.
- **`load_config` never validated what it loaded** (finding 11): only `save_config` did, so a
  `config.json` edited by hand came back silently and misbehaved somewhere far from its cause.
  It now warns and keeps going — refusing to load would lock the user out of the command that
  fixes the file.
- **Not changed — finding 8 could not be reproduced.** The review filed `/health` as
  unauthenticated (leaks the PID). The daemon's guard middleware wraps every route, including one
  appended to `app.router.routes` *after* `add_middleware`, which is the ordering that could
  plausibly have bypassed it. `test_daemon_guard_covers_health_route_appended_after_the_middleware`
  reproduces production exactly and shows an anonymous `GET /health` gets 401.

### CI
- **Tests now gate every pull request and every release** (finding 2): `build.yml` was the only
  workflow and triggered solely on `push: tags: v*`, so the 65 test files — `test_security.py`
  and the daemon E2E suite included — ran nowhere automatically, and a tag on a red tree still
  produced published binaries. New reusable `test.yml` runs the full suite (Windows + macOS,
  `QT_QPA_PLATFORM=offscreen`) on pull requests and master pushes, and `build` now `needs` it.
- **Lint gate**: `ruff` now runs in CI with the rule set pinned in `pyproject.toml`, so the
  finding count is a property of the repo rather than of whoever runs it (the review reported 250
  from its own invocation). 89 findings auto-fixed, 4 unused locals removed by hand, one
  deliberate `except: pass` documented — the activity-feed hook runs as a fresh process on every
  tool use, so there is no "warn once" to be had there. Deliberately not enabled: `BLE001`
  (48 blind excepts, nearly all documented decisions in recovery paths — that would mean 48
  `noqa` comments, not 48 fixes), import sorting, and `subprocess`-without-`check`, which needs a
  per-call judgement rather than a sweep.

## v1.4.2 — Windows fixes: vendored web asset + UTF-8 stdio (2026-07-17)

### Fixed
- **Dashboard web asset now tracked**: `.gitignore`'s `build/` pattern was unanchored and
  accidentally matched the vendored `three.module.js` under
  `brain/web/vendor/three-0.160.0/build/`, silently excluding it from the repo (and from
  release checkouts) while `dist/`/`build/` at the repo root stayed correctly ignored.
  Anchored to `/build/` and the file is now tracked.
- **UTF-8 stdio on Windows**: without `PYTHONUTF8=1`, piped stdio defaults to the system's
  legacy code page (cp1252) — but MCP JSON-RPC traffic and hook payloads are UTF-8, so
  umlauts/em-dashes in tool arguments and stored notes turned into mojibake. The daemon
  proxy and the activity-feed hook now force UTF-8 on stdin/stdout via `reconfigure()`
  (guarded with `hasattr` so it stays a no-op under pytest's captured stdio).

### Release integrity
- Release artifacts now ship with a `SHA256SUMS.txt` — verify a download with
  `Get-FileHash <file> -Algorithm SHA256` and compare against the matching line.

## v1.4.0 — Vault curation + full hardening pass (2026-06-11)

### Added
- **Vault curation** (`brain_mcp curate`): git-reversible vault cleanup — dead-link scan, semantic
  near-duplicate detection, directory archiving. Read-only `analyze` produces a proposal file;
  `apply` is human-gated with a full per-action diff preview, content fingerprints against
  concurrent edits, and surgical pathspec commits (never `git add -A`).
- **Indexer**: `exclude_dirs` support + stale-note pruning on (re)build, with version-history
  tombstones so archiving never destroys note history.

### Fixed (43 verified findings across 3 adversarial audit rounds; suite 243 → 408 tests)
- **Index integrity**: a failed embed can no longer leave a note silently mapped to its OLD
  content's vector (detach-then-embed across pipeline, watcher, store, and rollback);
  re-embedding no longer leaks the previous vector; `--force` clears stale stamps so partial
  failures can't collide ids; corrupted-index recovery now works on every writer path.
- **Lifecycle**: a slow model load no longer disables indexing for the whole session — the
  watcher and keyword search start instantly and vectors backfill when the model is ready;
  a writer whose model hard-fails releases the lock so a healthy sibling takes over;
  `brain_status` surfaces `model_failed` / `indexing_active` / `is_writer`.
- **CLI**: `brain_mcp index` takes the writer lock (no more racing a live server) and never
  reports success after an embed failure.
- **Daemon**: chunked-transfer bodies respect the size cap; the idle-shutdown respawn race is
  gone (lock released before teardown + launcher retry with backoff).
- **Dashboard**: real path containment for the vault file server (resolved paths, symlink-safe),
  reindex respects the writer lock, the activity feed requires a constant-time-checked session
  token (0600 token file), and three.js is vendored locally — no CDN at runtime.
- **Storage**: transactional + idempotent migrations, fail-closed writer lock, `fts_search`
  logs instead of silently returning empty, batched reconcile transactions.

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
