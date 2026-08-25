from __future__ import annotations
import asyncio
import atexit
import os
import sys
import threading
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from mcp.server.fastmcp import FastMCP
from brain_mcp.config import BrainConfig, load_config
from brain_mcp.indexer.embedder import SentenceTransformerBackend
from brain_mcp.indexer.pipeline import index_vault, reindex_note_chunks
from brain_mcp.indexer.reindex_queue import ReindexWorker
from brain_mcp.indexer.scanner import parse_note_file
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.indexer.watcher import BrainWatcher
from brain_mcp.process_guard import ParentDeathWatchdog
from brain_mcp.storage.database import BrainDB
from brain_mcp.storage.file_lock import WriterLock
from brain_mcp.tools.recent import handle_brain_recent
from brain_mcp.tools.regions import handle_brain_regions
from brain_mcp.tools.retrieve import handle_brain_retrieve
from brain_mcp.tools.store import handle_brain_store
from brain_mcp.tools.related import handle_brain_related
from brain_mcp.tools.classify_tool import handle_brain_classify, handle_brain_classify_feedback
from brain_mcp.tools.versioning import handle_brain_history, handle_brain_diff, handle_brain_rollback
from brain_mcp.indexer.reranker import CrossEncoderReRanker

WAIT_READY_TIMEOUT = 20
TOOL_TIMEOUT = 30
FAISS_SAVE_INTERVAL = 60
WRITER_PROMOTE_INTERVAL = 20  # readers retry the writer lock this often
# A model-failed instance retries the lock this many times LESS often, so a
# healthy sibling (20s cadence) wins the writer race while a lone broken
# instance still promotes eventually (FTS-only indexing beats none at all).
MODEL_FAILED_PROMOTE_EVERY = 6


@dataclass
class BrainState:
    config: BrainConfig
    db: BrainDB
    vectors: VectorStore
    embedder: SentenceTransformerBackend
    watcher: BrainWatcher | None = None
    reranker: CrossEncoderReRanker | None = None
    embed_pool: ThreadPoolExecutor | None = None
    watchdog: ParentDeathWatchdog | None = None
    reindexer: ReindexWorker | None = None
    # True only for the single instance that owns indexing (holds the writer
    # lock). Readers never write index.faiss / faiss_idx, so coexisting servers
    # can't clobber the shared index.
    is_writer: bool = True
    writer_lock: object | None = None
    # Dedicated pool for brain_store so a slow/stuck store (embedding) can't
    # occupy the shared default executor and starve the instant tools.
    store_pool: ThreadPoolExecutor | None = None
    # Degraded-state flags surfaced via brain_status: model_failed = the model
    # load finished but failed (search stays FTS-only); indexing_active = a
    # reconcile pass is running or the embedding pass is still pending.
    model_failed: bool = False
    indexing_active: bool = False
    _index_lock: object = field(default_factory=threading.Lock)
    _saver_started: bool = False
    _shutdown_lock: object = field(default_factory=threading.Lock)
    _shut_done: bool = False
    _shut_complete: object = field(default_factory=threading.Event)


class _ReadyGatedEmbedder:
    """Embedder facade for reconcile passes: raises instead of blocking on the
    model-load lock when the model isn't ready yet. That way the FTS part of
    index_vault (upserts, edges, chunk rows) completes immediately and only the
    embed batches are skipped (index_vault catches and logs them). The missed
    vectors are filled in by the embedding pass once the model is ready."""

    def __init__(self, inner: SentenceTransformerBackend) -> None:
        self._inner = inner

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    @property
    def is_ready(self) -> bool:
        return self._inner.is_ready

    def embed(self, texts):
        if not self._inner.is_ready:
            raise RuntimeError("embedding model not ready yet (FTS-only pass)")
        return self._inner.embed(texts)


def _index_vault(state: BrainState) -> None:
    if state.config.vault_path is None or not state.config.vault_path.is_dir():
        return
    index_vault(state.db, state.vectors, _ReadyGatedEmbedder(state.embedder),
                state.config.vault_path, state.config.folder_to_region,
                exclude_dirs=state.config.exclude_dirs)


def _handle_file_change(state: BrainState, path: str, event_type: str) -> None:
    try:
        rel = str(Path(path).relative_to(state.config.vault_path)).replace("\\", "/")
    except ValueError:
        return
    if event_type == "deleted":
        old_row = state.db.get_note_by_path(rel)
        if old_row:
            if old_row["faiss_idx"] is not None:
                state.vectors.remove([old_row["faiss_idx"]])
            chunk_faiss = state.db.delete_chunks_for_note(old_row["id"])
            if chunk_faiss:
                state.vectors.remove(chunk_faiss)  # don't leak chunk vectors
        state.db.delete_note(rel)
        return

    note = parse_note_file(Path(path), state.config.vault_path, state.config.folder_to_region)
    if note is None:
        return

    old_row = state.db.get_note_by_path(rel)
    hash_match = old_row is not None and old_row["content_hash"] == note["content_hash"]
    already_embedded = old_row is not None and old_row["faiss_idx"] is not None
    # Skip only if nothing changed AND it already has a vector. A row whose
    # faiss_idx is NULL (e.g. written by a read-only instance, or embedded while
    # the model was still loading) still needs embedding even if the hash matches.
    if hash_match and already_embedded:
        return

    note_id = state.db.upsert_note(
        path=note["path"], title=note["title"], content=note["content"],
        content_hash=note["content_hash"], region_idx=note["region_idx"],
        tags=note["tags"], word_count=note["word_count"],
        created_at=note["created_at"], modified_at=note["modified_at"],
    )
    # Detach-then-embed: the surviving stamp (COALESCE) points at the OLD
    # content's vector. Detach it before embedding so a failed embed leaves
    # the row retryable (faiss_idx IS NULL) instead of permanently stale-mapped
    # (the new hash already matches, so future events would skip it forever).
    old = state.db.clear_faiss_idx(note_id)
    if old is not None:
        state.vectors.remove([old])
    if not state.embedder.is_ready:
        return
    try:
        vec = state.embedder.embed([note["content"]])
        faiss_ids = state.vectors.add(vec)
        # Remove whatever stamp we actually displaced (read-modify-write inside
        # set_faiss_idx), not a snapshot read earlier: a racing reconcile pass
        # may have stamped a newer id in between, and only the id in the row
        # ever reaches remove() -- the loser would leak as an orphan forever.
        displaced = state.db.set_faiss_idx(note_id, faiss_ids[0])
        if displaced is not None:
            state.vectors.remove([displaced])
        # Refresh chunk vectors too -- otherwise search returns stale snippets
        # for edited notes and leaks orphaned chunk vectors. This is now the sole
        # indexing path for reader-written and external (Obsidian) edits.
        reindex_note_chunks(state.db, state.vectors, state.embedder,
                            note_id, note["title"], note["content"])
        print(f"Re-indexed: {rel}", file=sys.stderr)
    except Exception as exc:
        print(f"Embedding error for {rel}: {exc}", file=sys.stderr)


def _shutdown(state: BrainState) -> None:
    """Idempotent, single-winner cleanup.

    Both the lifespan 'finally' (stdin EOF) and the parent-death watchdog
    (_orphan_exit) can fire when a client disconnects -- sometimes at the same
    instant. Funnelling both through here means save()+close() run exactly once
    instead of racing on the shared db handle (which printed 'closed database'
    errors), and the re-index worker is drained before the DB is closed so no
    re-index is killed mid-write.
    """
    with state._shutdown_lock:
        first = not state._shut_done
        state._shut_done = True
    if not first:
        # Another thread already owns cleanup. Wait for it to FINISH before
        # returning, so a caller that follows _shutdown with os._exit (the
        # watchdog) can't pre-empt an in-flight drain/save on the winning path.
        state._shut_complete.wait(timeout=8)
        return

    try:
        if state.watchdog is not None:
            state.watchdog.stop()
        if state.watcher is not None:
            state.watcher.stop()
        if state.reindexer is not None:
            state.reindexer.stop()
            state.reindexer.join(timeout=5)  # let the in-flight re-index finish
        # Only the writer instance owns index.faiss; readers must never persist it.
        if state.is_writer:
            try:
                state.vectors.save(state.config.index_path)
            except Exception as exc:
                print(f"shutdown: index save failed: {exc}", file=sys.stderr)
        if state.embed_pool is not None:
            try:
                state.embed_pool.shutdown(wait=False, cancel_futures=True)
            except Exception as exc:
                print(f"shutdown: embed pool shutdown failed: {exc}", file=sys.stderr)
        if state.store_pool is not None:
            try:
                state.store_pool.shutdown(wait=False, cancel_futures=True)
            except Exception as exc:
                print(f"shutdown: store pool shutdown failed: {exc}", file=sys.stderr)
        try:
            state.db.close()
        except Exception as exc:
            print(f"shutdown: db close failed: {exc}", file=sys.stderr)
        if state.writer_lock is not None:
            try:
                state.writer_lock.release()
            except Exception as exc:
                print(f"shutdown: writer-lock release failed: {exc}", file=sys.stderr)
    finally:
        state._shut_complete.set()


def _orphan_exit(state: BrainState) -> None:
    """Called by the parent-death watchdog when the client that spawned this
    server is gone. Clean up, then force-exit so no orphaned server lingers
    holding brain.db. Lingering orphans (combined with the old global
    sibling-killer) were the root of the random 'Connection closed' disconnects.
    """
    print("Client/parent gone -> shutting down orphaned GYSTC server.", file=sys.stderr)
    _shutdown(state)
    sys.stderr.flush()
    os._exit(0)


def _periodic_faiss_save(state: BrainState) -> None:
    """Save FAISS index periodically so it survives process kills."""
    import time
    while not state._shut_done:
        time.sleep(FAISS_SAVE_INTERVAL)
        # Decide under the same lock _shutdown uses: never start a save after
        # cleanup began, and a demoted instance (model failure) must never save
        # its stale snapshot over the active writer's on-disk index.
        with state._shutdown_lock:
            if state._shut_done:
                return
            if not state.is_writer:
                continue
        try:
            if state.vectors.size > 0:
                state.vectors.save(state.config.index_path)
        except Exception as exc:
            print(f"WARNING: Periodic FAISS save failed: {exc}", file=sys.stderr)


def _reconcile(state: BrainState) -> None:
    """One reconcile pass over the vault. FTS rows/edges/chunks are always
    written; vectors only if the model is ready at pass start (otherwise the
    embed batches raise inside index_vault and are logged, and indexing_active
    stays up until the embedding pass completes). Serialised via _index_lock so
    startup, promotion and the deferred embedding pass never scan concurrently."""
    with state._index_lock:
        if state._shut_done:
            return
        ready = state.embedder.is_ready
        state.indexing_active = True
        try:
            _index_vault(state)
        except Exception as exc:
            print(f"ERROR: Writer reconcile scan failed: {exc}", file=sys.stderr)
        finally:
            # An FTS-only pass keeps the flag up: vectors are still missing
            # until the embedding pass (model-ready reconcile) finishes. A
            # model_failed instance is the exception -- its embedding pass can
            # never run, FTS-only is its terminal state, so the flag must come
            # down instead of reporting an eternal "index still building".
            state.indexing_active = (not ready) and not state.model_failed


def _start_indexing(state: BrainState) -> None:
    """Writer-only: watch the vault, then reconcile it. Called when this
    instance becomes the writer -- at startup or via promotion. Does NOT need
    the embedding model: the reconcile runs FTS-only until the model is ready
    (the embedding pass follows in _background_startup) and _handle_file_change
    already guards embedding."""
    # Same lock/ordering as _shutdown: never start indexing once cleanup began.
    with state._shutdown_lock:
        if state._shut_done:
            return
    # Index/DB consistency gate (writer-only funnel for startup AND promotion):
    # an empty store while rows still carry faiss_idx stamps means the on-disk
    # index was corrupt (load() deleted it) or never saved before a crash. The
    # reconcile only re-embeds faiss_idx IS NULL rows, so without clearing,
    # those notes never rebuild and fresh ids collide with the stale stamps
    # (wrongly-mapped search results, wrong-vector removal on edit).
    if state.vectors.corrupted_on_load or (
        state.vectors.size == 0 and state.db.has_faiss_stamps()
    ):
        print("WARNING: index.faiss corrupt/lost but rows carry faiss_idx stamps; "
              "clearing them so the reconcile re-embeds everything.", file=sys.stderr)
        state.db.clear_all_faiss_idx()
        state.vectors.corrupted_on_load = False
    vault_exists = state.config.vault_path is not None and state.config.vault_path.is_dir()
    if vault_exists and state.config.auto_index:
        # Watcher BEFORE the reconcile scan: changes landing mid-scan are queued
        # by the reindex worker instead of silently lost. The debounce plus the
        # hash/faiss_idx skip in _handle_file_change make the overlap harmless
        # (worst case one redundant re-embed). Re-index runs off the observer
        # thread, serially + debounced per path, so a burst of file changes
        # doesn't storm the global DB lock.
        reindexer = ReindexWorker(lambda p, e: _handle_file_change(state, p, e))
        reindexer.start()
        state.reindexer = reindexer
        watcher = BrainWatcher(state.config.vault_path, reindexer.submit,
                               exclude_dirs=state.config.exclude_dirs)
        watcher.start()
        state.watcher = watcher
    if vault_exists:
        # Always reconcile (incremental -- skips unchanged notes via content hash),
        # independent of index_on_startup, so notes a reader wrote while no writer
        # was alive get picked up. Otherwise they'd stay search-invisible.
        _reconcile(state)
    with state._shutdown_lock:
        if state._shut_done:
            return
    try:
        state.vectors.save(state.config.index_path)
    except Exception as exc:
        print(f"WARNING: Initial FAISS save failed: {exc}", file=sys.stderr)
    if not state._saver_started:
        # Once per process: the saver gates on is_writer itself, so it survives
        # demotion + re-promotion without stacking threads.
        state._saver_started = True
        saver = threading.Thread(target=_periodic_faiss_save, args=(state,), daemon=True)
        saver.start()


def _demote_writer(state: BrainState) -> None:
    """Hand the writer role back: stop our indexing machinery and release the
    lock so a sibling can promote. Used on hard model-load failure (this
    instance can never embed)."""
    with state._shutdown_lock:
        if state._shut_done:
            return
        lock = state.writer_lock
        state.writer_lock = None
        state.is_writer = False
        state.indexing_active = False
    watcher, state.watcher = state.watcher, None
    reindexer, state.reindexer = state.reindexer, None
    if watcher is not None:
        watcher.stop()
    if reindexer is not None:
        reindexer.stop()
        reindexer.join(timeout=5)
    if lock is not None:
        lock.release()


def _handle_model_failure(state: BrainState, writer_lock: WriterLock) -> None:
    """The model load finished but failed: search stays FTS-only. Surface the
    state (brain_status: model_failed) and, as writer, release the lock so a
    sibling whose model loaded fine can promote and embed."""
    state.model_failed = True
    print("ERROR: Embedding model failed to load -- search runs FTS-only "
          "(brain_status: model_failed).", file=sys.stderr)
    if state.is_writer:
        print("GYSTC: releasing writer lock after model failure so a sibling can promote.",
              file=sys.stderr)
        _demote_writer(state)
        # Fallback: re-enter the promotion race so a lone instance still gets
        # FTS-only indexing if no healthy sibling ever takes the lock. (Readers
        # already run their own promotion loop -- don't start a second one.)
        _promotion_loop(state, writer_lock)


def _should_attempt_promotion(state: BrainState, cycle: int) -> bool:
    """Healthy instances try every cycle; a model-failed instance (FTS-only
    forever) backs off to every MODEL_FAILED_PROMOTE_EVERY-th cycle so that a
    healthy sibling wins the writer race in the window between attempts."""
    if not state.model_failed:
        return True
    return cycle % MODEL_FAILED_PROMOTE_EVERY == 0


def _promotion_loop(state: BrainState, writer_lock: WriterLock) -> None:
    """Reader-only: periodically retry the writer lock. If the writer instance
    exits, take over indexing so reader writes don't pile up un-indexed."""
    import time
    cycle = 0
    while not state._shut_done:
        time.sleep(WRITER_PROMOTE_INTERVAL)
        if state._shut_done:
            return
        cycle += 1
        if not _should_attempt_promotion(state, cycle):
            continue
        if writer_lock.acquire():
            # Publish the promotion under the same lock _shutdown uses to set
            # _shut_done: either _shutdown sees is_writer/writer_lock and cleans
            # them up, or we see _shut_done and back out -- never both half-way.
            with state._shutdown_lock:
                promoted = not state._shut_done
                if promoted:
                    state.writer_lock = writer_lock
                    state.is_writer = True
            if not promoted:
                writer_lock.release()
                return
            # Readers serve a stale in-memory snapshot (they never see the old
            # writer's updates); saving it would clobber the on-disk index.
            # Reload from disk and swap it in BEFORE any indexing/saving.
            try:
                state.vectors = VectorStore.load(state.config.index_path,
                                                 dimension=state.embedder.dimension)
            except Exception as exc:
                print(f"WARNING: promotion index reload failed: {exc}", file=sys.stderr)
            print("GYSTC: promoted to writer (previous writer exited).", file=sys.stderr)
            _start_indexing(state)
            return


def _background_startup(state: BrainState, writer_lock: WriterLock) -> None:
    try:
        if state.is_writer:
            # Watcher + FTS-only reconcile + periodic saver need no model; start
            # them immediately so changes are captured and keyword search works
            # while the model loads.
            _start_indexing(state)
        else:
            # Read-only instance: never indexes/persists -- the writer owns
            # that. Retry the writer lock regardless of model state, in its own
            # thread so a hung model load can't block promotion.
            threading.Thread(target=_promotion_loop, args=(state, writer_lock),
                             daemon=True).start()

        # Shutdown-aware wait for the embedding model. Replaces the old one-shot
        # join(120) that, on slow loads, skipped startup indexing forever.
        while not state._shut_done:
            if state.embedder.wait_ready(timeout=2.0):
                break
        if state._shut_done:
            return
        if not state.embedder.is_ready:
            _handle_model_failure(state, writer_lock)
            return
        # Model ready. If we are (or were promoted to) the writer and the
        # reconcile ran FTS-only, run the embedding pass now.
        if state.is_writer and state.indexing_active:
            _reconcile(state)
        role = "writer" if state.is_writer else "read-only instance"
        print(f"Background startup complete ({role}).", file=sys.stderr)
    except Exception as exc:
        import traceback
        print(f"ERROR: Background startup crashed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


def _is_shared_daemon() -> bool:
    """True when this process hosts the single shared, process-global BrainState.

    The daemon sets GYSTC_NO_PARENT_WATCHDOG=1 (it must outlive its launcher, so
    it skips the parent-death watchdog) AND runs the MCP server with
    stateless_http=True, which re-enters brain_lifespan on EVERY request. Without
    a singleton that rebuilt the whole brain per call -- model reload (~15s),
    writer-election churn, and a reconcile scan that aborted mid-pass with
    "Cannot operate on a closed database" (the DB was closed by the per-request
    teardown) -- so the index never converged. The in-process stdio server
    (`serve --direct`) runs exactly one lifespan per process and is unaffected.

    Both conditions are required: GYSTC_NO_PARENT_WATCHDOG can be *inherited* by a
    `serve --direct` child, and taking the daemon branch there would skip _shutdown
    while cmd_serve os._exit(0)'s past atexit -- losing the final index save / DB
    close / reindex drain. stateless_http (set only in run_daemon) is the defining
    trait of the per-request-lifespan daemon, so gate on it too.
    """
    return (os.environ.get("GYSTC_NO_PARENT_WATCHDOG") == "1"
            and bool(mcp.settings.stateless_http))


def _build_brain_state() -> BrainState:
    """Construct the BrainState, elect the writer, and start background startup.

    Called once per process via _acquire_shared_state (the daemon re-enters the
    lifespan per request; this must not run more than once or the model reloads
    and the reconcile churns)."""
    import time
    t0 = time.perf_counter()
    config = load_config()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    db = BrainDB(config.db_path)
    embedder = SentenceTransformerBackend(config.model_name)
    model_thread = threading.Thread(target=embedder._load, daemon=True)
    model_thread.start()
    vectors = VectorStore.load(config.index_path, dimension=embedder.dimension)
    reranker = None
    if config.reranker == "cross-encoder":
        reranker = CrossEncoderReRanker()
        reranker.start_loading()
    embed_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gystc-embed")
    # Single worker: stores serialise on the DB write lock anyway, and a stuck
    # store makes later ones fail fast in the queue (with an honest "not saved"
    # timeout) instead of piling up threads.
    store_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gystc-store")
    state = BrainState(config=config, db=db, vectors=vectors, embedder=embedder,
                       reranker=reranker, embed_pool=embed_pool, store_pool=store_pool)
    # Elect a single writer instance. stdio MCP runs one server per client
    # (this CLI + Hermes + Command Center); only the lock holder owns index.faiss
    # and the faiss_idx column. Readers stay fully usable (FTS + the on-disk
    # index snapshot + the shared brain.db) but never persist the index.
    writer_lock = WriterLock(config.data_dir / "writer.lock")
    if writer_lock.acquire():
        state.is_writer = True
        state.writer_lock = writer_lock
    else:
        state.is_writer = False
    startup_ms = int((time.perf_counter() - t0) * 1000)
    role = "writer" if state.is_writer else "read-only"
    print(f"GYSTC MCP started in {startup_ms}ms ({role}). Vault: {config.vault_path}", file=sys.stderr)
    print(f"DB: {config.db_path} | Index: {vectors.size} vectors", file=sys.stderr)

    bg = threading.Thread(target=_background_startup, args=(state, writer_lock), daemon=True)
    bg.start()

    # Self-terminate if the client that spawned us disappears, instead of
    # lingering as an orphan holding brain.db. Replaces the old global
    # _kill_zombie_siblings, which disconnected live one-server-per-client
    # siblings (this CLI, Hermes, Command Center each spawn their own server).
    # The shared daemon skips this (it must outlive its launcher).
    if not _is_shared_daemon():
        watchdog = ParentDeathWatchdog(os.getppid(), lambda: _orphan_exit(state))
        watchdog.start()
        state.watchdog = watchdog
    return state


_SHARED_STATE: BrainState | None = None
_SHARED_STATE_LOCK = threading.Lock()


def _acquire_shared_state() -> BrainState:
    """Build the BrainState once per process and reuse it across lifespan entries.

    stateless_http re-enters brain_lifespan on every request; building per request
    reloaded the model and churned writer election + the reconcile (which aborted
    with "closed database"). Build-once keeps a single warm model, one writer
    election, and a continuously-running watcher."""
    global _SHARED_STATE
    with _SHARED_STATE_LOCK:
        if _SHARED_STATE is None:
            _SHARED_STATE = _build_brain_state()
            if _is_shared_daemon():
                # The daemon skips per-request teardown, so its graceful shutdown
                # (drain re-index, save index.faiss, release the writer lock) runs
                # once at process exit instead.
                atexit.register(_shutdown, _SHARED_STATE)
        return _SHARED_STATE


@asynccontextmanager
async def brain_lifespan(server: FastMCP) -> AsyncIterator[BrainState]:
    global _SHARED_STATE
    state = _acquire_shared_state()
    try:
        yield state
    finally:
        # Daemon (stateless_http): the lifespan re-runs per request -- must NOT
        # tear down the shared singleton here (that closed the DB mid-reconcile);
        # atexit handles process-exit cleanup. stdio / in-process: exactly one
        # lifespan per process -> tear down now, because cmd_serve os._exit(0)'s
        # right after, which would skip atexit.
        if not _is_shared_daemon():
            _shutdown(state)
            with _SHARED_STATE_LOCK:
                _SHARED_STATE = None
            print("GYSTC MCP stopped.", file=sys.stderr)

BRAIN_INSTRUCTIONS = """
You have access to a persistent knowledge vault organized into 12 brain regions.

TOOLS (8 total):
- brain_retrieve: Search by query and/or file context. Primary search tool.
- brain_store: Save important knowledge as a vault note.
- brain_related: Find notes connected to a specific note.
- brain_recent: See recently changed notes (instant, no model needed).
- brain_status: Health check — note counts, model status (instant).
- brain_regions: List or describe brain regions (instant).
- brain_classify: Classify notes into regions, correct misclassifications, or batch-reclassify.
- brain_versions: View history, diff, or rollback note versions.

WHEN TO SEARCH:
- User asks about a past decision, architecture, or project context.
- You need project-specific conventions not in the code.

WHEN NOT TO SEARCH:
- General programming questions — use your own knowledge.
- The user is asking you to write code — just write it.
- You already have enough context from the conversation.
- You just searched and got results — don't search again with a rephrased query.

KEEP IT FAST:
- brain_recent and brain_status need no model — use them when the model is still loading.
- brain_retrieve with only file_paths does graph traversal without embeddings.
- Don't chain multiple search calls. One brain_retrieve is usually enough.
""".strip()

def _retrieve_logic(state: BrainState, *, query, region, limit, threshold, file_paths, depth):
    """Non-blocking retrieve. If the embedding model isn't ready yet, fall back to
    FTS instantly instead of blocking a worker thread on wait_ready — that block
    was the old 6-20s first-search-of-session freeze."""
    fts_only = bool(query) and not state.embedder.is_ready
    result = handle_brain_retrieve(
        state.db, state.vectors, state.embedder,
        query=query, region=region, limit=limit, threshold=threshold,
        reranker=state.reranker, file_paths=file_paths, depth=depth,
        fts_only=fts_only,
    )
    if fts_only and query and isinstance(result, list):
        for entry in result:
            if isinstance(entry, dict):
                entry["fts_only"] = True
    return result


def _related_logic(state: BrainState, *, title, path, limit):
    """Non-blocking related. Falls back to backlinks-only when the model isn't
    ready yet, so it never blocks on wait_ready."""
    return handle_brain_related(
        state.db, state.vectors, state.embedder,
        title=title, path=path, limit=limit, semantic=state.embedder.is_ready,
    )


def _status_logic(state: BrainState) -> dict:
    """brain_status payload, incl. the degraded-state flags (model_failed /
    indexing_active / is_writer) so a broken or still-building index is visible
    to clients instead of only a stderr line."""
    counts = state.db.get_region_note_counts()
    edge_types = state.db.get_edge_type_counts()
    from brain_mcp.tools.recent import REGION_NAMES
    region_dist = {
        REGION_NAMES[idx]: cnt
        for idx, cnt in sorted(counts.items())
        if 0 <= idx < 12
    }
    return {
        "total_notes": state.db.get_note_count(),
        "total_vectors": state.vectors.size,
        "total_edges": sum(edge_types.values()),
        "edge_types": edge_types,
        "regions": region_dist,
        "model_loaded": state.embedder.is_ready,
        "model_failed": state.model_failed,
        "indexing_active": state.indexing_active,
        "is_writer": state.is_writer,
        "reranker_enabled": state.reranker is not None,
        "reranker_loaded": state.reranker.is_ready if state.reranker else False,
        "vault_path": str(state.config.vault_path),
    }


mcp = FastMCP("GYSTC", lifespan=brain_lifespan, instructions=BRAIN_INSTRUCTIONS)

# ---------------------------------------------------------------------------
# INSTANT TOOLS (no embedding model required)
# ---------------------------------------------------------------------------

@mcp.tool()
async def brain_status() -> dict:
    """Fast health check. Note/edge counts, model status, region distribution. Always instant."""
    state: BrainState = mcp.get_context().request_context.lifespan_context

    try:
        return await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(None, lambda: _status_logic(state)),
            timeout=TOOL_TIMEOUT,
        )
    except TimeoutError:
        return {"error": f"brain_status timed out after {TOOL_TIMEOUT}s."}

@mcp.tool()
async def brain_recent(days: int = 7, region: str | None = None, limit: int = 20) -> list[dict]:
    """Recently changed notes. Instant, no model needed.

    Args:
        days: Lookback window (default 7, max 365)
        region: Filter by region name
        limit: Max results (default 20, max 100)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context

    def _do():
        return handle_brain_recent(state.db, days=days, region=region, limit=limit)

    try:
        return await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(None, _do),
            timeout=TOOL_TIMEOUT,
        )
    except TimeoutError:
        return [{"error": f"brain_recent timed out after {TOOL_TIMEOUT}s."}]

@mcp.tool()
async def brain_regions(action: Literal["list", "describe", "customize"], region: str | None = None, description: str | None = None, color: str | None = None) -> dict | list[dict]:
    """List, describe, or customize brain regions.

    Args:
        action: "list", "describe", or "customize"
        region: Region name (required for describe/customize)
        description: New description (customize only)
        color: Hex color like #FF0000 (customize only)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context

    def _do():
        return handle_brain_regions(state.db, action=action, region=region, description=description, color=color)

    try:
        return await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(None, _do),
            timeout=TOOL_TIMEOUT,
        )
    except TimeoutError:
        return {"error": f"brain_regions timed out after {TOOL_TIMEOUT}s."}

# ---------------------------------------------------------------------------
# EMBEDDING TOOLS (need model, have FTS fallback + timeouts)
# ---------------------------------------------------------------------------

@mcp.tool()
async def brain_retrieve(
    query: str | None = None,
    region: str | None = None,
    limit: int = 10,
    threshold: float = 0.3,
    file_paths: list[str] | None = None,
    depth: int = 1,
) -> list[dict]:
    """Search the vault by meaning, keywords, and/or file context.

    Modes:
    - query only: hybrid semantic+keyword search (FAISS+FTS5+RRF)
    - file_paths only: graph traversal via backlinks (no model needed)
    - both: merged results with combined scoring

    Falls back to keyword-only search if the embedding model isn't ready yet.

    Args:
        query: Natural language search query
        region: Filter by region name
        limit: Max results (default 10, max 100)
        threshold: Min similarity (default 0.3)
        file_paths: Vault paths or filenames to find related notes via graph
        depth: Backlink graph hops 1-3 (default 1)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    try:
        return await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                state.embed_pool,
                lambda: _retrieve_logic(
                    state, query=query, region=region, limit=limit,
                    threshold=threshold, file_paths=file_paths, depth=depth,
                ),
            ),
            timeout=TOOL_TIMEOUT,
        )
    except TimeoutError:
        return [{"error": f"brain_retrieve timed out after {TOOL_TIMEOUT}s."}]

async def _brain_store_impl(state: BrainState, *, title, content, region, region_idx,
                            tags, folder, timeout: float = TOOL_TIMEOUT) -> dict:
    """brain_store body. Runs on the dedicated store pool (a slow/stuck store
    must not occupy the shared default executor and starve the instant tools).

    Timeout/late-write guard: once the client received a timeout error, a store
    that was still queued must NOT mutate afterwards -- the client believes it
    failed and may retry or give up. `started` lets the timeout path tell apart
    "never ran -> guaranteed no write" from "mid-write -> may still complete",
    so the response and reality can't silently diverge."""
    if state.config.vault_path is None:
        return {"error": "No vault_path configured"}

    started = threading.Event()
    cancelled = threading.Event()

    def _do() -> dict:
        started.set()
        if cancelled.is_set():
            # Client already got the timeout error while we sat in the queue.
            # Result is discarded (the future was abandoned) -- the point is to
            # NOT write anything.
            return {"error": "brain_store cancelled after client timeout (nothing written)."}
        return handle_brain_store(
            state.db, state.vectors, state.embedder, state.config.vault_path,
            title=title, content=content, region=region, region_idx=region_idx,
            tags=tags, folder=folder, watcher=state.watcher,
            persist=state.is_writer,
        )

    try:
        return await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(state.store_pool, _do),
            timeout=timeout,
        )
    except TimeoutError:
        cancelled.set()
        if not started.is_set():
            # Worker never started: the cancelled flag is guaranteed to be seen
            # before any write -> nothing was mutated.
            return {"error": f"brain_store timed out after {timeout}s while queued; "
                             "the note was NOT saved. Safe to retry."}
        # Worker is mid-flight: the write may still complete in the background.
        # Surface the ambiguity instead of pretending nothing happened.
        print(f"WARNING: brain_store('{title}') timed out mid-write; the write may "
              "still complete in the background.", file=sys.stderr)
        return {"error": f"brain_store timed out after {timeout}s mid-write; the note "
                         "may still have been saved -- check brain_recent before retrying."}


@mcp.tool()
async def brain_store(title: str, content: str, region: str | None = None, region_idx: int | None = None,
                tags: list[str] | None = None, folder: str = "") -> dict:
    """Save knowledge to the vault. Creates or updates a .md file. Auto-versioned.

    Args:
        title: Note title (becomes filename)
        content: Markdown content
        region: Brain region name (auto-detected if omitted)
        region_idx: Region index 0-11 (overrides name)
        tags: Additional tags (max 20)
        folder: Subfolder in vault (e.g. "02 Projekte")
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    return await _brain_store_impl(state, title=title, content=content, region=region,
                                   region_idx=region_idx, tags=tags, folder=folder)

@mcp.tool()
async def brain_related(title: str | None = None, path: str | None = None, limit: int = 10) -> list[dict] | dict:
    """Find notes connected to a specific note via backlinks and semantic similarity.

    Args:
        title: Note title to find relations for
        path: Note path (alternative to title)
        limit: Max results (default 10, max 100)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context

    try:
        return await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                state.embed_pool,
                lambda: _related_logic(state, title=title, path=path, limit=limit),
            ),
            timeout=TOOL_TIMEOUT,
        )
    except TimeoutError:
        return {"error": f"brain_related timed out after {TOOL_TIMEOUT}s."}

# ---------------------------------------------------------------------------
# MERGED TOOLS
# ---------------------------------------------------------------------------

@mcp.tool()
async def brain_classify(
    action: Literal["classify", "reclassify", "feedback"] = "classify",
    title: str | None = None,
    path: str | None = None,
    content: str | None = None,
    apply: bool = False,
    correct_region_idx: int | None = None,
    reason: str = "",
) -> dict:
    """Classify notes into brain regions. No API key needed — uses keyword rules.

    Actions:
    - "classify": Classify a single note (provide title, path, or content)
    - "reclassify": Batch-reclassify all Stammhirn notes (dry_run unless apply=True)
    - "feedback": Correct a misclassification (requires path + correct_region_idx)

    Args:
        action: "classify" (default), "reclassify", or "feedback"
        title: Note title
        path: Note path
        content: Note content (if not in DB)
        apply: Write changes to DB + file (default: false = dry run)
        correct_region_idx: Correct region 0-11 (feedback only)
        reason: Why this correction (feedback only)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context

    def _do():
        if action == "feedback":
            if not path:
                return {"error": "path required for feedback action"}
            if correct_region_idx is None:
                return {"error": "correct_region_idx required for feedback action"}
            return handle_brain_classify_feedback(
                state.db, state.config.vault_path, state.config.data_dir,
                path=path, correct_region_idx=correct_region_idx, reason=reason,
            )
        if action == "reclassify":
            result = handle_brain_classify(
                state.db, state.config.vault_path,
                batch=True, apply=apply,
            )
            result["dry_run"] = not apply
            return result
        return handle_brain_classify(
            state.db, state.config.vault_path,
            title=title, path=path, content=content, batch=False, apply=apply,
        )

    try:
        return await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(None, _do),
            timeout=TOOL_TIMEOUT,
        )
    except TimeoutError:
        return {"error": f"brain_classify timed out after {TOOL_TIMEOUT}s."}

@mcp.tool()
async def brain_versions(
    action: Literal["history", "diff", "rollback"],
    path: str = "",
    version_id: int | None = None,
) -> dict | list[dict]:
    """Manage note version history.

    Actions:
    - "history": List versions of a note
    - "diff": Compare current note with a previous version
    - "rollback": Restore a note to a previous version

    Args:
        action: "history", "diff", or "rollback"
        path: Note path in the vault
        version_id: Version ID (required for diff/rollback)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context

    def _do():
        if not path:
            return {"error": "path is required"}
        if action == "history":
            return handle_brain_history(state.db, path=path)
        if action == "diff":
            if version_id is None:
                return {"error": "version_id required for diff"}
            return handle_brain_diff(state.db, path=path, version_id=version_id)
        if action == "rollback":
            if version_id is None:
                return {"error": "version_id required for rollback"}
            if state.config.vault_path is None:
                return {"error": "No vault_path configured"}
            return handle_brain_rollback(
                state.db, state.config.vault_path, path=path,
                version_id=version_id, watcher=state.watcher,
                vectors=state.vectors, embedder=state.embedder,
                persist=state.is_writer,
            )
        return {"error": f"Unknown action: {action}. Use 'history', 'diff', or 'rollback'."}

    try:
        return await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(None, _do),
            timeout=TOOL_TIMEOUT,
        )
    except TimeoutError:
        return {"error": f"brain_versions timed out after {TOOL_TIMEOUT}s."}
