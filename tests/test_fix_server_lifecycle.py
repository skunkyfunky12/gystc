"""Regression tests for the server-lifecycle audit fixes (brain_mcp/server.py).

Covers:
- (0)  Startup must not dead-end on a slow/failed model load: watcher + FTS-only
       reconcile start immediately, the model wait is shutdown-aware, degraded
       state is surfaced (indexing_active / model_failed), and a writer whose
       model hard-fails releases the writer lock so a sibling can promote.
- (3)  Reader->writer promotion must reload index.faiss from disk instead of
       saving its stale in-memory snapshot over the previous writer's index.
- (13) The watcher starts BEFORE the reconcile scan so changes landing during
       the scan are queued, not lost.
- (23) A brain_store that was still queued when the client received the timeout
       error must NOT mutate afterwards; mid-flight timeouts are reported as
       ambiguous. Store work runs on a dedicated pool, not the default executor.
- (24) Promotion backs out (and releases the lock) when shutdown wins the race,
       decided under the same lock _shutdown uses.
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import brain_mcp.server as server
from brain_mcp.config import BrainConfig
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.server import (
    BrainState,
    _background_startup,
    _brain_store_impl,
    _handle_file_change,
    _promotion_loop,
    _reconcile,
    _should_attempt_promotion,
    _shutdown,
    _start_indexing,
    _status_logic,
)
from brain_mcp.storage.database import BrainDB
from brain_mcp.storage.file_lock import WriterLock
from tests.conftest import MockEmbedder


class StubEmbedder(MockEmbedder):
    """Controllable embedder mirroring SentenceTransformerBackend semantics:
    a ready event that can finish as success or hard failure, and an optional
    embed gate to simulate a stuck encode."""

    def __init__(self, *, ready: bool = False) -> None:
        self._ready_evt = threading.Event()
        self._failed = False
        self.embed_gate: threading.Event | None = None
        self.embed_thread_names: list[str] = []
        if ready:
            self._ready_evt.set()

    def finish_load(self, *, fail: bool = False) -> None:
        self._failed = fail
        self._ready_evt.set()

    def wait_ready(self, timeout: float = 60.0) -> bool:
        return self._ready_evt.wait(timeout)

    @property
    def is_ready(self) -> bool:  # type: ignore[override]
        return self._ready_evt.is_set() and not self._failed

    def embed(self, texts):
        self.embed_thread_names.append(threading.current_thread().name)
        if self.embed_gate is not None:
            self.embed_gate.wait(timeout=10)
        if not self.is_ready:
            raise RuntimeError("model not loaded")
        return super().embed(texts)


def _wait_for(predicate, timeout: float = 8.0, msg: str = "condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    pytest.fail(f"timed out waiting for: {msg}")


def _make_state(tmp_path, *, embedder, is_writer=True, with_vault=True) -> BrainState:
    vault = None
    if with_vault:
        vault = tmp_path / "vault"
        vault.mkdir(exist_ok=True)
    return BrainState(
        config=BrainConfig(data_dir=tmp_path, vault_path=vault),
        db=BrainDB(tmp_path / "brain.db"),
        vectors=VectorStore(dimension=384),
        embedder=embedder,
        is_writer=is_writer,
    )


# ---------------------------------------------------------------------------
# (0) + (13): startup must index FTS-only immediately, watcher before scan
# ---------------------------------------------------------------------------

def test_watcher_and_fts_reconcile_start_before_model_ready(tmp_path):
    embedder = StubEmbedder(ready=False)
    state = _make_state(tmp_path, embedder=embedder)
    (state.config.vault_path / "alpha.md").write_text(
        "# Alpha\nSome content.", encoding="utf-8"
    )
    lock = WriterLock(tmp_path / "writer.lock")
    assert lock.acquire()
    state.writer_lock = lock

    bg = threading.Thread(target=_background_startup, args=(state, lock), daemon=True)
    bg.start()
    try:
        # Watcher + FTS rows must exist long before the model is ready.
        _wait_for(lambda: state.watcher is not None and state.watcher.is_running,
                  msg="watcher running before model ready")
        _wait_for(lambda: state.db.get_note_by_path("alpha.md") is not None,
                  msg="FTS-only reconcile upserted the note")
        assert state.vectors.size == 0, "no vectors may exist before the model is ready"
        # Degraded state is surfaced, not just a stderr line.
        status = _status_logic(state)
        assert status["indexing_active"] is True
        assert status["model_loaded"] is False
        assert status["model_failed"] is False

        # Model becomes ready -> the embedding pass fills in the vectors.
        embedder.finish_load()
        _wait_for(lambda: state.db.get_note_by_path("alpha.md")["faiss_idx"] is not None,
                  msg="embedding pass assigned faiss_idx")
        _wait_for(lambda: state.indexing_active is False,
                  msg="indexing_active cleared after embedding pass")
        bg.join(timeout=8)
        assert not bg.is_alive(), "background startup thread must finish"
    finally:
        _shutdown(state)


def test_background_startup_model_wait_bails_on_shutdown(tmp_path):
    """The old one-shot join(120) blocked the startup thread for two minutes and
    then dead-ended. The wait must be shutdown-aware."""
    embedder = StubEmbedder(ready=False)  # never finishes loading
    state = _make_state(tmp_path, embedder=embedder, with_vault=False)
    lock = WriterLock(tmp_path / "writer.lock")
    assert lock.acquire()
    state.writer_lock = lock

    bg = threading.Thread(target=_background_startup, args=(state, lock), daemon=True)
    bg.start()
    time.sleep(0.3)
    _shutdown(state)  # sets _shut_done
    bg.join(timeout=6)
    assert not bg.is_alive(), "startup thread must bail when shutdown is flagged"


def test_writer_releases_lock_on_hard_model_failure(tmp_path, monkeypatch):
    """A writer whose model hard-fails can never embed; it must surface
    model_failed and release the writer lock so a sibling can promote."""
    # Large interval: the demoted instance must not steal the lock back before
    # the 'sibling' (this test) grabs it.
    monkeypatch.setattr(server, "WRITER_PROMOTE_INTERVAL", 5)
    embedder = StubEmbedder(ready=False)
    state = _make_state(tmp_path, embedder=embedder, with_vault=False)
    lock = WriterLock(tmp_path / "writer.lock")
    assert lock.acquire()
    state.writer_lock = lock

    bg = threading.Thread(target=_background_startup, args=(state, lock), daemon=True)
    bg.start()
    sibling = WriterLock(tmp_path / "writer.lock")
    try:
        embedder.finish_load(fail=True)
        _wait_for(lambda: state.model_failed, msg="model_failed flag set")
        _wait_for(lambda: not state.is_writer, msg="failed writer demoted")
        assert _status_logic(state)["model_failed"] is True
        # The lock must now be acquirable by a sibling.
        _wait_for(sibling.acquire, msg="sibling could acquire the released writer lock")
    finally:
        state._shut_done = True  # stop the fallback promotion loop
        sibling.release()
        _shutdown(state)


# ---------------------------------------------------------------------------
# (3): promotion must reload the on-disk index, never save the stale snapshot
# ---------------------------------------------------------------------------

def test_promotion_reloads_on_disk_index_not_stale_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WRITER_PROMOTE_INTERVAL", 0.05)
    embedder = StubEmbedder(ready=True)
    state = _make_state(tmp_path, embedder=embedder, is_writer=False, with_vault=False)

    # The previous writer left 2 vectors on disk ...
    disk_store = VectorStore(dimension=384)
    disk_store.add(embedder.embed(["one", "two"]))
    disk_store.save(state.config.index_path)
    # ... while this reader's in-memory snapshot is stale (1 junk vector).
    state.vectors.add(embedder.embed(["stale"]))
    assert state.vectors.size == 1

    lock = WriterLock(tmp_path / "writer.lock")
    promo = threading.Thread(target=_promotion_loop, args=(state, lock), daemon=True)
    promo.start()
    try:
        _wait_for(lambda: state.is_writer, msg="reader promoted to writer")
        _wait_for(lambda: state.vectors.size == 2,
                  msg="promotion swapped in the on-disk index (2 vectors)")
        promo.join(timeout=8)
        # The on-disk index must still hold the previous writer's 2 vectors --
        # the old code saved the stale 1-vector snapshot over it.
        assert VectorStore.load(state.config.index_path, dimension=384).size == 2
    finally:
        _shutdown(state)


# ---------------------------------------------------------------------------
# (24): promotion/shutdown TOCTOU
# ---------------------------------------------------------------------------

def test_promotion_backs_out_when_shutdown_wins_after_acquire(tmp_path, monkeypatch):
    """Simulate _shutdown winning exactly between lock acquire and promotion
    publish: the promotion must back out under state._shutdown_lock, release the
    writer lock, and never start indexing."""
    monkeypatch.setattr(server, "WRITER_PROMOTE_INTERVAL", 0.05)
    embedder = StubEmbedder(ready=True)
    state = _make_state(tmp_path, embedder=embedder, is_writer=False, with_vault=False)

    class TocTouLock(WriterLock):
        def acquire(self) -> bool:
            got = super().acquire()
            if got:
                # _shutdown flips the flag under _shutdown_lock at this instant.
                with state._shutdown_lock:
                    state._shut_done = True
            return got

    lock = TocTouLock(tmp_path / "writer.lock")
    promo = threading.Thread(target=_promotion_loop, args=(state, lock), daemon=True)
    promo.start()
    promo.join(timeout=8)
    assert not promo.is_alive(), "promotion loop must exit"
    assert state.is_writer is False, "promotion must not publish after shutdown won"
    assert state.writer_lock is None
    assert state.watcher is None and state.reindexer is None, "indexing must not start"
    # The lock must have been released on back-out.
    sibling = WriterLock(tmp_path / "writer.lock")
    assert sibling.acquire(), "backed-out promotion must release the writer lock"
    sibling.release()


# ---------------------------------------------------------------------------
# (23): brain_store timeout vs. late mutation + dedicated executor
# ---------------------------------------------------------------------------

def test_store_timed_out_in_queue_never_mutates_late(tmp_path):
    """If the client already received the timeout error while the store was
    still queued, the worker must NOT write the note afterwards."""
    embedder = StubEmbedder(ready=True)
    state = _make_state(tmp_path, embedder=embedder)
    state.store_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gystc-store")
    blocker = threading.Event()
    state.store_pool.submit(blocker.wait, 10)  # occupy the single worker

    try:
        result = asyncio.run(_brain_store_impl(
            state, title="late-note", content="should never land",
            region=None, region_idx=None, tags=None, folder="", timeout=0.3,
        ))
        assert "error" in result
        assert "NOT saved" in result["error"], "queued timeout must guarantee no write"

        blocker.set()  # drain the queue: the cancelled store runs (and must no-op)
        state.store_pool.shutdown(wait=True)
        assert not (state.config.vault_path / "late-note.md").exists(), \
            "store mutated the vault after the client was told it failed"
        assert state.db.get_note_by_path("late-note.md") is None, \
            "store mutated the DB after the client was told it failed"
    finally:
        blocker.set()
        _shutdown(state)


# ---------------------------------------------------------------------------
# Review round 2 (0/5): writer paths must honor corrupted_on_load / stale stamps
# ---------------------------------------------------------------------------

def test_start_indexing_clears_stale_stamps_when_index_lost(tmp_path):
    """Empty vector store + rows that still carry faiss_idx stamps = the index
    file was corrupted (load() deleted it) or never saved before a crash. The
    reconcile only re-embeds faiss_idx IS NULL rows, so without clearing, those
    notes never rebuild and fresh ids collide with the stale ones."""
    embedder = StubEmbedder(ready=False)
    state = _make_state(tmp_path, embedder=embedder, with_vault=False)
    state.db.upsert_note(
        path="ghost.md", title="Ghost", content="body", content_hash="h1",
        region_idx=3, tags=[], word_count=1,
        created_at="2026-01-01", modified_at="2026-01-01", faiss_idx=7,
    )
    assert state.db.has_faiss_stamps()
    assert state.vectors.size == 0
    try:
        _start_indexing(state)
        assert not state.db.has_faiss_stamps(), \
            "stale faiss_idx stamps must be cleared when the index is empty/lost"
    finally:
        _shutdown(state)


def test_start_indexing_clears_stamps_on_corrupted_load_flag(tmp_path):
    embedder = StubEmbedder(ready=False)
    state = _make_state(tmp_path, embedder=embedder, with_vault=False)
    state.db.upsert_note(
        path="ghost.md", title="Ghost", content="body", content_hash="h1",
        region_idx=3, tags=[], word_count=1,
        created_at="2026-01-01", modified_at="2026-01-01", faiss_idx=7,
    )
    state.vectors.corrupted_on_load = True
    try:
        _start_indexing(state)
        assert not state.db.has_faiss_stamps()
    finally:
        _shutdown(state)


def test_start_indexing_keeps_stamps_when_index_matches(tmp_path):
    """A healthy store (vectors present, no corruption) must NOT be wiped."""
    embedder = StubEmbedder(ready=True)
    state = _make_state(tmp_path, embedder=embedder, with_vault=False)
    ids = state.vectors.add(embedder.embed(["one"]))
    state.db.upsert_note(
        path="ok.md", title="Ok", content="body", content_hash="h1",
        region_idx=3, tags=[], word_count=1,
        created_at="2026-01-01", modified_at="2026-01-01", faiss_idx=ids[0],
    )
    try:
        _start_indexing(state)
        assert state.db.has_faiss_stamps(), "healthy stamps must survive startup"
    finally:
        _shutdown(state)


# ---------------------------------------------------------------------------
# Review round 2 (1): indexing_active must not lie forever after model failure
# ---------------------------------------------------------------------------

def test_reconcile_clears_indexing_active_for_model_failed_instance(tmp_path):
    """A model-failed instance's FTS-only pass is its terminal steady state:
    the embedding pass that normally clears the flag can never run, so the
    flag must come down with the pass instead of reporting an eternal
    'index still building'."""
    embedder = StubEmbedder(ready=False)
    embedder.finish_load(fail=True)
    state = _make_state(tmp_path, embedder=embedder, with_vault=False)
    state.model_failed = True
    _reconcile(state)
    assert state.indexing_active is False, \
        "model_failed + reconcile done => indexing_active must be False"
    status = _status_logic(state)
    assert status["model_failed"] is True
    assert status["indexing_active"] is False


def test_reconcile_keeps_indexing_active_while_model_still_loading(tmp_path):
    """Contrast: a merely-slow model keeps the flag up — vectors are still
    coming once the embedding pass runs."""
    embedder = StubEmbedder(ready=False)  # loading, not failed
    state = _make_state(tmp_path, embedder=embedder, with_vault=False)
    _reconcile(state)
    assert state.indexing_active is True


# ---------------------------------------------------------------------------
# Review round 2 (4): model-failed instances must let healthy siblings win
# ---------------------------------------------------------------------------

def test_promotion_backoff_prefers_healthy_siblings(tmp_path):
    embedder = StubEmbedder(ready=True)
    state = _make_state(tmp_path, embedder=embedder, is_writer=False, with_vault=False)
    # Healthy instance: every cycle attempts.
    assert all(_should_attempt_promotion(state, c) for c in range(1, 8))
    # Model-failed instance: attempts only every Nth cycle (lone-instance
    # fallback still promotes eventually, healthy siblings win the race).
    state.model_failed = True
    attempts = [c for c in range(1, 19) if _should_attempt_promotion(state, c)]
    assert attempts, "a lone model-failed instance must still promote eventually"
    assert len(attempts) <= 3, f"model-failed must back off, attempted at {attempts}"


# ---------------------------------------------------------------------------
# Review round 2 (6): displaced faiss stamps must be removed from the store
# ---------------------------------------------------------------------------

def test_file_change_removes_displaced_stamp_from_racing_pass(tmp_path, monkeypatch):
    """If a reconcile embedding pass stamps the same note between our snapshot
    read and our set_faiss_idx, that displaced vector must be removed -- only
    the id currently in the row is ever passed to remove(), so the loser would
    otherwise leak as an orphan forever."""
    embedder = StubEmbedder(ready=True)
    state = _make_state(tmp_path, embedder=embedder)
    note_path = state.config.vault_path / "alpha.md"
    note_path.write_text("# Alpha\nOriginal.", encoding="utf-8")
    _handle_file_change(state, str(note_path), "created")
    row = state.db.get_note_by_path("alpha.md")
    assert row is not None and row["faiss_idx"] is not None

    # Plant the 'racing' vector the concurrent pass would have added.
    racing_id = state.vectors.add(embedder.embed(["racing snapshot"]))[0]
    real_set = state.db.set_faiss_idx

    def racing_set(note_id, faiss_idx):
        real_set(note_id, faiss_idx)
        return racing_id  # the row held the racer's stamp when we overwrote it

    monkeypatch.setattr(state.db, "set_faiss_idx", racing_set)
    note_path.write_text("# Alpha\nEdited content.", encoding="utf-8")
    _handle_file_change(state, str(note_path), "modified")

    assert state.vectors.reconstruct(racing_id) is None, \
        "the displaced (racing) vector must be removed from the store"
    new_row = state.db.get_note_by_path("alpha.md")
    assert state.vectors.reconstruct(new_row["faiss_idx"]) is not None


def test_store_timed_out_midflight_reports_ambiguity_and_uses_dedicated_pool(tmp_path):
    """A store stuck mid-embed cannot be undone -- the error must say the write
    may still complete (no silent divergence), and the work must run on the
    dedicated store pool, not the shared default executor."""
    embedder = StubEmbedder(ready=True)
    embedder.embed_gate = threading.Event()  # blocks embed -> simulates stuck encode
    state = _make_state(tmp_path, embedder=embedder)
    state.store_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gystc-store")

    try:
        result = asyncio.run(_brain_store_impl(
            state, title="stuck-note", content="content",
            region=None, region_idx=None, tags=None, folder="", timeout=0.3,
        ))
        assert "error" in result
        assert "may" in result["error"], "mid-flight timeout must surface the ambiguity"

        embedder.embed_gate.set()
        state.store_pool.shutdown(wait=True)
        assert embedder.embed_thread_names, "embed never ran"
        assert all(n.startswith("gystc-store") for n in embedder.embed_thread_names), \
            f"store ran on the wrong executor: {embedder.embed_thread_names}"
    finally:
        embedder.embed_gate.set()
        _shutdown(state)
