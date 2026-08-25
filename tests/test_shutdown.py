"""Shutdown-path tests.

The parent-death watchdog (_orphan_exit) and the lifespan 'finally' can both
run when the client disconnects. Both must funnel through a single idempotent
_shutdown() so save()+close() never run twice / from two threads (which raced
on the shared db handle and printed scary 'closed database' errors), and so the
reindex worker is drained before the DB is closed.
"""
from __future__ import annotations

import threading


from brain_mcp.config import BrainConfig
from brain_mcp.server import BrainState, _shutdown
from brain_mcp.storage.database import BrainDB
from tests.conftest import MockEmbedder


class _FakeWatcher:
    def __init__(self) -> None:
        self.stops = 0

    def stop(self) -> None:
        self.stops += 1


class _FakeReindexer:
    def __init__(self) -> None:
        self.stops = 0
        self.joins = 0

    def stop(self) -> None:
        self.stops += 1

    def join(self, timeout=None) -> None:
        self.joins += 1


class _FakeVectors:
    def __init__(self) -> None:
        self.saves = 0

    def save(self, path) -> None:
        self.saves += 1


def _state(tmp_path, *, is_writer=True) -> BrainState:
    return BrainState(
        config=BrainConfig(data_dir=tmp_path),
        db=BrainDB(tmp_path / "brain.db"),
        vectors=_FakeVectors(),
        embedder=MockEmbedder(),
        watcher=_FakeWatcher(),
        reindexer=_FakeReindexer(),
        is_writer=is_writer,
    )


def test_braindb_close_is_idempotent(tmp_path):
    db = BrainDB(tmp_path / "brain.db")
    db.close()
    db.close()  # must not raise


def test_shutdown_runs_cleanup_exactly_once(tmp_path):
    state = _state(tmp_path)
    _shutdown(state)
    _shutdown(state)
    assert state.watcher.stops == 1
    assert state.reindexer.stops == 1
    assert state.reindexer.joins == 1, "reindexer must be drained (joined) before close"
    assert state.vectors.saves == 1


def test_concurrent_shutdown_is_single_winner(tmp_path):
    state = _state(tmp_path)
    barrier = threading.Barrier(8)

    def run():
        barrier.wait()
        _shutdown(state)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert state.vectors.saves == 1, "save() ran more than once under concurrent shutdown"
    assert state.reindexer.joins == 1


def test_reader_instance_does_not_persist_index(tmp_path):
    """A read-only (non-writer) instance must never write index.faiss -- that is
    what prevents two coexisting servers from clobbering the shared index."""
    state = _state(tmp_path, is_writer=False)
    _shutdown(state)
    assert state.vectors.saves == 0


def test_shutdown_loser_blocks_until_winner_completes(tmp_path):
    """The watchdog (_orphan_exit) follows _shutdown with os._exit(0). If it is the
    loser of the shutdown race it must WAIT for the winner's drain+save to finish,
    or os._exit would kill an in-flight re-index / drop the final save."""
    import time

    started = threading.Event()
    save_count = {"n": 0}

    class _SlowVectors:
        def save(self, path) -> None:
            started.set()
            time.sleep(0.3)
            save_count["n"] += 1

    state = _state(tmp_path)
    state.vectors = _SlowVectors()

    winner = threading.Thread(target=lambda: _shutdown(state))
    winner.start()
    assert started.wait(1.0), "winner never started its save"

    # This is the 'loser' call: _shut_done is already set, so it must block on
    # _shut_complete until the winner's save() finishes.
    _shutdown(state)
    assert save_count["n"] == 1, "loser returned before the winner finished cleanup"
    winner.join(timeout=2)
