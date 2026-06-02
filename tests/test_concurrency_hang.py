"""Regression tests for the executor/blocking hang (2026-06-02 audit).

The bug: embedding tools called ``embedder.wait_ready(20s)`` *inside* the
executor worker thread, so the first search of a session (model still loading)
froze for up to 20s, and a flood of searches starved even the "instant" tools.

These tests pin the fix: when the model isn't ready, ``brain_retrieve`` falls
back to FTS instantly and ``brain_related`` falls back to backlinks-only
instantly -- they must never block on ``wait_ready``.
"""
from __future__ import annotations

import time

from brain_mcp.config import BrainConfig
from brain_mcp.indexer.embedder import SentenceTransformerBackend
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.server import BrainState, _related_logic, _retrieve_logic
from brain_mcp.storage.database import BrainDB

# A real backend that is deliberately never loaded. If any code path calls
# wait_ready() on it, it blocks the full timeout -- which is exactly the hang
# these tests forbid (asserted via the <2s budget below).
NOT_READY_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _state_with_unready_model(tmp_path) -> BrainState:
    db = BrainDB(tmp_path / "brain.db")
    a = db.upsert_note(
        path="a.md", title="Alpha", content="alpha talks about caching and retries",
        content_hash="h1", region_idx=0, tags=[], word_count=6,
        created_at="2026-06-01", modified_at="2026-06-01",
    )
    db.upsert_note(
        path="b.md", title="Beta", content="beta also discusses caching",
        content_hash="h2", region_idx=0, tags=[], word_count=4,
        created_at="2026-06-01", modified_at="2026-06-01",
    )
    beta = db.get_note_by_title("Beta")
    db.upsert_edge(a, beta["id"], link_text="Beta")

    embedder = SentenceTransformerBackend(NOT_READY_MODEL)  # NOT loaded
    assert embedder.is_ready is False
    vectors = VectorStore(dimension=embedder.dimension)
    config = BrainConfig(data_dir=tmp_path)
    return BrainState(config=config, db=db, vectors=vectors, embedder=embedder)


def test_retrieve_falls_back_to_fts_instantly(tmp_path):
    state = _state_with_unready_model(tmp_path)
    t = time.perf_counter()
    result = _retrieve_logic(
        state, query="caching", region=None, limit=10,
        threshold=0.3, file_paths=None, depth=1,
    )
    elapsed = time.perf_counter() - t

    assert elapsed < 2.0, f"retrieve blocked {elapsed:.1f}s (must not wait_ready)"
    assert isinstance(result, list)
    titles = {r.get("title") for r in result if isinstance(r, dict)}
    assert "Alpha" in titles, "FTS fallback should still find keyword matches"
    assert all(r.get("fts_only") for r in result if isinstance(r, dict))


def test_related_falls_back_to_backlinks_instantly(tmp_path):
    state = _state_with_unready_model(tmp_path)
    t = time.perf_counter()
    result = _related_logic(state, title="Alpha", path=None, limit=10)
    elapsed = time.perf_counter() - t

    assert elapsed < 2.0, f"related blocked {elapsed:.1f}s (must not wait_ready)"
    assert isinstance(result, list)
    titles = {r.get("title") for r in result if isinstance(r, dict)}
    assert "Beta" in titles, "backlink relation should resolve without embeddings"
