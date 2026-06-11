"""Regression tests for find_near_duplicates (audit finding 15):
chunk vectors share the FAISS index with note vectors; with k=top_k=4 a chunked
note's own chunk vectors eat the whole budget and true note-level near-dups are
silently missed. Also: the per-note search loop must be batched (one matrix
search), not ~N sequential single-vector searches.
"""
from __future__ import annotations

import numpy as np

from brain_mcp.curation.semantic import find_near_duplicates
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.storage.database import BrainDB


def _add_note(db, vectors, path: str, vec) -> int:
    nid = db.upsert_note(path=path, title=path, content=path, content_hash="h_" + path,
                         region_idx=0, tags=[], word_count=1,
                         created_at="2026-01-01", modified_at="2026-01-01")
    fid = vectors.add(np.array([vec], dtype=np.float32))[0]
    db.set_faiss_idx(nid, fid)
    return fid


def _seed_with_chunks(tmp_path):
    """Note a.md is 'chunked': 3 chunk vectors nearly identical to its note vector
    live in the SAME index (as pipeline.py really does), with no note row.
    Note b.md is a true near-duplicate of a.md."""
    db = BrainDB(tmp_path / "brain.db")
    vectors = VectorStore(dimension=4)
    _add_note(db, vectors, "a.md", [1.0, 0.0, 0.0, 0.0])
    # chunk vectors: in the index, NOT note-level (no db note row points at them)
    for _ in range(3):
        vectors.add(np.array([[0.999, 0.01, 0.0, 0.0]], dtype=np.float32))
    _add_note(db, vectors, "b.md", [0.98, 0.05, 0.0, 0.0])   # ~0.999 similar to a
    return db, vectors


def test_chunked_note_near_dup_still_found_with_default_top_k(tmp_path):
    # With the old per-note k=4 search, a's budget was [a, chunk, chunk, chunk]
    # and b's was [b, chunk, chunk, chunk] -> the real a~b pair was never reported.
    db, vectors = _seed_with_chunks(tmp_path)
    try:
        cands = find_near_duplicates(db, vectors, threshold=0.9)  # default top_k
        pairs = {tuple(sorted([c["file"], c["detail"]["similar_to"]])) for c in cands}
        assert ("a.md", "b.md") in pairs, \
            "chunk vectors must not eat the near-duplicate search budget"
    finally:
        db.close()


def test_chunk_vectors_never_reported_as_duplicates(tmp_path):
    db, vectors = _seed_with_chunks(tmp_path)
    try:
        cands = find_near_duplicates(db, vectors, threshold=0.9)
        files = {c["file"] for c in cands} | {c["detail"]["similar_to"] for c in cands}
        assert files <= {"a.md", "b.md"}
    finally:
        db.close()


def test_search_is_batched_not_per_note(tmp_path):
    # The shared store must not be hit with one search() call per note.
    db, vectors = _seed_with_chunks(tmp_path)
    calls = {"n": 0}
    orig = vectors.search

    def counting_search(query, k=10):
        calls["n"] += 1
        return orig(query, k=k)

    vectors.search = counting_search
    try:
        find_near_duplicates(db, vectors, threshold=0.9)
        assert calls["n"] <= 1, "must be one batched matrix search (or a note-only index)"
    finally:
        db.close()


def test_existing_behavior_unchanged_for_unchunked_notes(tmp_path):
    db = BrainDB(tmp_path / "brain.db")
    vectors = VectorStore(dimension=4)
    _add_note(db, vectors, "a.md", [1.0, 0.0, 0.0, 0.0])
    _add_note(db, vectors, "b.md", [0.99, 0.1, 0.0, 0.0])
    _add_note(db, vectors, "c.md", [0.0, 0.0, 1.0, 0.0])
    try:
        cands = find_near_duplicates(db, vectors, threshold=0.9, top_k=3)
        pairs = {tuple(sorted([c["file"], c["detail"]["similar_to"]])) for c in cands}
        assert ("a.md", "b.md") in pairs
        assert all("c.md" not in p for p in pairs)
        assert all(c["tier"] == "yellow" and c["kind"] == "near_duplicate" for c in cands)
    finally:
        db.close()


def test_valid_paths_still_filters_pollution(tmp_path):
    db, vectors = _seed_with_chunks(tmp_path)
    try:
        cands = find_near_duplicates(db, vectors, valid_paths={"a.md"}, threshold=0.9)
        assert cands == []
    finally:
        db.close()
