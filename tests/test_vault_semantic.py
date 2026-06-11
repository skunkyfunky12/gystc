"""Semantic near-duplicate detection over the existing FAISS index (read-only).
This is the layer the deterministic detectors miss: notes that are SIMILAR but
not byte-identical (the real "stale memory / mergeable duplicate" candidates)."""
from __future__ import annotations

import numpy as np

from brain_mcp.curation.semantic import find_near_duplicates
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.storage.database import BrainDB


def _seed(tmp_path):
    db = BrainDB(tmp_path / "brain.db")
    vectors = VectorStore(dimension=4)
    rows = [
        ("a.md", np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)),
        ("b.md", np.array([[0.99, 0.1, 0.0, 0.0]], dtype=np.float32)),  # ~ a
        ("c.md", np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)),   # unrelated
    ]
    for path, vec in rows:
        nid = db.upsert_note(path=path, title=path, content=path, content_hash="h_" + path,
                             region_idx=0, tags=[], word_count=1,
                             created_at="2026-01-01", modified_at="2026-01-01")
        db.set_faiss_idx(nid, vectors.add(vec)[0])
    return db, vectors


def test_reconstruct_roundtrip(tmp_path):
    db, vectors = _seed(tmp_path)
    try:
        v = vectors.reconstruct(0)
        assert v is not None and v.shape == (4,)
    finally:
        db.close()


def test_find_near_duplicates(tmp_path):
    db, vectors = _seed(tmp_path)
    try:
        cands = find_near_duplicates(db, vectors, threshold=0.9, top_k=3)
        pairs = {tuple(sorted([c["file"], c["detail"]["similar_to"]])) for c in cands}
        assert ("a.md", "b.md") in pairs
        assert all("c.md" not in p for p in pairs)
        assert all(c["tier"] == "yellow" and c["kind"] == "near_duplicate" for c in cands)
    finally:
        db.close()


def test_valid_paths_filters_out_pollution(tmp_path):
    db, vectors = _seed(tmp_path)
    try:
        # Excluding b.md -> the a~b pair must disappear (simulates skipping non-vault nodes).
        cands = find_near_duplicates(db, vectors, valid_paths={"a.md", "c.md"}, threshold=0.9)
        assert cands == []
    finally:
        db.close()
