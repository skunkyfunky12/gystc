"""Semantic near-duplicate detection over the persisted FAISS index (READ-ONLY).

Finds notes that are highly similar but not byte-identical -- the real
"stale memory / mergeable duplicate" candidates the deterministic detectors miss.
Reads the existing index directly (no daemon, no re-embedding): reconstructs the
note-level vectors and runs ONE batched note-vs-note search.

The shared index also holds CHUNK vectors (pipeline adds both); searching it
directly lets a chunked note's own near-identical chunk vectors eat the whole
top-k budget, silently hiding true note-level near-dups. So we search a
temporary index built from note-level vectors only."""
from __future__ import annotations

import faiss
import numpy as np

from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.storage.database import BrainDB


def find_near_duplicates(db: BrainDB, vectors: VectorStore, *,
                         valid_paths: set[str] | None = None,
                         threshold: float = 0.88, top_k: int = 4) -> list[dict]:
    """Flag near-duplicate note pairs. Pass `valid_paths` (the real vault .md
    paths) to restrict to actual notes and skip index pollution (graphify code
    nodes, code symbols, etc.)."""
    notes = [n for n in db.get_all_notes() if n["faiss_idx"] is not None]
    if valid_paths is not None:
        notes = [n for n in notes if n["path"] in valid_paths]
    # Reconstruct ONLY note-level vectors: chunk vectors never enter the search
    # space, so they can't consume the top-k budget.
    kept: list = []
    recon: list[np.ndarray] = []
    for n in notes:
        vec = vectors.reconstruct(int(n["faiss_idx"]))
        if vec is None:
            continue
        kept.append(n)
        recon.append(vec)
    if len(kept) < 2:
        return []
    mat = np.ascontiguousarray(np.stack(recon), dtype=np.float32)
    faiss.normalize_L2(mat)  # stored vectors are already normalised; keep IP == cosine
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    k = min(top_k + 1, len(kept))  # +1: each query finds its own vector first
    scores, ids = index.search(mat, k)  # ONE batched search (BLAS matmul), not n loops

    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for qi, n in enumerate(kept):
        for score, pos in zip(scores[qi], ids[qi]):
            pos = int(pos)
            if pos < 0 or pos == qi or score < threshold:
                continue
            other = kept[pos]
            pair = tuple(sorted([n["path"], other["path"]]))
            if pair in seen:
                continue
            seen.add(pair)
            out.append({
                "file": n["path"], "kind": "near_duplicate",
                "problem": f"~{score:.0%} similar to another note",
                "action": "review for merge/archive",
                "tier": "yellow",
                "detail": {"similar_to": other["path"], "score": round(float(score), 3)},
            })
    return out
