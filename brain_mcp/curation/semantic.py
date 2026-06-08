"""Semantic near-duplicate detection over the persisted FAISS index (READ-ONLY).

Finds notes that are highly similar but not byte-identical -- the real
"stale memory / mergeable duplicate" candidates the deterministic detectors miss.
Reads the existing index directly (no daemon, no re-embedding): for each note's
stored vector, query its nearest neighbours and flag high-similarity pairs."""
from __future__ import annotations

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
    by_fid = {int(n["faiss_idx"]): n for n in notes}
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for n in notes:
        own = int(n["faiss_idx"])
        vec = vectors.reconstruct(own)
        if vec is None:
            continue
        scores, ids = vectors.search(vec.reshape(1, -1), k=top_k)
        for score, fid in zip(scores[0], ids[0]):
            fid = int(fid)
            if fid == own or score < threshold:
                continue
            other = by_fid.get(fid)
            if other is None:
                continue
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
