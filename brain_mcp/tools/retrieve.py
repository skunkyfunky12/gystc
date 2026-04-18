# brain_mcp/tools/retrieve.py
from __future__ import annotations

import json

from brain_mcp.indexer.embedder import EmbeddingBackend
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.recent import REGION_NAMES, REGION_NAME_TO_IDX, resolve_region_idx


def _fts_snippet(db: BrainDB, note_id: int, query: str) -> str | None:
    """Try to get an FTS5 snippet for the note matching the query.

    Returns None if FTS doesn't match (pure semantic hit).
    """
    try:
        safe_query = BrainDB._sanitize_fts_query(query)
        row = db.execute(
            """SELECT snippet(notes_fts, 1, '', '', '...', 40) AS snip
               FROM notes_fts
               WHERE notes_fts MATCH ? AND rowid = ?""",
            (safe_query, note_id),
        ).fetchone()
        if row and row["snip"]:
            return row["snip"]
    except Exception:
        pass
    return None


def handle_brain_retrieve(
    db: BrainDB,
    vectors: VectorStore,
    embedder: EmbeddingBackend,
    query: str,
    region: str | None = None,
    limit: int = 10,
    threshold: float = 0.3,
) -> list[dict]:
    query = query[:1000]
    limit = max(1, min(limit, 100))

    if vectors.size == 0:
        return []

    query_vec = embedder.embed([query])
    scores, ids = vectors.search(query_vec, k=min(limit * 3, vectors.size))

    faiss_indices = [int(i) for i in ids[0] if i >= 0]
    if not faiss_indices:
        return []

    notes = db.get_notes_by_faiss_indices(faiss_indices)
    note_map = {n["faiss_idx"]: n for n in notes}

    region_idx_filter = resolve_region_idx(region)
    results = []
    for faiss_id, score in zip(ids[0], scores[0]):
        faiss_id = int(faiss_id)
        score = float(score)
        if faiss_id < 0 or score < threshold:
            continue
        note = note_map.get(faiss_id)
        if note is None:
            continue
        if region_idx_filter is not None and note["region_idx"] != region_idx_filter:
            continue

        # REVIEW FIX: Use FTS5 snippet instead of naive content[:200]
        snippet = _fts_snippet(db, note["id"], query)
        if snippet is None:
            # Pure semantic match -- fall back to content prefix
            content = note["content"] or ""
            snippet = content[:200].strip()
            if len(content) > 200:
                snippet += "..."

        results.append({
            "title": note["title"],
            "path": note["path"],
            "region": REGION_NAMES[note["region_idx"]] if 0 <= note["region_idx"] < 12 else "Stammhirn",
            "region_idx": note["region_idx"],
            "similarity": round(score, 4),
            "snippet": snippet,
            "tags": json.loads(note["tags"]) if note["tags"] else [],
            "created": note["created_at"],
            "modified": note["modified_at"],
            "word_count": note["word_count"],
        })
        if len(results) >= limit:
            break

    return results
