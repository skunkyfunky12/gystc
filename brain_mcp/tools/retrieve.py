from __future__ import annotations

import json
import sqlite3

from brain_mcp.indexer.embedder import EmbeddingBackend
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.recent import REGION_NAMES, resolve_region_idx


def _safe_parse_tags(tags_str: str | None) -> list[str]:
    if not tags_str:
        return []
    try:
        return json.loads(tags_str)
    except (json.JSONDecodeError, TypeError):
        return []


def _rrf_merge(
    faiss_ranked: list[tuple[int, float]],
    fts_ranked: list[tuple[int, int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion. Merges FAISS (note_id, score) and FTS (note_id, rank)."""
    scores: dict[int, float] = {}
    for rank, (note_id, _) in enumerate(faiss_ranked):
        scores[note_id] = scores.get(note_id, 0) + 1.0 / (k + rank + 1)
    for note_id, fts_rank in fts_ranked:
        scores[note_id] = scores.get(note_id, 0) + 1.0 / (k + fts_rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


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
    region_idx_filter = resolve_region_idx(region)

    # --- FAISS semantic search ---
    faiss_note_hits: dict[int, dict] = {}
    if vectors.size > 0:
        query_vec = embedder.embed([query])
        scores, ids = vectors.search(query_vec, k=min(limit * 5, vectors.size))

        faiss_indices = [int(i) for i in ids[0] if i >= 0]
        notes = db.get_notes_by_faiss_indices(faiss_indices)
        chunks = db.get_chunks_by_faiss_indices(faiss_indices)

        note_map = {n["faiss_idx"]: n for n in notes}
        chunk_map = {c["faiss_idx"]: c for c in chunks}

        for faiss_id, score in zip(ids[0], scores[0]):
            faiss_id = int(faiss_id)
            score = float(score)
            if faiss_id < 0 or score < threshold:
                continue

            note = note_map.get(faiss_id)
            chunk = chunk_map.get(faiss_id)

            if note is not None:
                nid = note["id"]
                if nid not in faiss_note_hits or score > faiss_note_hits[nid]["score"]:
                    faiss_note_hits[nid] = {
                        "score": score,
                        "note": note,
                        "chunk": None,
                    }
            elif chunk is not None:
                nid = chunk["note_id"]
                if nid not in faiss_note_hits or score > faiss_note_hits[nid]["score"]:
                    faiss_note_hits[nid] = {
                        "score": score,
                        "note": None,
                        "chunk": chunk,
                    }

    # --- FTS5 keyword search ---
    fts_results = db.fts_search(query, limit=limit * 3)
    fts_ranked = [(row["id"], rank) for rank, row in enumerate(fts_results)]
    fts_note_map = {row["id"]: row for row in fts_results}

    # --- RRF fusion ---
    faiss_ranked = sorted(faiss_note_hits.items(), key=lambda x: x[1]["score"], reverse=True)
    faiss_for_rrf = [(nid, data["score"]) for nid, data in faiss_ranked]
    merged = _rrf_merge(faiss_for_rrf, fts_ranked)

    # --- Build results ---
    results = []
    for note_id, rrf_score in merged:
        hit = faiss_note_hits.get(note_id)
        note_row = None
        chunk_row = None

        if hit:
            note_row = hit["note"]
            chunk_row = hit["chunk"]
        if note_row is None and note_id in fts_note_map:
            note_row = fts_note_map[note_id]
        if note_row is None:
            note_row = db.get_note_by_id(note_id)
        if note_row is None:
            continue

        r_idx = note_row["region_idx"]
        if region_idx_filter is not None and r_idx != region_idx_filter:
            continue

        entry = {
            "title": note_row["title"],
            "path": note_row["path"],
            "region": REGION_NAMES[r_idx] if 0 <= r_idx < 12 else "Stammhirn",
            "region_idx": r_idx,
            "similarity": round(rrf_score, 4),
            "tags": _safe_parse_tags(note_row["tags"]),
            "created": note_row["created_at"],
            "modified": note_row["modified_at"],
            "word_count": note_row["word_count"],
        }

        if chunk_row is not None:
            entry["chunk_heading"] = chunk_row["heading"]
            entry["snippet"] = chunk_row["content"][:300].strip()
        else:
            content = note_row["content"] or ""
            entry["snippet"] = content[:200].strip()
            if len(content) > 200:
                entry["snippet"] += "..."

        results.append(entry)
        if len(results) >= limit:
            break

    return results
