# brain_mcp/tools/related.py
from __future__ import annotations

from brain_mcp.indexer.embedder import EmbeddingBackend
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.recent import REGION_NAMES
from brain_mcp.tools.retrieve import MAX_BFS_NEIGHBORS


def handle_brain_related(
    db: BrainDB,
    vectors: VectorStore,
    embedder: EmbeddingBackend,
    title: str | None = None,
    path: str | None = None,
    limit: int = 10,
    semantic: bool = True,
) -> list[dict] | dict:
    limit = max(1, min(limit, 100))

    if path:
        source = db.get_note_by_path(path)
    elif title:
        source = db.get_note_by_title(title)
    else:
        return {"error": "Provide either title or path"}

    if source is None:
        return {"error": f"Note not found: {title or path}"}

    semantic_scores: dict[int, float] = {}
    if semantic and source["faiss_idx"] is not None and vectors.size > 1:
        # Reuse the stored vector (microseconds) instead of re-embedding the
        # full note content (20-100ms CPU) on every call. Fall back to
        # embedding only if the id is missing from the index.
        query_vec = vectors.reconstruct(source["faiss_idx"])
        if query_vec is not None:
            query_vec = query_vec.reshape(1, -1)
        else:
            query_vec = embedder.embed([source["content"] or source["title"]])
        scores, ids = vectors.search(
            query_vec,
            k=min(limit * 2, vectors.size),
        )
        notes = db.get_notes_by_faiss_indices([int(i) for i in ids[0] if i >= 0])
        faiss_to_note = {n["faiss_idx"]: n for n in notes}
        for fid, score in zip(ids[0], scores[0]):
            fid = int(fid)
            note = faiss_to_note.get(fid)
            if note and note["id"] != source["id"]:
                semantic_scores[note["id"]] = float(score)

    neighbor_ids = db.get_neighbor_ids(source["id"], depth=1)
    if len(neighbor_ids) > MAX_BFS_NEIGHBORS:
        # Hub/MOC notes can have hundreds of backlinks; cap the fan-out like
        # retrieve does instead of fetching every neighbor.
        neighbor_ids = set(sorted(neighbor_ids)[:MAX_BFS_NEIGHBORS])

    all_ids = set(semantic_scores.keys()) | neighbor_ids
    results = []
    # One batched query instead of a lock-acquiring lookup per candidate.
    for note in db.get_notes_by_ids(list(all_ids)):
        nid = note["id"]
        if nid == source["id"]:
            continue

        sem_score = semantic_scores.get(nid, 0.0)
        graph_score = 1.0 if nid in neighbor_ids else 0.0
        combined = sem_score * 0.6 + graph_score * 0.4

        if sem_score > 0 and graph_score > 0:
            rel_type = "both"
        elif graph_score > 0:
            rel_type = "backlink"
        else:
            rel_type = "semantic"

        results.append({
            "title": note["title"],
            "path": note["path"],
            "region": REGION_NAMES[note["region_idx"]] if 0 <= note["region_idx"] < 12 else "Stammhirn",
            "score": round(combined, 4),
            "relation_type": rel_type,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:limit]
