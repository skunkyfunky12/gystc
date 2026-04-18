# brain_mcp/tools/context.py
from __future__ import annotations

import json
from pathlib import Path

from brain_mcp.indexer.embedder import EmbeddingBackend
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.recent import REGION_NAMES

# REVIEW FIX: Cap BFS neighbors to prevent exponential blowup on dense graphs
MAX_BFS_NEIGHBORS = 200


def handle_brain_context(
    db: BrainDB,
    vectors: VectorStore,
    embedder: EmbeddingBackend,
    file_paths: list[str] | None = None,
    task_description: str | None = None,
    depth: int = 1,
    max_notes: int = 10,
) -> list[dict] | dict:
    depth = max(1, min(depth, 3))
    max_notes = max(1, min(max_notes, 100))

    if not file_paths and not task_description:
        return {"error": "Provide file_paths, task_description, or both"}

    scored: dict[int, dict] = {}

    if file_paths:
        for fp in file_paths:
            stem = Path(fp).stem
            note = db.get_note_by_path(fp) or db.get_note_by_title(stem)
            if note is None:
                continue
            neighbor_ids = db.get_neighbor_ids(note["id"], depth=depth)
            # REVIEW FIX: Cap max neighbors from BFS
            neighbor_list = list(neighbor_ids)[:MAX_BFS_NEIGHBORS]
            for nid in neighbor_list:
                if nid not in scored:
                    neighbor = db.get_note_by_id(nid)
                    if neighbor:
                        scored[nid] = {
                            "note": neighbor,
                            "graph_score": 1.0,
                            "semantic_score": 0.0,
                            "reason": f"backlink from {note['title']} ({depth} hop)",
                        }

    if task_description and vectors.size > 0:
        query_vec = embedder.embed([task_description[:1000]])
        k = min(max_notes * 2, vectors.size)
        scores, ids = vectors.search(query_vec, k=k)
        faiss_indices = [int(i) for i in ids[0] if i >= 0]
        notes = db.get_notes_by_faiss_indices(faiss_indices)
        faiss_to_note = {n["faiss_idx"]: n for n in notes}

        for fid, score in zip(ids[0], scores[0]):
            fid = int(fid)
            score = float(score)
            note = faiss_to_note.get(fid)
            if note is None:
                continue
            nid = note["id"]
            if nid in scored:
                scored[nid]["semantic_score"] = score
                scored[nid]["reason"] += f" + semantic match ({score:.2f})"
            else:
                scored[nid] = {
                    "note": note,
                    "graph_score": 0.0,
                    "semantic_score": score,
                    "reason": f"semantic match ({score:.2f})",
                }

    source_paths = set(file_paths or [])
    results = []
    for nid, data in scored.items():
        note = data["note"]
        if note["path"] in source_paths:
            continue
        combined = data["semantic_score"] * 0.6 + data["graph_score"] * 0.4
        results.append({
            "title": note["title"],
            "path": note["path"],
            "region": REGION_NAMES[note["region_idx"]] if 0 <= note["region_idx"] < 12 else "Stammhirn",
            "similarity": round(combined, 4),
            "relevance_reason": data["reason"],
            "word_count": note["word_count"],
        })

    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results[:max_notes]
