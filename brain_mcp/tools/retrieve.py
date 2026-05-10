from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from brain_mcp.indexer.embedder import EmbeddingBackend
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.recent import REGION_NAMES, resolve_region_idx

MAX_BFS_NEIGHBORS = 200


@runtime_checkable
class ReRanker(Protocol):
    def rerank(self, query: str, candidates: list[dict]) -> list[dict]: ...


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
    scores: dict[int, float] = {}
    for rank, (note_id, _) in enumerate(faiss_ranked):
        scores[note_id] = scores.get(note_id, 0) + 1.0 / (k + rank + 1)
    for note_id, fts_rank in fts_ranked:
        scores[note_id] = scores.get(note_id, 0) + 1.0 / (k + fts_rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _graph_neighbors(
    db: BrainDB, file_paths: list[str], depth: int,
) -> dict[int, dict]:
    scored: dict[int, dict] = {}
    for fp in file_paths:
        stem = Path(fp).stem
        note = db.get_note_by_path(fp) or db.get_note_by_title(stem)
        if note is None:
            continue
        neighbor_ids = list(db.get_neighbor_ids(note["id"], depth=depth))[:MAX_BFS_NEIGHBORS]
        neighbors = db.get_notes_by_ids(neighbor_ids)
        for neighbor in neighbors:
            nid = neighbor["id"]
            if nid not in scored:
                scored[nid] = {
                    "note": neighbor,
                    "graph_score": 1.0,
                    "reason": f"backlink from {note['title']}",
                }
    return scored


def handle_brain_retrieve(
    db: BrainDB,
    vectors: VectorStore,
    embedder: EmbeddingBackend,
    query: str | None = None,
    region: str | None = None,
    limit: int = 10,
    threshold: float = 0.3,
    reranker: ReRanker | None = None,
    file_paths: list[str] | None = None,
    depth: int = 1,
    fts_only: bool = False,
) -> list[dict]:
    if not query and not file_paths:
        return [{"error": "Provide query, file_paths, or both"}]

    limit = max(1, min(limit, 100))
    depth = max(1, min(depth, 3))
    region_idx_filter = resolve_region_idx(region)

    graph_scored: dict[int, dict] = {}
    source_paths: set[str] = set()
    if file_paths:
        source_paths = set(file_paths)
        graph_scored = _graph_neighbors(db, file_paths, depth)

    semantic_scored: dict[int, dict] = {}
    if query:
        query = query[:1000]

        faiss_note_hits: dict[int, dict] = {}
        if not fts_only and vectors.size > 0:
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
                        faiss_note_hits[nid] = {"score": score, "note": note, "chunk": None}
                elif chunk is not None:
                    nid = chunk["note_id"]
                    if nid not in faiss_note_hits or score > faiss_note_hits[nid]["score"]:
                        faiss_note_hits[nid] = {"score": score, "note": None, "chunk": chunk}

        fts_results = db.fts_search(query, limit=limit * 3)
        fts_ranked = [(row["id"], rank) for rank, row in enumerate(fts_results)]
        fts_note_map = {row["id"]: row for row in fts_results}

        faiss_ranked = sorted(faiss_note_hits.items(), key=lambda x: x[1]["score"], reverse=True)
        faiss_for_rrf = [(nid, data["score"]) for nid, data in faiss_ranked]
        merged = _rrf_merge(faiss_for_rrf, fts_ranked)

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
            semantic_scored[note_id] = {
                "note": note_row,
                "semantic_score": rrf_score,
                "chunk": chunk_row,
            }

    all_ids = set(graph_scored.keys()) | set(semantic_scored.keys())
    results = []
    for nid in all_ids:
        graph = graph_scored.get(nid)
        semantic = semantic_scored.get(nid)

        note_row = None
        chunk_row = None
        if semantic:
            note_row = semantic["note"]
            chunk_row = semantic.get("chunk")
        if note_row is None and graph:
            note_row = graph["note"]
        if note_row is None:
            continue

        if note_row["path"] in source_paths:
            continue

        r_idx = note_row["region_idx"]
        if region_idx_filter is not None and r_idx != region_idx_filter:
            continue

        sem_score = semantic["semantic_score"] if semantic else 0.0
        graph_score = graph["graph_score"] if graph else 0.0

        if file_paths and query:
            combined = sem_score * 0.6 + graph_score * 0.4
        elif file_paths:
            combined = graph_score
        else:
            combined = sem_score

        entry = {
            "title": note_row["title"],
            "path": note_row["path"],
            "region": REGION_NAMES[r_idx] if 0 <= r_idx < 12 else "Stammhirn",
            "region_idx": r_idx,
            "similarity": round(combined, 4),
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

        if graph and semantic:
            entry["match_type"] = "graph+semantic"
        elif graph:
            entry["match_type"] = "graph"
            entry["relevance_reason"] = graph["reason"]

        results.append(entry)

    results.sort(key=lambda r: r["similarity"], reverse=True)
    results = results[:limit]

    if reranker is not None and query:
        results = reranker.rerank(query, results)

    return results
