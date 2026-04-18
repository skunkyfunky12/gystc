# brain_mcp/indexer/pipeline.py
"""Shared indexing pipeline used by server startup and CLI."""
from __future__ import annotations

import sys
import time
from pathlib import Path

from brain_mcp.indexer.embedder import EmbeddingBackend
from brain_mcp.indexer.scanner import scan_vault
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.storage.database import BrainDB


def index_vault(
    db: BrainDB,
    vectors: VectorStore,
    embedder: EmbeddingBackend,
    vault_path: Path,
    folder_to_region: dict[str, int],
    force: bool = False,
) -> int:
    """Scan vault, skip unchanged files, upsert notes, embed new/changed, build edges.

    Returns the number of newly embedded notes.
    """
    notes = scan_vault(vault_path, folder_to_region)
    title_to_id: dict[str, int] = {}
    to_embed: list[tuple[int, str]] = []
    old_faiss_ids: list[int] = []

    for note in notes:
        if not force:
            existing_hash = db.get_content_hash(note["path"])
            if existing_hash == note["content_hash"]:
                row = db.get_note_by_path(note["path"])
                if row:
                    title_to_id[note["title"]] = row["id"]
                continue

        # Collect old FAISS index before upsert to remove ghost vectors
        old_row = db.get_note_by_path(note["path"])
        if old_row and old_row["faiss_idx"] is not None:
            old_faiss_ids.append(old_row["faiss_idx"])

        note_id = db.upsert_note(
            path=note["path"], title=note["title"], content=note["content"],
            content_hash=note["content_hash"], region_idx=note["region_idx"],
            tags=note["tags"], word_count=note["word_count"],
            created_at=note["created_at"], modified_at=note["modified_at"],
        )
        title_to_id[note["title"]] = note_id
        to_embed.append((note_id, note["content"]))

    if to_embed:
        t0 = time.time()
        texts = [t for _, t in to_embed]
        try:
            vecs = embedder.embed(texts)
            faiss_ids = vectors.add(vecs)
            for (note_id, _), fid in zip(to_embed, faiss_ids):
                db.set_faiss_idx(note_id, fid)
            # Only remove old vectors after new ones are established
            if old_faiss_ids:
                vectors.remove(old_faiss_ids)
            elapsed = time.time() - t0
            print(f"Indexed {len(to_embed)} new/changed notes in {elapsed:.1f}s.", file=sys.stderr)
        except Exception as exc:
            print(f"Embedding error during indexing: {exc}", file=sys.stderr)
    elif old_faiss_ids:
        vectors.remove(old_faiss_ids)

    # Build edges from backlinks
    for note in notes:
        src_id = title_to_id.get(note["title"])
        if src_id is None:
            continue
        for bl_title in note.get("backlink_titles", []):
            tgt_id = title_to_id.get(bl_title)
            if tgt_id and src_id != tgt_id:
                db.upsert_edge(src_id, tgt_id, link_text=bl_title)

    return len(to_embed)
