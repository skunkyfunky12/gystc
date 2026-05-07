# brain_mcp/indexer/pipeline.py
"""Shared indexing pipeline used by server startup and CLI."""
from __future__ import annotations

import sys
import time
from pathlib import Path

from brain_mcp.indexer.chunker import split_into_chunks
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

    if force:
        vectors.reset()

    for note in notes:
        if not force:
            existing_hash = db.get_content_hash(note["path"])
            if existing_hash == note["content_hash"]:
                row = db.get_note_by_path(note["path"])
                if row:
                    title_to_id[note["title"]] = row["id"]
                    if row["faiss_idx"] is None:
                        to_embed.append((row["id"], note["content"]))
                continue

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
            elapsed = time.time() - t0
            print(f"Indexed {len(to_embed)} new/changed notes in {elapsed:.1f}s.", file=sys.stderr)
        except Exception as exc:
            print(f"Embedding error during indexing: {exc}", file=sys.stderr)

    # Build edges from backlinks
    for note in notes:
        src_id = title_to_id.get(note["title"])
        if src_id is None:
            continue
        for bl_title in note.get("backlink_titles", []):
            tgt_id = title_to_id.get(bl_title)
            if tgt_id and src_id != tgt_id:
                db.upsert_edge(src_id, tgt_id, link_text=bl_title)

    # --- Chunking phase ---
    chunks_to_embed: list[tuple[int, int, str]] = []  # (note_id, chunk_idx, content)
    for note in notes:
        note_id = title_to_id.get(note["title"])
        if note_id is None:
            row = db.get_note_by_path(note["path"])
            if row:
                note_id = row["id"]
            else:
                continue

        chunks = split_into_chunks(note["content"], note["title"])
        if not chunks:
            old_chunk_faiss = db.delete_chunks_for_note(note_id)
            if old_chunk_faiss:
                vectors.remove(old_chunk_faiss)
            continue

        existing = db.get_chunks_for_note(note_id)
        existing_hashes = {c["chunk_idx"]: c["content_hash"] for c in existing}
        needs_update = force or len(chunks) != len(existing) or any(
            c["chunk_idx"] not in existing_hashes
            or existing_hashes[c["chunk_idx"]] != c["content_hash"]
            for c in chunks
        )
        if not needs_update:
            if all(c["faiss_idx"] is not None for c in existing):
                continue

        old_chunk_faiss = db.replace_chunks(note_id, chunks)
        if old_chunk_faiss:
            vectors.remove(old_chunk_faiss)
        for chunk in chunks:
            chunks_to_embed.append((note_id, chunk["chunk_idx"], chunk["content"]))

    if chunks_to_embed:
        t0 = time.time()
        texts = [t for _, _, t in chunks_to_embed]
        try:
            vecs = embedder.embed(texts)
            faiss_ids = vectors.add(vecs)
            for (note_id, chunk_idx, _), fid in zip(chunks_to_embed, faiss_ids):
                db.set_chunk_faiss_idx(note_id, chunk_idx, fid)
            elapsed = time.time() - t0
            print(f"Chunked {len(chunks_to_embed)} sections in {elapsed:.1f}s.", file=sys.stderr)
        except Exception as exc:
            print(f"Chunk embedding error: {exc}", file=sys.stderr)

    return len(to_embed)
