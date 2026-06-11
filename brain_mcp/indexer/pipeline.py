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
    exclude_dirs: list[str] | None = None,
) -> int:
    """Scan vault, skip unchanged files, upsert notes, embed new/changed, build edges.

    Returns the number of newly embedded notes.
    """
    notes = scan_vault(vault_path, folder_to_region, exclude_dirs=exclude_dirs)
    title_to_id: dict[str, int] = {}
    to_embed: list[tuple[int, str]] = []

    # Prune notes that vanished from the scan (deleted on disk OR moved into a
    # now-excluded dir). Otherwise a rebuild leaves orphan rows whose stale
    # faiss_idx collide with freshly assigned ids -> corrupt search. Fail-safes:
    # - never prune on an empty scan, so a transient scan glitch can't wipe the
    #   whole brain;
    # - a note absent from the scan whose file still EXISTS on disk (and is not
    #   excluded) was dropped by a transient read error -> keep it and log,
    #   because deleting would CASCADE its version history away.
    scanned_paths = {note["path"] for note in notes}
    if scanned_paths:
        excluded = {d.casefold() for d in (exclude_dirs or [])}
        keep_paths = set(scanned_paths)
        for db_path in db.get_all_note_paths():
            if db_path in keep_paths:
                continue
            parts = Path(db_path).parts
            intentionally_excluded = ".obsidian" in parts or (
                excluded and not excluded.isdisjoint(p.casefold() for p in parts)
            )
            if intentionally_excluded:
                continue  # excluded on purpose -> prune (history is tombstoned)
            try:
                still_on_disk = (vault_path / db_path).exists()
            except OSError as exc:
                print(f"WARNING: reconcile: cannot stat {db_path} ({exc}); keeping note",
                      file=sys.stderr)
                keep_paths.add(db_path)
                continue
            if still_on_disk:
                print(
                    f"WARNING: reconcile: {db_path} missing from scan but still on disk "
                    "(transient read error?); keeping note + version history",
                    file=sys.stderr,
                )
                keep_paths.add(db_path)
        stale_faiss = db.delete_notes_not_in(keep_paths)
        if stale_faiss and not force:
            vectors.remove(stale_faiss)

    if force:
        vectors.reset()
        # The store is empty now: any surviving stamp is stale by definition,
        # and the moment fresh ids get assigned (e.g. the chunks batch succeeds
        # while the notes batch fails) stale stamps collide with them — chunk
        # hits then map to the WRONG note. Clear stamps with the store.
        db.clear_all_faiss_idx()

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
        if not force:
            # Detach-then-embed (BrainDB.clear_faiss_idx): the surviving stamp
            # points at the OLD content's vector. Detach it NOW so a failed
            # embed batch leaves the row retryable (faiss_idx IS NULL) instead
            # of permanently mapped to stale content — the hash is already
            # updated, so later runs would skip it forever. On force the store
            # was reset and stamps bulk-cleared above.
            old = db.clear_faiss_idx(note_id)
            if old is not None:
                vectors.remove([old])
        title_to_id[note["title"]] = note_id
        to_embed.append((note_id, note["content"]))

    if to_embed:
        t0 = time.time()
        texts = [t for _, t in to_embed]
        try:
            vecs = embedder.embed(texts)
            faiss_ids = vectors.add(vecs)
            displaced = db.set_faiss_indices(
                [(note_id, fid) for (note_id, _), fid in zip(to_embed, faiss_ids)]
            )
            # Drop the vectors these stamps displaced (re-embedded notes kept
            # their old vector before -- an orphan leak on every edit). Never
            # on force: reset() already cleared the store and the stale ids
            # may now belong to freshly-added vectors (same guard as below).
            if displaced and not force:
                vectors.remove(displaced)
            elapsed = time.time() - t0
            print(f"Indexed {len(to_embed)} new/changed notes in {elapsed:.1f}s.", file=sys.stderr)
        except Exception as exc:
            print(f"Embedding error during indexing: {exc}", file=sys.stderr)

    # Build edges from backlinks (batched: one commit for the whole set -- the
    # per-edge commit was one WAL fsync PER BACKLINK on every writer startup).
    edge_rows: list[tuple[int, int, str]] = []
    for note in notes:
        src_id = title_to_id.get(note["title"])
        if src_id is None:
            continue
        for bl_title in note.get("backlink_titles", []):
            tgt_id = title_to_id.get(bl_title)
            if tgt_id and src_id != tgt_id:
                edge_rows.append((src_id, tgt_id, bl_title))
    db.upsert_edges(edge_rows)

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
            # On force the store was reset() -> these stale ids are either gone
            # or now belong to freshly-added note vectors; removing them would
            # silently delete a real note's vector. reset() already cleared them.
            if old_chunk_faiss and not force:
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
        # See note above: never feed stale chunk ids back into remove() on force,
        # or they collide with freshly-assigned note-vector ids (silent loss).
        if old_chunk_faiss and not force:
            vectors.remove(old_chunk_faiss)
        for chunk in chunks:
            chunks_to_embed.append((note_id, chunk["chunk_idx"], chunk["content"]))

    if chunks_to_embed:
        t0 = time.time()
        texts = [t for _, _, t in chunks_to_embed]
        try:
            vecs = embedder.embed(texts)
            faiss_ids = vectors.add(vecs)
            db.set_chunk_faiss_indices(
                [(note_id, chunk_idx, fid)
                 for (note_id, chunk_idx, _), fid in zip(chunks_to_embed, faiss_ids)]
            )
            elapsed = time.time() - t0
            print(f"Chunked {len(chunks_to_embed)} sections in {elapsed:.1f}s.", file=sys.stderr)
        except Exception as exc:
            print(f"Chunk embedding error: {exc}", file=sys.stderr)

    return len(to_embed)


def reindex_note_chunks(
    db: BrainDB,
    vectors: VectorStore,
    embedder: EmbeddingBackend,
    note_id: int,
    title: str,
    content: str,
) -> None:
    """Refresh ONE note's chunk vectors after an interactive edit (store/rollback).

    Without this, editing a note through brain_store/rollback left its old chunk
    vectors in FAISS, so search kept returning stale snippets for that note.
    """
    chunks = split_into_chunks(content, title)
    if not chunks:
        old = db.delete_chunks_for_note(note_id)
        if old:
            vectors.remove(old)
        return
    old = db.replace_chunks(note_id, chunks)
    if old:
        vectors.remove(old)
    vecs = embedder.embed([c["content"] for c in chunks])
    faiss_ids = vectors.add(vecs)
    db.set_chunk_faiss_indices(
        [(note_id, chunk["chunk_idx"], fid) for chunk, fid in zip(chunks, faiss_ids)]
    )
