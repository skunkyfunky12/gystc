from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from brain_mcp.indexer.embedder import EmbeddingBackend
from brain_mcp.indexer.pipeline import reindex_note_chunks
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.storage.database import BrainDB

if TYPE_CHECKING:
    from brain_mcp.indexer.watcher import BrainWatcher


def _safe_parse_tags(tags_str) -> list[str]:
    if not tags_str:
        return []
    try:
        parsed = json.loads(tags_str)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def handle_brain_history(db: BrainDB, path: str) -> list[dict] | dict:
    note = db.get_note_by_path(path)
    if note is None:
        return {"error": f"Note not found: {path}"}
    versions = db.get_versions(note["id"], limit=50)
    return [
        {
            "id": v["id"],
            "content_hash": v["content_hash"],
            "title": v["title"],
            "word_count": v["word_count"],
            "versioned_at": v["versioned_at"],
            "reason": v["reason"],
        }
        for v in versions
    ]


def handle_brain_diff(db: BrainDB, path: str, version_id: int) -> dict:
    note = db.get_note_by_path(path)
    if note is None:
        return {"error": f"Note not found: {path}"}
    version = db.get_version(version_id)
    if version is None:
        return {"error": f"Version not found: {version_id}"}
    if version["note_id"] != note["id"]:
        return {"error": "Version does not belong to this note"}

    current_lines = (note["content"] or "").splitlines(keepends=True)
    version_lines = version["content"].splitlines(keepends=True)
    diff = difflib.unified_diff(
        version_lines, current_lines,
        fromfile=f"{path} (version {version_id})",
        tofile=f"{path} (current)",
    )
    return {
        "diff": "".join(diff),
        "version_id": version_id,
        "versioned_at": version["versioned_at"],
        "current_hash": note["content_hash"],
        "version_hash": version["content_hash"],
    }


def handle_brain_rollback(
    db: BrainDB,
    vault_path: Path,
    path: str,
    version_id: int,
    watcher: BrainWatcher | None = None,
    vectors: VectorStore | None = None,
    embedder: EmbeddingBackend | None = None,
    persist: bool = True,
) -> dict:
    note = db.get_note_by_path(path)
    if note is None:
        return {"error": f"Note not found: {path}"}
    version = db.get_version(version_id)
    if version is None:
        return {"error": f"Version not found: {version_id}"}
    if version["note_id"] != note["id"]:
        return {"error": "Version does not belong to this note"}

    file_path = vault_path / path
    if not file_path.resolve().is_relative_to(vault_path.resolve()):
        return {"error": "Path escapes vault directory"}
    if not file_path.exists():
        return {"error": f"File not found in vault: {path}"}

    resolved = str(file_path.resolve())
    if watcher is not None:
        watcher.add_pending_write(resolved)

    file_path.write_text(version["content"], encoding="utf-8")

    restored_hash = version["content_hash"]
    restored_tags = _safe_parse_tags(version["tags"])

    # Read-only instances only restore the file; the writer instance owns the DB
    # + index and re-indexes the restored file via its watcher.
    reindexed = False
    if persist:
        old_faiss_idx = note["faiss_idx"] if note["faiss_idx"] is not None else None
        note_id = db.upsert_note(
            path=path,
            title=version["title"],
            content=version["content"],
            content_hash=restored_hash,
            region_idx=version["region_idx"],
            tags=restored_tags, word_count=version["word_count"],
            created_at=note["created_at"],
            modified_at=note["modified_at"],
        )

        # The watcher is suppressed for this write (pending-write), so it won't
        # re-index for us. Drop the stale vector and re-embed the restored content,
        # or search would keep returning the pre-rollback version.
        if vectors is not None and old_faiss_idx is not None:
            vectors.remove([old_faiss_idx])
        if vectors is not None and embedder is not None and embedder.is_ready:
            try:
                fid = vectors.add(embedder.embed([version["content"]]))[0]
                db.set_faiss_idx(note_id, fid)
                reindex_note_chunks(db, vectors, embedder, note_id,
                                    version["title"], version["content"])
                reindexed = True
            except Exception as exc:
                print(f"Rollback re-embed failed for {path}: {exc}", file=sys.stderr)

    return {
        "rolled_back": True,
        "path": path,
        "version_id": version_id,
        "restored_hash": restored_hash,
        "tags": restored_tags,
        "reindexed": reindexed,
    }
