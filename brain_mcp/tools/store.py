# brain_mcp/tools/store.py
from __future__ import annotations

import os
import re
import sys
from datetime import datetime, UTC
from pathlib import Path

from typing import TYPE_CHECKING

from brain_mcp.indexer.embedder import EmbeddingBackend
from brain_mcp.indexer.pipeline import reindex_note_chunks
from brain_mcp.indexer.scanner import REGION_TAG_TO_IDX, compute_content_hash
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.classifier import classify_region
from brain_mcp.tools.recent import REGION_NAMES, resolve_region_idx

if TYPE_CHECKING:
    from brain_mcp.indexer.watcher import BrainWatcher

_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_TRAVERSAL = re.compile(r'\.\.[\\/]?')
_BRAIN_TAG_RE = re.compile(r'\n*#brain/[\w-]+\n?')
MAX_CONTENT_SIZE = 1024 * 1024  # 1MB
MAX_TITLE_LEN = 200
MAX_TAGS = 20
MAX_TAG_LEN = 100

IDX_TO_SLUG = {idx: slug for slug, idx in REGION_TAG_TO_IDX.items()}
REGION_NAME_TO_SLUG = {REGION_NAMES[idx]: slug for idx, slug in IDX_TO_SLUG.items()}


# Windows resolves these names to devices no matter the extension or directory,
# so "CON" would ask the OS for the console instead of creating CON.md. Kept as a
# fixed set (Windows only defines these) rather than probing the platform: a
# vault written on macOS is often opened on Windows, so the file must be
# creatable everywhere, not just where it was stored.
_WINDOWS_DEVICE_NAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + [f"COM{i}" for i in range(1, 10)]
    + [f"LPT{i}" for i in range(1, 10)]
)


def sanitize_title(title: str) -> str:
    title = _TRAVERSAL.sub("", title)
    title = _BAD_CHARS.sub("", title)
    title = title.strip()[:MAX_TITLE_LEN]
    # The device name is whatever precedes the first dot, so "CON.backup" is
    # reserved too. A trailing underscore is enough to make the name ordinary
    # and keeps the title readable.
    stem, dot, rest = title.partition(".")
    if stem.strip().upper() in _WINDOWS_DEVICE_NAMES:
        title = f"{stem}_{dot}{rest}"
    return title


def handle_brain_store(
    db: BrainDB,
    vectors: VectorStore,
    embedder: EmbeddingBackend,
    vault_root: Path,
    title: str,
    content: str,
    region: str | None = None,
    region_idx: int | None = None,
    tags: list[str] | None = None,
    folder: str = "",
    watcher: BrainWatcher | None = None,
    persist: bool = True,
) -> dict:
    if len(content) > MAX_CONTENT_SIZE:
        return {"error": f"Content exceeds {MAX_CONTENT_SIZE} byte limit"}

    if region_idx is not None and not (0 <= region_idx < 12):
        return {"error": f"region_idx must be 0-11, got {region_idx}"}

    safe_title = sanitize_title(title)
    if not safe_title:
        return {"error": "Title is empty after sanitization"}

    title_was_sanitized = safe_title != title

    tags = (tags or [])[:MAX_TAGS]
    tags = [t[:MAX_TAG_LEN] for t in tags]

    # SECURITY: Reject folder if it contains path traversal attempts
    if _TRAVERSAL.search(folder):
        return {"error": "Path traversal in folder not allowed"}

    folder_clean = folder.strip("/\\")
    target_dir = (vault_root / folder_clean) if folder_clean else vault_root
    target = target_dir / f"{safe_title}.md"

    try:
        resolved = target.resolve()
        if not resolved.is_relative_to(vault_root.resolve()):
            return {"error": "Path escapes vault directory"}
    except (ValueError, OSError):
        return {"error": "Invalid path"}

    # Return error for invalid region instead of silently defaulting
    if region and resolve_region_idx(region) is None:
        return {"error": f"Unknown region: {region}. Use brain_regions(action='list')."}

    if region_idx is not None and 0 <= region_idx < 12:
        r_idx = region_idx
    elif region:
        r_idx = resolve_region_idx(region)  # type: ignore[assignment]
    else:
        rel_path = f"{folder_clean}/{safe_title}.md" if folder_clean else f"{safe_title}.md"
        r_idx = classify_region(safe_title, content, path=rel_path)

    content = _BRAIN_TAG_RE.sub('', content).rstrip()
    region_slug = REGION_NAME_TO_SLUG.get(REGION_NAMES[r_idx])
    if region_slug:
        content += f"\n\n#brain/{region_slug}\n"

    target_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = target.with_suffix(".md.tmp")

    # Register pending write BEFORE os.replace to prevent watcher race
    if watcher is not None:
        watcher.add_pending_write(str(resolved))

    try:
        tmp_file.write_text(content, encoding="utf-8")
        os.replace(str(tmp_file), str(target))
    except OSError as e:
        return {"error": f"Write failed: {e}"}

    rel_path = str(target.relative_to(vault_root)).replace("\\", "/")

    now = datetime.now(UTC)
    content_hash = compute_content_hash(content)
    word_count = len(content.split())

    # Read-only (non-writer) instances write the .md file but never touch
    # brain.db / index.faiss — the single writer instance owns indexing and will
    # pick this file up via its watcher. This prevents coexisting servers from
    # clobbering the shared FAISS index and faiss_idx column.
    indexed = False
    if persist:
        note_id = db.upsert_note(
            path=rel_path, title=safe_title, content=content, content_hash=content_hash,
            region_idx=r_idx, tags=tags, word_count=word_count,
            created_at=now.strftime("%Y-%m-%d"),
            modified_at=now.isoformat(),
        )
        # Detach-then-embed: the surviving stamp (upsert keeps it via COALESCE)
        # points at the OLD content's vector. Detach before embedding so a
        # failed embed leaves the row retryable (faiss_idx IS NULL, picked up
        # by the reconcile) instead of permanently mapped to stale content.
        old = db.clear_faiss_idx(note_id)
        if old is not None:
            vectors.remove([old])

        if embedder.is_ready:
            try:
                vec = embedder.embed([content])
                faiss_ids = vectors.add(vec)
                displaced = db.set_faiss_idx(note_id, faiss_ids[0])
                if displaced is not None:
                    # A racing reconcile pass stamped this note between our
                    # detach and our stamp -- drop the loser, never leak it.
                    vectors.remove([displaced])
                # Refresh chunk vectors too — otherwise search returns stale snippets
                # for this note after every edit.
                reindex_note_chunks(db, vectors, embedder, note_id, safe_title, content)
                indexed = True
            except Exception as exc:
                print(f"Embedding failed for '{title}': {exc}", file=sys.stderr)

    result = {
        "path": rel_path,
        "region": REGION_NAMES[r_idx],
        "region_idx": r_idx,
        "indexed": indexed,
        "word_count": word_count,
    }
    if not persist:
        result["persisted_by"] = "primary"

    if title_was_sanitized:
        result["title_sanitized"] = True
        result["original_title"] = title
        result["safe_title"] = safe_title

    return result
