# brain_mcp/tools/store.py
from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from typing import TYPE_CHECKING

from brain_mcp.indexer.embedder import EmbeddingBackend
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


def sanitize_title(title: str) -> str:
    title = _TRAVERSAL.sub("", title)
    title = _BAD_CHARS.sub("", title)
    return title.strip()[:MAX_TITLE_LEN]


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

    now = datetime.now(timezone.utc)
    content_hash = compute_content_hash(content)
    word_count = len(content.split())
    # Read old FAISS index before upsert to remove ghost vector
    old_row = db.get_note_by_path(rel_path)
    old_faiss_idx = old_row["faiss_idx"] if old_row and old_row["faiss_idx"] is not None else None

    note_id = db.upsert_note(
        path=rel_path, title=safe_title, content=content, content_hash=content_hash,
        region_idx=r_idx, tags=tags, word_count=word_count,
        created_at=now.strftime("%Y-%m-%d"),
        modified_at=now.isoformat(),
    )

    indexed = False
    if embedder.is_ready:
        try:
            vec = embedder.embed([content])
            if old_faiss_idx is not None:
                vectors.remove([old_faiss_idx])
            faiss_ids = vectors.add(vec)
            db.set_faiss_idx(note_id, faiss_ids[0])
            indexed = True
        except RuntimeError as exc:
            print(f"Embedding failed for '{title}': {exc}", file=sys.stderr)

    result = {
        "path": rel_path,
        "region": REGION_NAMES[r_idx],
        "region_idx": r_idx,
        "indexed": indexed,
        "word_count": word_count,
    }

    if title_was_sanitized:
        result["title_sanitized"] = True
        result["original_title"] = title
        result["safe_title"] = safe_title

    return result
