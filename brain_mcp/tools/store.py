# brain_mcp/tools/store.py
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

from brain_mcp.indexer.embedder import EmbeddingBackend
from brain_mcp.indexer.scanner import REGION_TAG_TO_IDX, compute_content_hash
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.recent import REGION_NAMES, REGION_NAME_TO_IDX

_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_TRAVERSAL = re.compile(r'\.\.[\\/]?')
_BRAIN_TAG_RE = re.compile(r'\n*#brain/[\w-]+\n?')
MAX_CONTENT_SIZE = 1024 * 1024  # 1MB
MAX_TITLE_LEN = 200
MAX_TAGS = 20
MAX_TAG_LEN = 100

# REVIEW FIX: Robust REGION_NAME_TO_SLUG construction
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
    pending_writes: dict[str, float] | None = None,
) -> dict:
    if len(content) > MAX_CONTENT_SIZE:
        return {"error": f"Content exceeds {MAX_CONTENT_SIZE} byte limit"}

    safe_title = sanitize_title(title)
    if not safe_title:
        return {"error": "Title is empty after sanitization"}

    # REVIEW FIX: Warn on sanitized title
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

    # REVIEW FIX: Return error for invalid region instead of silently defaulting
    if region and region not in REGION_NAME_TO_IDX:
        return {"error": f"Unknown region: {region}. Use brain_regions(action='list')."}

    if region_idx is not None and 0 <= region_idx < 12:
        r_idx = region_idx
    elif region:
        r_idx = REGION_NAME_TO_IDX[region]
    else:
        r_idx = 9  # Stammhirn default

    # REVIEW FIX: Strip existing #brain/ tags before appending new one
    content = _BRAIN_TAG_RE.sub('', content).rstrip()
    region_slug = REGION_NAME_TO_SLUG.get(REGION_NAMES[r_idx])
    if region_slug:
        content += f"\n\n#brain/{region_slug}\n"

    target_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = target.with_suffix(".md.tmp")

    # REVIEW FIX: Register pending_writes BEFORE os.replace to prevent watcher race
    if pending_writes is not None:
        pending_writes[str(resolved)] = datetime.now(timezone.utc).timestamp()

    try:
        tmp_file.write_text(content, encoding="utf-8")
        os.replace(str(tmp_file), str(target))
    except OSError as e:
        # Undo pending_writes registration on failure
        if pending_writes is not None:
            pending_writes.pop(str(resolved), None)
        return {"error": f"Write failed: {e}"}

    rel_path = str(target.relative_to(vault_root)).replace("\\", "/")

    # REVIEW FIX: Use UTC timestamps
    now = datetime.now(timezone.utc)
    content_hash = compute_content_hash(content)
    word_count = len(content.split())
    note_id = db.upsert_note(
        path=rel_path, title=safe_title, content=content, content_hash=content_hash,
        region_idx=r_idx, tags=tags, word_count=word_count,
        created_at=now.strftime("%Y-%m-%d"),
        modified_at=now.isoformat(),
    )

    vec = embedder.embed([content])
    faiss_ids = vectors.add(vec)
    db.set_faiss_idx(note_id, faiss_ids[0])

    result = {
        "path": rel_path,
        "region": REGION_NAMES[r_idx],
        "region_idx": r_idx,
        "indexed": True,
        "word_count": word_count,
    }

    # REVIEW FIX: Warn on sanitized title
    if title_was_sanitized:
        result["title_sanitized"] = True
        result["original_title"] = title
        result["safe_title"] = safe_title

    return result
