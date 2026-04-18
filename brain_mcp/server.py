from __future__ import annotations
import json
import re
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from brain_mcp.config import BrainConfig, load_config
from brain_mcp.indexer.embedder import SentenceTransformerBackend
from brain_mcp.indexer.scanner import scan_vault, compute_content_hash, REGION_TAG_TO_IDX, _BRAIN_TAG_RE
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.indexer.watcher import BrainWatcher
from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.recent import handle_brain_recent
from brain_mcp.tools.regions import handle_brain_regions
from brain_mcp.tools.retrieve import handle_brain_retrieve
from brain_mcp.tools.store import handle_brain_store
from brain_mcp.tools.related import handle_brain_related
from brain_mcp.tools.context import handle_brain_context

@dataclass
class BrainState:
    config: BrainConfig
    db: BrainDB
    vectors: VectorStore
    embedder: SentenceTransformerBackend
    watcher: BrainWatcher | None = None


def _index_vault(state: BrainState) -> None:
    """Scan vault, skip unchanged files, upsert notes, embed new/changed, build edges."""
    if state.config.vault_path is None or not state.config.vault_path.is_dir():
        return
    notes = scan_vault(state.config.vault_path, state.config.folder_to_region)
    title_to_id: dict[str, int] = {}
    to_embed: list[tuple[int, str]] = []

    for note in notes:
        existing_hash = state.db.get_content_hash(note["path"])
        if existing_hash == note["content_hash"]:
            row = state.db.get_note_by_path(note["path"])
            if row:
                title_to_id[note["title"]] = row["id"]
            continue
        note_id = state.db.upsert_note(
            path=note["path"], title=note["title"], content=note["content"],
            content_hash=note["content_hash"], region_idx=note["region_idx"],
            tags=note["tags"], word_count=note["word_count"],
            created_at=note["created_at"], modified_at=note["modified_at"],
        )
        title_to_id[note["title"]] = note_id
        to_embed.append((note_id, note["content"]))

    if to_embed:
        texts = [t for _, t in to_embed]
        try:
            vecs = state.embedder.embed(texts)
            faiss_ids = state.vectors.add(vecs)
            for (note_id, _), fid in zip(to_embed, faiss_ids):
                state.db.set_faiss_idx(note_id, fid)
        except Exception as exc:
            print(f"Embedding error during indexing: {exc}", file=sys.stderr)
        print(f"Indexed {len(to_embed)} new/changed notes.", file=sys.stderr)

    for note in notes:
        src_id = title_to_id.get(note["title"])
        if src_id is None:
            continue
        for bl_title in note.get("backlink_titles", []):
            tgt_id = title_to_id.get(bl_title)
            if tgt_id and src_id != tgt_id:
                state.db.upsert_edge(src_id, tgt_id, link_text=bl_title)


def _handle_file_change(state: BrainState, path: str, event_type: str) -> None:
    """Callback for the watcher: re-index a single file on create/modify/delete."""
    try:
        rel = str(Path(path).relative_to(state.config.vault_path)).replace("\\", "/")
    except ValueError:
        return
    if event_type == "deleted":
        state.db.delete_note(rel)
        return
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return

    content_hash = compute_content_hash(text)
    if state.db.get_content_hash(rel) == content_hash:
        return

    brain_tags = _BRAIN_TAG_RE.findall(text)
    region_idx = REGION_TAG_TO_IDX.get(brain_tags[0], 9) if brain_tags else 9
    all_tags = list(set(re.findall(r"#[\w/-]+", text)))[:20]
    word_count = len(text.split())
    now = datetime.now(timezone.utc)

    note_id = state.db.upsert_note(
        path=rel, title=Path(path).stem, content=text, content_hash=content_hash,
        region_idx=region_idx, tags=all_tags, word_count=word_count,
        created_at=now.strftime("%Y-%m-%d"), modified_at=now.isoformat(),
    )
    try:
        vec = state.embedder.embed([text])
        faiss_ids = state.vectors.add(vec)
        state.db.set_faiss_idx(note_id, faiss_ids[0])
    except Exception as exc:
        print(f"Embedding error for {rel}: {exc}", file=sys.stderr)
    print(f"Re-indexed: {rel}", file=sys.stderr)


@asynccontextmanager
async def brain_lifespan(server: FastMCP) -> AsyncIterator[BrainState]:
    config = load_config()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    db = BrainDB(config.db_path)
    vectors = VectorStore.load(config.index_path, dimension=384)
    embedder = SentenceTransformerBackend(config.model_name)
    state = BrainState(config=config, db=db, vectors=vectors, embedder=embedder)
    print(f"Brain MCP Server started. Vault: {config.vault_path}", file=sys.stderr)
    print(f"DB: {config.db_path} | Index: {vectors.size} vectors", file=sys.stderr)

    # Startup indexing
    vault_exists = config.vault_path is not None and config.vault_path.is_dir()
    if vault_exists and config.index_on_startup:
        _index_vault(state)

    # File watcher
    if vault_exists and config.auto_index:
        watcher = BrainWatcher(config.vault_path, lambda p, e: _handle_file_change(state, p, e))
        watcher.start()
        state.watcher = watcher

    try:
        yield state
    finally:
        if state.watcher is not None:
            state.watcher.stop()
        vectors.save(config.index_path)
        db.close()
        print("Brain MCP Server stopped.", file=sys.stderr)

mcp = FastMCP("Neural Brain", lifespan=brain_lifespan)

@mcp.tool()
def brain_recent(days: int = 7, region: str | None = None, limit: int = 20) -> list[dict]:
    """Show recently modified notes in the vault.
    Args:
        days: Lookback window in days (default 7, max 365)
        region: Filter by brain region name (e.g. "Hippocampus")
        limit: Max results (default 20, max 100)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    return handle_brain_recent(state.db, days=days, region=region, limit=limit)

@mcp.tool()
def brain_regions(action: str, region: str | None = None, description: str | None = None, color: str | None = None) -> dict | list[dict]:
    """List, describe, or customize brain region definitions.
    Args:
        action: "list" (all regions), "describe" (one region detail), or "customize" (update)
        region: Region name (required for describe/customize)
        description: New description (customize only)
        color: New hex color like #FF0000 (customize only)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    return handle_brain_regions(state.db, action=action, region=region, description=description, color=color)

@mcp.tool()
def brain_retrieve(query: str, region: str | None = None, limit: int = 10, threshold: float = 0.3) -> list[dict]:
    """Semantic search across all vault notes. Finds notes by meaning, not just keywords.

    Args:
        query: Natural language search query
        region: Filter by brain region name (e.g. "Hippocampus")
        limit: Max results (default 10, max 100)
        threshold: Min cosine similarity (default 0.3)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    return handle_brain_retrieve(state.db, state.vectors, state.embedder, query=query, region=region, limit=limit, threshold=threshold)

@mcp.tool()
def brain_store(title: str, content: str, region: str | None = None, region_idx: int | None = None,
                tags: list[str] | None = None, folder: str = "") -> dict:
    """Create or update a note in the vault.

    Args:
        title: Note title (becomes filename)
        content: Markdown content
        region: Brain region name (auto-detected if omitted)
        region_idx: Region index (overrides name)
        tags: Additional tags (max 20)
        folder: Subfolder in vault
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    if state.config.vault_path is None:
        return {"error": "No vault_path configured"}
    return handle_brain_store(
        state.db, state.vectors, state.embedder, state.config.vault_path,
        title=title, content=content, region=region, region_idx=region_idx,
        tags=tags, folder=folder, pending_writes=state.watcher._pending_writes if state.watcher else {},
    )

@mcp.tool()
def brain_related(title: str | None = None, path: str | None = None, limit: int = 10) -> list[dict] | dict:
    """Find notes related to a specific note by meaning or backlinks.

    Args:
        title: Note title to find relations for
        path: Note path (alternative to title)
        limit: Max results (default 10, max 100)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    return handle_brain_related(state.db, state.vectors, state.embedder, title=title, path=path, limit=limit)

@mcp.tool()
def brain_context(file_paths: list[str] | None = None, task_description: str | None = None,
                  depth: int = 1, max_notes: int = 10) -> list[dict] | dict:
    """Get contextually relevant notes for current work. Combines file proximity and semantic search.

    Args:
        file_paths: Files currently being edited
        task_description: What you are doing
        depth: Backlink graph hops 1-3 (default 1)
        max_notes: Max notes returned (default 10, max 100)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    return handle_brain_context(state.db, state.vectors, state.embedder,
                                 file_paths=file_paths, task_description=task_description,
                                 depth=depth, max_notes=max_notes)

# ---------------------------------------------------------------------------
# MCP Resources
# ---------------------------------------------------------------------------

@mcp.resource("brain://regions")
def resource_regions() -> str:
    """List all brain regions with note counts."""
    state: BrainState = mcp.get_context().request_context.lifespan_context
    result = handle_brain_regions(state.db, action="list")
    return json.dumps(result, indent=2)

@mcp.resource("brain://recent")
def resource_recent() -> str:
    """Recently modified notes (last 7 days, up to 20)."""
    state: BrainState = mcp.get_context().request_context.lifespan_context
    result = handle_brain_recent(state.db, days=7, limit=20)
    return json.dumps(result, indent=2)

@mcp.resource("brain://stats")
def resource_stats() -> str:
    """Overall vault statistics."""
    state: BrainState = mcp.get_context().request_context.lifespan_context
    notes = state.db.get_all_notes()
    counts = state.db.get_region_note_counts()
    return json.dumps({
        "total_notes": len(notes),
        "total_vectors": state.vectors.size,
        "regions_with_notes": len(counts),
        "vault_path": str(state.config.vault_path),
    }, indent=2)
