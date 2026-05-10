from __future__ import annotations
import asyncio
import json
import sys
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from brain_mcp.config import BrainConfig, load_config
from brain_mcp.indexer.embedder import SentenceTransformerBackend
from brain_mcp.indexer.pipeline import index_vault
from brain_mcp.indexer.scanner import parse_note_file
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.indexer.watcher import BrainWatcher
from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.recent import handle_brain_recent
from brain_mcp.tools.regions import handle_brain_regions
from brain_mcp.tools.retrieve import handle_brain_retrieve
from brain_mcp.tools.store import handle_brain_store
from brain_mcp.tools.related import handle_brain_related
from brain_mcp.tools.classify_tool import handle_brain_classify, handle_brain_classify_feedback
from brain_mcp.tools.versioning import handle_brain_history, handle_brain_diff, handle_brain_rollback
from brain_mcp.indexer.reranker import CrossEncoderReRanker

WAIT_READY_TIMEOUT = 10
TOOL_TIMEOUT = 30


@dataclass
class BrainState:
    config: BrainConfig
    db: BrainDB
    vectors: VectorStore
    embedder: SentenceTransformerBackend
    watcher: BrainWatcher | None = None
    reranker: CrossEncoderReRanker | None = None


def _index_vault(state: BrainState) -> None:
    if state.config.vault_path is None or not state.config.vault_path.is_dir():
        return
    index_vault(state.db, state.vectors, state.embedder,
                state.config.vault_path, state.config.folder_to_region)


def _handle_file_change(state: BrainState, path: str, event_type: str) -> None:
    try:
        rel = str(Path(path).relative_to(state.config.vault_path)).replace("\\", "/")
    except ValueError:
        return
    if event_type == "deleted":
        old_row = state.db.get_note_by_path(rel)
        if old_row and old_row["faiss_idx"] is not None:
            state.vectors.remove([old_row["faiss_idx"]])
        state.db.delete_note(rel)
        return

    note = parse_note_file(Path(path), state.config.vault_path, state.config.folder_to_region)
    if note is None:
        return

    if state.db.get_content_hash(rel) == note["content_hash"]:
        return

    old_row = state.db.get_note_by_path(rel)
    old_faiss_idx = old_row["faiss_idx"] if old_row and old_row["faiss_idx"] is not None else None

    note_id = state.db.upsert_note(
        path=note["path"], title=note["title"], content=note["content"],
        content_hash=note["content_hash"], region_idx=note["region_idx"],
        tags=note["tags"], word_count=note["word_count"],
        created_at=note["created_at"], modified_at=note["modified_at"],
    )
    try:
        vec = state.embedder.embed([note["content"]])
        faiss_ids = state.vectors.add(vec)
        state.db.set_faiss_idx(note_id, faiss_ids[0])
        if old_faiss_idx is not None:
            state.vectors.remove([old_faiss_idx])
        print(f"Re-indexed: {rel}", file=sys.stderr)
    except Exception as exc:
        print(f"Embedding error for {rel}: {exc}", file=sys.stderr)


def _background_startup(state: BrainState, model_thread: threading.Thread) -> None:
    try:
        vault_exists = state.config.vault_path is not None and state.config.vault_path.is_dir()
        model_thread.join(timeout=120)
        if model_thread.is_alive():
            print("WARNING: Model still loading after 120s, skipping startup indexing.", file=sys.stderr)
            return
        if not state.embedder.is_ready:
            print("ERROR: Model thread finished but model not ready.", file=sys.stderr)
            return
        if vault_exists and state.config.index_on_startup:
            try:
                _index_vault(state)
            except Exception as exc:
                print(f"ERROR: Startup indexing failed: {exc}", file=sys.stderr)
        if vault_exists and state.config.auto_index:
            watcher = BrainWatcher(state.config.vault_path, lambda p, e: _handle_file_change(state, p, e))
            watcher.start()
            state.watcher = watcher
        print("Background startup complete.", file=sys.stderr)
    except Exception as exc:
        import traceback
        print(f"ERROR: Background startup crashed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


@asynccontextmanager
async def brain_lifespan(server: FastMCP) -> AsyncIterator[BrainState]:
    import time
    t0 = time.perf_counter()
    config = load_config()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    db = BrainDB(config.db_path)
    embedder = SentenceTransformerBackend(config.model_name)
    model_thread = threading.Thread(target=embedder._load, daemon=True)
    model_thread.start()
    vectors = VectorStore.load(config.index_path, dimension=embedder.dimension)
    reranker = None
    if config.reranker == "cross-encoder":
        reranker = CrossEncoderReRanker()
        reranker.start_loading()
    state = BrainState(config=config, db=db, vectors=vectors, embedder=embedder, reranker=reranker)
    startup_ms = int((time.perf_counter() - t0) * 1000)
    print(f"GYSTC MCP started in {startup_ms}ms. Vault: {config.vault_path}", file=sys.stderr)
    print(f"DB: {config.db_path} | Index: {vectors.size} vectors", file=sys.stderr)

    bg = threading.Thread(target=_background_startup, args=(state, model_thread), daemon=True)
    bg.start()

    try:
        yield state
    finally:
        if state.watcher is not None:
            state.watcher.stop()
        try:
            vectors.save(config.index_path)
        except Exception as exc:
            print(f"ERROR: Failed to save FAISS index: {exc}", file=sys.stderr)
        try:
            db.close()
        except Exception as exc:
            print(f"ERROR: Failed to close database: {exc}", file=sys.stderr)
        print("GYSTC MCP stopped.", file=sys.stderr)

BRAIN_INSTRUCTIONS = """
You have access to a persistent knowledge vault organized into 12 brain regions.

TOOLS (8 total):
- brain_retrieve: Search by query and/or file context. Primary search tool.
- brain_store: Save important knowledge as a vault note.
- brain_related: Find notes connected to a specific note.
- brain_recent: See recently changed notes (instant, no model needed).
- brain_status: Health check — note counts, model status (instant).
- brain_regions: List or describe brain regions (instant).
- brain_classify: Classify notes into regions, correct misclassifications, or batch-reclassify.
- brain_versions: View history, diff, or rollback note versions.

WHEN TO SEARCH:
- User asks about a past decision, architecture, or project context.
- You need project-specific conventions not in the code.

WHEN NOT TO SEARCH:
- General programming questions — use your own knowledge.
- The user is asking you to write code — just write it.
- You already have enough context from the conversation.
- You just searched and got results — don't search again with a rephrased query.

KEEP IT FAST:
- brain_recent and brain_status need no model — use them when the model is still loading.
- brain_retrieve with only file_paths does graph traversal without embeddings.
- Don't chain multiple search calls. One brain_retrieve is usually enough.
""".strip()

mcp = FastMCP("GYSTC", lifespan=brain_lifespan, instructions=BRAIN_INSTRUCTIONS)

# ---------------------------------------------------------------------------
# INSTANT TOOLS (no embedding model required)
# ---------------------------------------------------------------------------

@mcp.tool()
def brain_status() -> dict:
    """Fast health check. Note/edge counts, model status, region distribution. Always instant."""
    state: BrainState = mcp.get_context().request_context.lifespan_context
    counts = state.db.get_region_note_counts()
    edge_types = state.db.get_edge_type_counts()
    from brain_mcp.tools.recent import REGION_NAMES
    region_dist = {
        REGION_NAMES[idx]: cnt
        for idx, cnt in sorted(counts.items())
        if 0 <= idx < 12
    }
    return {
        "total_notes": state.db.get_note_count(),
        "total_vectors": state.vectors.size,
        "total_edges": sum(edge_types.values()),
        "edge_types": edge_types,
        "regions": region_dist,
        "model_loaded": state.embedder.is_ready,
        "reranker_enabled": state.reranker is not None,
        "reranker_loaded": state.reranker.is_ready if state.reranker else False,
        "vault_path": str(state.config.vault_path),
    }

@mcp.tool()
def brain_recent(days: int = 7, region: str | None = None, limit: int = 20) -> list[dict]:
    """Recently changed notes. Instant, no model needed.

    Args:
        days: Lookback window (default 7, max 365)
        region: Filter by region name
        limit: Max results (default 20, max 100)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    return handle_brain_recent(state.db, days=days, region=region, limit=limit)

@mcp.tool()
def brain_regions(action: str, region: str | None = None, description: str | None = None, color: str | None = None) -> dict | list[dict]:
    """List, describe, or customize brain regions.

    Args:
        action: "list", "describe", or "customize"
        region: Region name (required for describe/customize)
        description: New description (customize only)
        color: Hex color like #FF0000 (customize only)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    return handle_brain_regions(state.db, action=action, region=region, description=description, color=color)

# ---------------------------------------------------------------------------
# EMBEDDING TOOLS (need model, have FTS fallback + timeouts)
# ---------------------------------------------------------------------------

@mcp.tool()
async def brain_retrieve(
    query: str | None = None,
    region: str | None = None,
    limit: int = 10,
    threshold: float = 0.3,
    file_paths: list[str] | None = None,
    depth: int = 1,
) -> list[dict]:
    """Search the vault by meaning, keywords, and/or file context.

    Modes:
    - query only: hybrid semantic+keyword search (FAISS+FTS5+RRF)
    - file_paths only: graph traversal via backlinks (no model needed)
    - both: merged results with combined scoring

    Falls back to keyword-only search if the embedding model isn't ready yet.

    Args:
        query: Natural language search query
        region: Filter by region name
        limit: Max results (default 10, max 100)
        threshold: Min similarity (default 0.3)
        file_paths: Vault paths or filenames to find related notes via graph
        depth: Backlink graph hops 1-3 (default 1)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context

    needs_model = bool(query) and not (file_paths and not query)
    fts_only = False

    if needs_model and not state.embedder.is_ready:
        waited = state.embedder.wait_ready(timeout=WAIT_READY_TIMEOUT)
        if not waited or not state.embedder.is_ready:
            fts_only = True

    def _do():
        return handle_brain_retrieve(
            state.db, state.vectors, state.embedder,
            query=query, region=region, limit=limit, threshold=threshold,
            reranker=state.reranker, file_paths=file_paths, depth=depth,
            fts_only=fts_only,
        )

    try:
        result = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(None, _do),
            timeout=TOOL_TIMEOUT,
        )
        if fts_only and query and isinstance(result, list):
            for entry in result:
                if isinstance(entry, dict):
                    entry["fts_only"] = True
        return result
    except asyncio.TimeoutError:
        return [{"error": f"brain_retrieve timed out after {TOOL_TIMEOUT}s."}]

@mcp.tool()
async def brain_store(title: str, content: str, region: str | None = None, region_idx: int | None = None,
                tags: list[str] | None = None, folder: str = "") -> dict:
    """Save knowledge to the vault. Creates or updates a .md file. Auto-versioned.

    Args:
        title: Note title (becomes filename)
        content: Markdown content
        region: Brain region name (auto-detected if omitted)
        region_idx: Region index 0-11 (overrides name)
        tags: Additional tags (max 20)
        folder: Subfolder in vault (e.g. "02 Projekte")
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    if state.config.vault_path is None:
        return {"error": "No vault_path configured"}
    try:
        return await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                None,
                lambda: handle_brain_store(
                    state.db, state.vectors, state.embedder, state.config.vault_path,
                    title=title, content=content, region=region, region_idx=region_idx,
                    tags=tags, folder=folder, watcher=state.watcher,
                ),
            ),
            timeout=TOOL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return {"error": f"brain_store timed out after {TOOL_TIMEOUT}s."}

@mcp.tool()
async def brain_related(title: str | None = None, path: str | None = None, limit: int = 10) -> list[dict] | dict:
    """Find notes connected to a specific note via backlinks and semantic similarity.

    Args:
        title: Note title to find relations for
        path: Note path (alternative to title)
        limit: Max results (default 10, max 100)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context

    def _do():
        if not state.embedder.wait_ready(timeout=WAIT_READY_TIMEOUT):
            return {"error": f"Embedding model timed out ({WAIT_READY_TIMEOUT}s). Use brain_recent instead."}
        if not state.embedder.is_ready:
            return {"error": "Embedding model failed to load."}
        return handle_brain_related(state.db, state.vectors, state.embedder, title=title, path=path, limit=limit)

    try:
        return await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(None, _do),
            timeout=TOOL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return {"error": f"brain_related timed out after {TOOL_TIMEOUT}s."}

# ---------------------------------------------------------------------------
# MERGED TOOLS
# ---------------------------------------------------------------------------

@mcp.tool()
def brain_classify(
    action: str = "classify",
    title: str | None = None,
    path: str | None = None,
    content: str | None = None,
    apply: bool = False,
    correct_region_idx: int | None = None,
    reason: str = "",
) -> dict:
    """Classify notes into brain regions. No API key needed — uses keyword rules.

    Actions:
    - "classify": Classify a single note (provide title, path, or content)
    - "reclassify": Batch-reclassify all Stammhirn notes (dry_run unless apply=True)
    - "feedback": Correct a misclassification (requires path + correct_region_idx)

    Args:
        action: "classify" (default), "reclassify", or "feedback"
        title: Note title
        path: Note path
        content: Note content (if not in DB)
        apply: Write changes to DB + file (default: false = dry run)
        correct_region_idx: Correct region 0-11 (feedback only)
        reason: Why this correction (feedback only)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context

    if action == "feedback":
        if not path:
            return {"error": "path required for feedback action"}
        if correct_region_idx is None:
            return {"error": "correct_region_idx required for feedback action"}
        return handle_brain_classify_feedback(
            state.db, state.config.vault_path, state.config.data_dir,
            path=path, correct_region_idx=correct_region_idx, reason=reason,
        )

    if action == "reclassify":
        result = handle_brain_classify(
            state.db, state.config.vault_path,
            batch=True, apply=apply,
        )
        result["dry_run"] = not apply
        return result

    return handle_brain_classify(
        state.db, state.config.vault_path,
        title=title, path=path, content=content, batch=False, apply=apply,
    )

@mcp.tool()
def brain_versions(
    action: str,
    path: str = "",
    version_id: int | None = None,
) -> dict | list[dict]:
    """Manage note version history.

    Actions:
    - "history": List versions of a note
    - "diff": Compare current note with a previous version
    - "rollback": Restore a note to a previous version

    Args:
        action: "history", "diff", or "rollback"
        path: Note path in the vault
        version_id: Version ID (required for diff/rollback)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context

    if not path:
        return {"error": "path is required"}

    if action == "history":
        return handle_brain_history(state.db, path=path)

    if action == "diff":
        if version_id is None:
            return {"error": "version_id required for diff"}
        return handle_brain_diff(state.db, path=path, version_id=version_id)

    if action == "rollback":
        if version_id is None:
            return {"error": "version_id required for rollback"}
        if state.config.vault_path is None:
            return {"error": "No vault_path configured"}
        return handle_brain_rollback(
            state.db, state.config.vault_path, path=path,
            version_id=version_id, watcher=state.watcher,
        )

    return {"error": f"Unknown action: {action}. Use 'history', 'diff', or 'rollback'."}
