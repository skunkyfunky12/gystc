from __future__ import annotations
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
from brain_mcp.tools.context import handle_brain_context
from brain_mcp.tools.classify_tool import handle_brain_classify, handle_brain_classify_feedback
from brain_mcp.tools.versioning import handle_brain_history, handle_brain_diff, handle_brain_rollback
from brain_mcp.indexer.reranker import CrossEncoderReRanker

@dataclass
class BrainState:
    config: BrainConfig
    db: BrainDB
    vectors: VectorStore
    embedder: SentenceTransformerBackend
    watcher: BrainWatcher | None = None
    reranker: CrossEncoderReRanker | None = None


def _index_vault(state: BrainState) -> None:
    """Scan vault, skip unchanged files, upsert notes, embed new/changed, build edges."""
    if state.config.vault_path is None or not state.config.vault_path.is_dir():
        return
    index_vault(state.db, state.vectors, state.embedder,
                state.config.vault_path, state.config.folder_to_region)


def _handle_file_change(state: BrainState, path: str, event_type: str) -> None:
    """Callback for the watcher: re-index a single file on create/modify/delete."""
    try:
        rel = str(Path(path).relative_to(state.config.vault_path)).replace("\\", "/")
    except ValueError:
        return  # File outside vault scope
    if event_type == "deleted":
        # Remove FAISS vector before deleting the note
        old_row = state.db.get_note_by_path(rel)
        if old_row and old_row["faiss_idx"] is not None:
            state.vectors.remove([old_row["faiss_idx"]])
        state.db.delete_note(rel)
        return

    # Use shared parse_note_file instead of reimplementing tag extraction
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
        # Only remove old vector after new one is established
        if old_faiss_idx is not None:
            state.vectors.remove([old_faiss_idx])
        print(f"Re-indexed: {rel}", file=sys.stderr)
    except Exception as exc:
        print(f"Embedding error for {rel}: {exc}", file=sys.stderr)


def _background_startup(state: BrainState, model_thread: threading.Thread) -> None:
    """Run model loading + indexing in background so MCP server starts instantly."""
    try:
        vault_exists = state.config.vault_path is not None and state.config.vault_path.is_dir()
        model_thread.join(timeout=120)
        if model_thread.is_alive():
            print("WARNING: Model still loading after 120s, skipping startup indexing.", file=sys.stderr)
            return
        if not state.embedder.is_ready:
            print("ERROR: Model thread finished but model not ready — _load() likely crashed. Check stderr above.", file=sys.stderr)
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
    print(f"Brain MCP Server started in {startup_ms}ms. Vault: {config.vault_path}", file=sys.stderr)
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
        print("Brain MCP Server stopped.", file=sys.stderr)

BRAIN_INSTRUCTIONS = """
You have access to a persistent knowledge vault — a long-term memory organized like a brain.
The vault contains notes (neurons) connected by backlinks (synapses), organized into 12 brain regions.

HOW TO USE YOUR BRAIN:
- When you need knowledge about a topic, technology, or project: call brain_retrieve with a natural language query.
- When you start working on a codebase or task: call brain_context with the file paths or a task description.
- When you learn something worth remembering: call brain_store to save it as a note.
- When you want to see what's been worked on recently: call brain_recent.
- When exploring connections between topics: call brain_related.

WHEN TO SEARCH PROACTIVELY:
- You encounter a project, tool, or concept that might have vault notes — search before answering from general knowledge.
- The user asks about a previous decision, architecture, or plan — the vault likely has it.
- You're about to suggest an approach — check if there's prior context in the vault first.
- You're unsure about project-specific conventions or history — the vault knows.

WHAT NOT TO DO:
- Don't wait for the user to tell you to search. If vault knowledge would improve your answer, search.
- Don't dump full vault notes into responses. Summarize and reference paths so the user can dig deeper.
- Don't store trivial or ephemeral information. The vault is for knowledge worth keeping.

THE 12 BRAIN REGIONS:
Each note belongs to a region based on its function (not topic):
  0 Praefrontaler Cortex — Architecture, decisions, planning
  1 Motorischer Cortex — API writes, actions, execution
  2 Sensorischer Cortex — Data intake, references, input
  3 Hippocampus — Memory, sessions, personal notes
  4 Kleinhirn — Precision algorithms
  5 Nucleus Accumbens — Subscriptions, pricing, rewards
  6 Broca-Areal — AI, prompts, agents, language
  7 Visueller Cortex — UI, themes, design
  8 Thalamus — Index, MOC, data relay
  9 Stammhirn — Config, infrastructure (default for unclassified)
 10 Basalganglien — Pipelines, ETL, background jobs
 11 Amygdala — Auth, team, social interaction

VERSION HISTORY:
Every note change is automatically versioned. Use brain_history, brain_diff, and brain_rollback
to explore and restore previous versions. The vault has no Git — this IS the version control.
""".strip()

mcp = FastMCP("GYSTC", lifespan=brain_lifespan, instructions=BRAIN_INSTRUCTIONS)

@mcp.tool()
def brain_recent(days: int = 7, region: str | None = None, limit: int = 20) -> list[dict]:
    """See what was recently added or changed in the vault. Use to catch up on recent work.

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
async def brain_retrieve(query: str, region: str | None = None, limit: int = 10, threshold: float = 0.3) -> list[dict]:
    """Search the vault by meaning AND keywords. Use proactively whenever vault knowledge could improve your answer.

    Combines semantic search (FAISS) with keyword search (FTS5) via Reciprocal Rank Fusion.
    Returns matching notes and chunks with snippets. Long notes are chunked at headings
    so results point to the exact relevant section.

    WHEN TO USE: Before answering questions about projects, decisions, architecture, tools,
    or anything that might be documented in the vault. Search first, then answer.

    Args:
        query: Natural language search query (what are you looking for?)
        region: Filter by brain region name (e.g. "Hippocampus")
        limit: Max results (default 10, max 100)
        threshold: Min similarity (default 0.3)
    """
    import asyncio
    state: BrainState = mcp.get_context().request_context.lifespan_context
    if not state.embedder.wait_ready(timeout=90):
        return [{"error": "Embedding model still loading. Use brain_status to check, or try brain_recent/brain_regions instead."}]
    return await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: handle_brain_retrieve(state.db, state.vectors, state.embedder, query=query, region=region, limit=limit, threshold=threshold, reranker=state.reranker),
    )

@mcp.tool()
async def brain_store(title: str, content: str, region: str | None = None, region_idx: int | None = None,
                tags: list[str] | None = None, folder: str = "") -> dict:
    """Save knowledge to the vault. Use when you learn something worth remembering long-term.

    Creates or updates a note as a .md file in the vault. Previous content is automatically
    versioned (use brain_history to see versions). The note is embedded and searchable immediately.

    WHEN TO USE: After discovering important decisions, architecture patterns, project context,
    or anything the user might need again in future sessions. Don't store trivial or ephemeral info.

    Args:
        title: Note title (becomes filename)
        content: Markdown content (use headings, links, and structure)
        region: Brain region name (auto-detected if omitted)
        region_idx: Region index 0-11 (overrides name)
        tags: Additional tags (max 20)
        folder: Subfolder in vault (e.g. "02 Projekte")
    """
    import asyncio
    state: BrainState = mcp.get_context().request_context.lifespan_context
    if state.config.vault_path is None:
        return {"error": "No vault_path configured"}
    return await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: handle_brain_store(
            state.db, state.vectors, state.embedder, state.config.vault_path,
            title=title, content=content, region=region, region_idx=region_idx,
            tags=tags, folder=folder, watcher=state.watcher,
        ),
    )

@mcp.tool()
async def brain_related(title: str | None = None, path: str | None = None, limit: int = 10) -> list[dict] | dict:
    """Explore connections from a specific note. Finds related notes via backlinks and semantic similarity.

    Use to discover related knowledge, follow thought chains, or understand how topics connect.

    Args:
        title: Note title to find relations for
        path: Note path (alternative to title)
        limit: Max results (default 10, max 100)
    """
    import asyncio
    state: BrainState = mcp.get_context().request_context.lifespan_context
    if not state.embedder.wait_ready(timeout=90):
        return {"error": "Embedding model still loading. Use brain_status to check."}
    return await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: handle_brain_related(state.db, state.vectors, state.embedder, title=title, path=path, limit=limit),
    )

@mcp.tool()
async def brain_context(file_paths: list[str] | None = None, task_description: str | None = None,
                  depth: int = 1, max_notes: int = 10) -> list[dict] | dict:
    """Load vault context relevant to your current work. Use when switching tasks or starting new work.

    Combines the files you're working on with semantic search and backlink graph traversal
    to find the most relevant vault notes. Better than brain_retrieve when you have specific
    files or a task description rather than a search query.

    WHEN TO USE: At the start of a task, when switching topics, or when you need
    broader context about the area you're working in.

    Args:
        file_paths: Files currently being edited (vault paths or project paths)
        task_description: What you are doing (natural language)
        depth: Backlink graph hops 1-3 (default 1, higher = broader context)
        max_notes: Max notes returned (default 10, max 100)
    """
    import asyncio
    state: BrainState = mcp.get_context().request_context.lifespan_context
    if task_description and not state.embedder.wait_ready(timeout=90):
        return {"error": "Embedding model still loading. Use brain_status to check, or try without task_description."}
    return await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: handle_brain_context(state.db, state.vectors, state.embedder,
                                     file_paths=file_paths, task_description=task_description,
                                     depth=depth, max_notes=max_notes),
    )

@mcp.tool()
async def brain_reindex(force: bool = False) -> dict:
    """Re-index the vault. Use after bulk edits, graphify exports, or when results seem stale.

    Args:
        force: Re-embed all notes even if unchanged (default: false)
    """
    import asyncio
    import time
    state: BrainState = mcp.get_context().request_context.lifespan_context
    if state.config.vault_path is None or not state.config.vault_path.is_dir():
        return {"error": "No vault_path configured or directory not found"}

    def _do_reindex():
        t0 = time.time()
        try:
            count = index_vault(state.db, state.vectors, state.embedder,
                                state.config.vault_path, state.config.folder_to_region, force=force)
        except Exception as exc:
            print(f"ERROR: Reindex failed: {exc}", file=sys.stderr)
            return {"error": f"Reindex failed during indexing: {exc}", "indexed": 0}
        save_warning = None
        try:
            state.vectors.save(state.config.index_path)
        except Exception as exc:
            save_warning = str(exc)
            print(f"ERROR: Failed to save index after reindex: {exc}", file=sys.stderr)
        total = len(state.db.get_all_notes())
        elapsed = round(time.time() - t0, 1)
        print(f"Reindex complete: {count} new/changed, {total} total in {elapsed}s", file=sys.stderr)
        result = {"indexed": count, "total": total, "elapsed_seconds": elapsed}
        if save_warning:
            result["warning"] = f"Index not persisted to disk: {save_warning}"
        return result

    return await asyncio.get_event_loop().run_in_executor(None, _do_reindex)

@mcp.tool()
def brain_reclassify(dry_run: bool = True) -> dict:
    """Re-classify notes stuck in Stammhirn (region 9) using the auto-classifier.

    Updates both the DB region_idx AND the #brain/ tag in the .md file so
    changes survive reindexing.

    Args:
        dry_run: If true, only show what would change without applying (default: true)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    result = handle_brain_classify(
        state.db, state.config.vault_path,
        batch=True, apply=not dry_run,
    )
    result["dry_run"] = dry_run
    return result

@mcp.tool()
def brain_enrich(graph_json_path: str, dry_run: bool = True) -> dict:
    """Import graphify graph.json edges into brain.db to enrich the knowledge graph.

    Matches graphify nodes to existing brain notes by title, then imports
    their relationships as typed edges (graphify:contains, graphify:calls, etc.).
    Old graphify edges are replaced on each import.

    Args:
        graph_json_path: Absolute path to graphify's graph.json output
        dry_run: If true, only show what would be imported (default: true)
    """
    from pathlib import Path as _Path
    from scripts.import_graphify import import_graphify

    state: BrainState = mcp.get_context().request_context.lifespan_context
    graph_path = _Path(graph_json_path)
    if not graph_path.exists():
        return {"error": f"File not found: {graph_json_path}"}

    result = import_graphify(graph_path, state.db, dry_run=dry_run)

    # Also return current edge type distribution
    if not dry_run:
        result["edge_types"] = state.db.get_edge_type_counts()

    return result

@mcp.tool()
def brain_status() -> dict:
    """Fast health check. Returns note/edge counts, model status, and region distribution.
    Does NOT require the embedding model — always responds instantly.
    """
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
        "reranker_loaded": state.reranker.is_ready if state.reranker else False,
        "reranker_enabled": state.reranker is not None,
        "vault_path": str(state.config.vault_path),
    }

@mcp.tool()
def brain_history(path: str) -> list[dict] | dict:
    """Show version history of a note. Use to see what changed and when.

    Args:
        path: Note path in the vault (e.g. "02 Projekte/My Note.md")
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    return handle_brain_history(state.db, path=path)

@mcp.tool()
def brain_diff(path: str, version_id: int) -> dict:
    """Show differences between the current note and a previous version.

    Args:
        path: Note path in the vault
        version_id: Version ID from brain_history
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    return handle_brain_diff(state.db, path=path, version_id=version_id)

@mcp.tool()
def brain_rollback(path: str, version_id: int) -> dict:
    """Restore a note to a previous version. Writes old content back to the .md file.
    The current content is saved as a new version before overwriting.

    Args:
        path: Note path in the vault
        version_id: Version ID from brain_history to restore
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    if state.config.vault_path is None:
        return {"error": "No vault_path configured"}
    return handle_brain_rollback(
        state.db, state.config.vault_path, path=path,
        version_id=version_id, watcher=state.watcher,
    )

@mcp.tool()
def brain_classify(
    title: str | None = None,
    path: str | None = None,
    content: str | None = None,
    batch: bool = False,
    apply: bool = False,
) -> dict:
    """Classify a note into a brain region, or batch-classify all Stammhirn notes.

    Uses the FUNKTION > FORM > THEMA decision tree — no LLM or API key needed.

    Args:
        title: Note title to classify
        path: Note path (alternative to title)
        content: Note content (if not in DB yet)
        batch: If true, classify all Stammhirn notes
        apply: If true, write classification to DB + file tag
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    return handle_brain_classify(
        state.db, state.config.vault_path,
        title=title, path=path, content=content, batch=batch, apply=apply,
    )

@mcp.tool()
def brain_classify_feedback(
    path: str,
    correct_region_idx: int,
    reason: str = "",
) -> dict:
    """Correct a misclassification. Updates DB + file tag. Logs for future improvement.

    Args:
        path: Note path in the vault
        correct_region_idx: The correct region index (0-11)
        reason: Why this correction is needed (for the log)
    """
    state: BrainState = mcp.get_context().request_context.lifespan_context
    return handle_brain_classify_feedback(
        state.db, state.config.vault_path, state.config.data_dir,
        path=path, correct_region_idx=correct_region_idx, reason=reason,
    )

@mcp.tool()
def brain_autolink(project_subfolder: str, dry_run: bool = True) -> dict:
    """Auto-link graphify notes to parent documentation.

    Reads _autolink.json from a project's graphify folder, then:
    1. Adds Dokumentation backlinks FROM graphify notes TO main docs
    2. Updates Code-Referenzen IN main docs with new graphify notes

    Args:
        project_subfolder: Vault subfolder (e.g. "02 Projekte/D2D-Scout")
        dry_run: If True, only show what would change without writing
    """
    import re as _re
    state: BrainState = mcp.get_context().request_context.lifespan_context
    vault = state.config.vault_path
    graphify_dir = vault / project_subfolder / "graphify"
    if not graphify_dir.resolve().is_relative_to(vault.resolve()):
        return {"error": "project_subfolder escapes vault directory"}
    config_path = graphify_dir / "_autolink.json"

    if not config_path.exists():
        return {"error": f"No _autolink.json in {graphify_dir}"}

    config = json.loads(config_path.read_text(encoding="utf-8"))
    rules = config.get("rules")
    if not rules:
        return {"error": "No 'rules' found in _autolink.json"}
    doc_folder = vault / config.get("doc_folder", "")
    if not doc_folder.resolve().is_relative_to(vault.resolve()):
        return {"error": "doc_folder escapes vault directory"}

    doc_notes: dict[str, list[str]] = {r["doc"]: [] for r in rules}
    notes_linked = 0
    refs_added = 0

    # Phase 1: Match graphify notes to docs
    for note_path in sorted(graphify_dir.glob("*.md")):
        if note_path.name.startswith("_"):
            continue
        note_name = note_path.stem
        # Read source_file from frontmatter
        text = note_path.read_text(encoding="utf-8")
        source = ""
        if text.startswith("---"):
            try:
                end = text.index("---", 3)
                for line in text[3:end].split("\n"):
                    if line.startswith("source_file:"):
                        source = line.split(":", 1)[1].strip().strip('"').strip("'")
            except ValueError:
                pass

        matched_doc = None
        for rule in rules:
            for pattern in rule["patterns"]:
                if pattern.lower() in note_name.lower() or pattern.lower() in source.lower():
                    matched_doc = rule["doc"]
                    break
            if matched_doc:
                break

        if matched_doc:
            doc_notes[matched_doc].append(note_name)
            link = f"[[{matched_doc}]]"
            if link not in text:
                notes_linked += 1
                if not dry_run:
                    doc_line = f"\n## Dokumentation\n- {link}\n"
                    lines = text.split("\n")
                    insert_idx = len(lines)
                    for i, line in enumerate(lines):
                        if line.startswith("#graphify/") or line.startswith("#brain/"):
                            insert_idx = i
                            break
                    lines.insert(insert_idx, doc_line.strip())
                    note_path.write_text("\n".join(lines), encoding="utf-8")

    # Phase 2: Update Code-Referenzen in main docs
    for doc_name, notes in doc_notes.items():
        if not notes:
            continue
        doc_path = doc_folder / f"{doc_name}.md"
        if not doc_path.exists():
            continue
        text = doc_path.read_text(encoding="utf-8")
        if "## Code-Referenzen" not in text:
            continue

        ref_section = text.split("## Code-Referenzen")[1].split("\n#")[0]
        existing = set(_re.findall(r"\[\[([^\]]+)\]\]", ref_section))

        new_notes = [n for n in notes if n not in existing and not n.startswith("_")]
        if new_notes and not dry_run:
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("#brain/") and i > 0:
                    for note in sorted(new_notes):
                        lines.insert(i, f"- [[{note}]]")
                        i += 1
                        refs_added += 1
                    break
            doc_path.write_text("\n".join(lines), encoding="utf-8")
        else:
            refs_added += len(new_notes)

    return {
        "dry_run": dry_run,
        "docs_with_matches": {k: len(v) for k, v in doc_notes.items() if v},
        "notes_needing_doc_link": notes_linked,
        "refs_to_add": refs_added,
        "total_graphify_notes": sum(1 for _ in graphify_dir.glob("*.md") if not _.name.startswith("_")),
    }

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
