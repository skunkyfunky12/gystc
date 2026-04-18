from __future__ import annotations
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from mcp.server.fastmcp import FastMCP
from brain_mcp.config import BrainConfig, load_config
from brain_mcp.indexer.embedder import SentenceTransformerBackend
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.recent import handle_brain_recent
from brain_mcp.tools.regions import handle_brain_regions
from brain_mcp.tools.retrieve import handle_brain_retrieve

@dataclass
class BrainState:
    config: BrainConfig
    db: BrainDB
    vectors: VectorStore
    embedder: object | None = None

@asynccontextmanager
async def brain_lifespan(server: FastMCP) -> AsyncIterator[BrainState]:
    config = load_config()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    db = BrainDB(config.db_path)
    vectors = VectorStore.load(config.index_path, dimension=384)
    print(f"Brain MCP Server started. Vault: {config.vault_path}", file=sys.stderr)
    print(f"DB: {config.db_path} | Index: {vectors.size} vectors", file=sys.stderr)
    embedder = SentenceTransformerBackend(config.model_name)
    try:
        yield BrainState(config=config, db=db, vectors=vectors, embedder=embedder)
    finally:
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
    if state.embedder is None:
        return [{"error": "Embedding model not available"}]
    return handle_brain_retrieve(state.db, state.vectors, state.embedder, query=query, region=region, limit=limit, threshold=threshold)
